#!/usr/bin/env python
"""Launch the n_zernike transfer sweep on ONE node, all GPUs.

Submits a single sbatch job that pins to one node (auto-picked from
sinfo for the most GPUs, or --node), requests --gres=gpu:N for that
node's full GPU count, and inside the sbatch body forks one Python
worker per --n-zernike target with CUDA_VISIBLE_DEVICES set. No
external orchestrator or polling -- this is a one-shot fork+wait.

The default n_zernike grid is ``[32, 64, 128, 256]``. The number of
GPUs requested matches that grid length so each worker gets its own
GPU; if your node has more GPUs the extras sit idle, if fewer the
launcher errors before submission so you don't accidentally
serialize.

Each worker runs train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_
transfer.py with --init-from <source_checkpoint> --n-zernike <N>,
landing in its own subdir under the run root. Stdout/stderr per
worker are captured separately.

Usage:
    # auto-pick the best idle node, default grid [32, 64, 128, 256]
    poetry run python train/ppo/launch_zernike_atm_transfer.py \\
        --source-checkpoint atmos_models/r0_25cm_seed.../checkpoints/latest.pt

    # custom grid
    poetry run python train/ppo/launch_zernike_atm_transfer.py \\
        --source-checkpoint .../latest.pt \\
        --n-zernike 16 32 64

    # pin to a specific node
    poetry run python train/ppo/launch_zernike_atm_transfer.py \\
        --source-checkpoint .../latest.pt \\
        --node make0123

    # dry-run -- print the sbatch script and exit
    poetry run python train/ppo/launch_zernike_atm_transfer.py \\
        --source-checkpoint .../latest.pt --dry-run
"""
from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time
from typing import Optional

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)
from train.ppo.finetune_sinfo import (
    query_sinfo_nodes, select_best_nodes)


_DEFAULT_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_transfer.py"
_RUN_PREFIX = "zernike_atm_transfer"
_DEFAULT_N_ZERNIKE = [32, 64, 128, 256]
# --n-zernike is only forwarded to the worker when the chosen train
# script accepts it. The basis-change transfer script does; the
# absolute-control transfer script does not (same n_zernike as the
# source by design). We detect by name -- crude but explicit.
_TRAIN_SCRIPTS_WITHOUT_N_ZERNIKE = {
    "train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_absolute.py",
}
# Per-experiment short tag that goes into the run id so two different
# transfer experiments (e.g. basis-size change vs incremental->
# absolute) land in distinguishable dirs under atmos_finetuning/.
# Override per-invocation with --experiment-tag. Falls back to a
# best-effort derivation from the script filename.
_SCRIPT_TAG_MAP = {
    "train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_transfer.py": "basis",
    "train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_absolute.py": "absolute",
}


def _derive_experiment_tag(train_script: str) -> str:
    """Map a training script path to a short identifying tag for the
    run id. Known scripts are looked up in _SCRIPT_TAG_MAP; unknown
    ones fall back to stripping the long shared prefix and the .py
    suffix from the basename."""
    if train_script in _SCRIPT_TAG_MAP:
        return _SCRIPT_TAG_MAP[train_script]
    basename = os.path.basename(train_script)
    for prefix in ("train_ppo_elf_dm_strehl_only_zernike_atm_",
                   "train_ppo_"):
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
            break
    return basename.removesuffix(".py") or "unknown"


def _build_sbatch_script(args, run_id: str, output_root: str,
                         node_name: Optional[str], n_workers: int,
                         per_worker_seeds: list[int],
                         train_script: str,
                         resume: bool = False) -> str:
    """One sbatch job: spawn one Python worker per --n-zernike via
    CUDA_VISIBLE_DEVICES, then wait. bash fork+wait keeps the launcher
    simple and avoids needing a separate node-block runner module.

    When ``resume`` is True, the per-worker invocation looks up its
    most-recent ``ppo_optomech_*/checkpoints/latest.pt`` inside the
    worker's existing ``out_dir`` and passes it via ``--resume-from``
    instead of ``--source-checkpoint`` so training continues from the
    prior model + optimizer + global_step. Workers without any prior
    checkpoint fall back to init-from-source (same as a fresh run).
    """
    job_name = f"ztrn-{run_id[-8:]}"
    log_root = os.path.join(output_root, "_logs")
    gres = f"gpu:{n_workers}"
    nodelist = (f"#SBATCH --nodelist={node_name}\n        "
                if node_name else "")

    # Build the per-worker invocations as a bash array so the body of
    # the sbatch is one tidy loop.
    n_zernike_str = " ".join(str(n) for n in args.n_zernike)
    seeds_str = " ".join(str(s) for s in per_worker_seeds)
    r0_flag = f" --r0 {args.r0}" if args.r0 is not None else ""
    # The basis-change transfer script accepts --n-zernike per worker
    # (each worker trains a different basis size). The absolute-control
    # transfer script does NOT (same n_zernike as the source by
    # design), so we drop the flag when targeting that script and let
    # the source's basis size flow through.
    pass_n_zernike = train_script not in _TRAIN_SCRIPTS_WITHOUT_N_ZERNIKE
    n_zernike_flag_template = (
        ' --n-zernike "${n}"' if pass_n_zernike else "")

    # Resume-mode shell snippet: per worker, look up the most-recent
    # latest.pt under the worker's out_dir; if found, use --resume-from
    # to continue training. Otherwise fall back to the source ckpt.
    if resume:
        ckpt_block = (
            '    resume_ckpt=$(ls -t "${out_dir}"/ppo_optomech_*'
            '/checkpoints/latest.pt 2>/dev/null | head -1)\n'
            '    if [ -n "$resume_ckpt" ]; then\n'
            '        echo "[launcher]   resume from: $resume_ckpt"\n'
            '        ckpt_flag="--resume-from $resume_ckpt"\n'
            '    else\n'
            '        echo "[launcher]   no prior latest.pt; '
            'init from source"\n'
            '        ckpt_flag="--source-checkpoint $SOURCE_CKPT"\n'
            '    fi')
        ckpt_invocation = '${ckpt_flag}'
    else:
        ckpt_block = ''
        ckpt_invocation = '--source-checkpoint "${SOURCE_CKPT}"'

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        {nodelist}#SBATCH --gres={gres}
        #SBATCH --output={log_root}/sbatch-%j.out
        #SBATCH --error={log_root}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_root}
        cd {HPC_WORKDIR}

        N_ZERNIKE=({n_zernike_str})
        SEEDS=({seeds_str})
        SOURCE_CKPT="{args.source_checkpoint}"
        OUTPUT_ROOT="{output_root}"

        # Fork one worker per GPU slot. Each pins to CUDA_VISIBLE_DEVICES=i,
        # writes its run dir under OUTPUT_ROOT/n<NN>_seed<seed>/, and runs
        # the transfer training script. stdout/stderr land per-worker.
        for i in "${{!N_ZERNIKE[@]}}"; do
            n="${{N_ZERNIKE[$i]}}"
            seed="${{SEEDS[$i]}}"
            label="n${{n}}_seed${{seed}}"
            out_dir="${{OUTPUT_ROOT}}/${{label}}"
            mkdir -p "${{out_dir}}"
            echo "[launcher] starting worker $i: n_zernike=$n  "\\
                 "GPU=$i  seed=$seed  -> $out_dir"
        {ckpt_block}
            CUDA_VISIBLE_DEVICES=$i poetry run python -u {train_script} \\
                --hpc \\
                {ckpt_invocation}{n_zernike_flag_template} \\
                --seed "${{seed}}"{r0_flag} \\
                --run-dir "${{out_dir}}" \\
                > "${{out_dir}}/stdout.log" 2> "${{out_dir}}/stderr.log" &
        done

        wait
        echo "[launcher] all {n_workers} workers finished"
    """)


def main():
    p = argparse.ArgumentParser(
        description="Single-node n_zernike transfer-learning sweep.")
    p.add_argument("--source-checkpoint", required=True,
                   help="Path to the source atmospheric DM model "
                        "checkpoint (e.g. atmos_models/r0_25cm_seed.../"
                        "checkpoints/latest.pt).")
    p.add_argument("--train-script", type=str, default=_DEFAULT_TRAIN_SCRIPT,
                   help=f"Path to the training script to launch per "
                        f"worker (default: {_DEFAULT_TRAIN_SCRIPT}). "
                        "Use train/ppo/train_ppo_elf_dm_strehl_only_"
                        "zernike_atm_absolute.py for the same-arch "
                        "incremental->absolute transfer.")
    p.add_argument("--n-zernike", type=int, nargs="+",
                   default=_DEFAULT_N_ZERNIKE,
                   help="Target Zernike basis sizes (default: "
                        f"{_DEFAULT_N_ZERNIKE}). Each is a separate "
                        "worker pinned to its own GPU. For training "
                        "scripts that don't accept --n-zernike "
                        "(e.g. the absolute-control transfer), the "
                        "list is still used to size the worker count "
                        "but the flag itself is dropped.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="One seed per --n-zernike value (lengths must "
                        "match). Default: fresh random seed per worker.")
    p.add_argument("--r0", type=float, default=None,
                   help="Override atmosphere r0_total_m (meters at 500 nm) "
                        "for every worker. Default: use the parent script's "
                        "default (0.12 m, Teide-typical median). When "
                        "fine-tuning a source model trained at a different "
                        "r0 you usually want this set to match the source's "
                        "training r0.")
    p.add_argument("--node", type=str, default=None,
                   help="Pin the sbatch to a specific node. Default: "
                        "sinfo-pick the biggest idle GPU node with "
                        ">= len(--n-zernike) GPUs. Pass 'any' to let "
                        "SLURM choose without --nodelist.")
    p.add_argument("--run-id", type=str, default=None,
                   help=f"Unique run id (default: {_RUN_PREFIX}_"
                        "<experiment-tag>_<ts>).")
    p.add_argument("--experiment-tag", type=str, default=None,
                   help="Short tag baked into the run id to keep "
                        "different transfer experiments (basis-size vs "
                        "absolute-control vs ...) in distinguishable "
                        "dirs under atmos_finetuning/. Default: "
                        "auto-derived from --train-script "
                        "(basis | absolute | <script basename>).")
    p.add_argument("--output-root", type=str, default=None,
                   help="Output dir. Default: atmos_finetuning/"
                        "<run_id>/. Each n_zernike worker lands in "
                        "<run_id>/n<NN>_seed<seed>/.")
    p.add_argument("--slurm-time", type=str, default=SLURM_TIME,
                   help=f"SLURM --time for the sbatch (default {SLURM_TIME}).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing atmos-transfer job rooted "
                        "at --output-root. For each worker subdir "
                        "<output-root>/n<N>_seed<seed>/, the sbatch "
                        "body looks up the most recent "
                        "ppo_optomech_*/checkpoints/latest.pt and "
                        "passes it via --resume-from (full model + "
                        "optimizer + global_step resume) instead of "
                        "--source-checkpoint. Workers with no prior "
                        "checkpoint fall back to init-from-source. "
                        "Pass the same --n-zernike and --seeds as the "
                        "original submission so the worker subdirs "
                        "line up.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch script without submitting.")
    args = p.parse_args()

    n_workers = len(args.n_zernike)
    if n_workers < 1:
        print("Error: --n-zernike must have at least one value")
        sys.exit(1)

    if args.seeds is not None:
        if len(args.seeds) != n_workers:
            print(f"Error: --seeds must have {n_workers} values to "
                  f"match --n-zernike ({n_workers} values)")
            sys.exit(1)
        per_worker_seeds = [int(s) for s in args.seeds]
    else:
        per_worker_seeds = [secrets.randbelow(MAX_SEED)
                            for _ in range(n_workers)]

    if not os.path.isfile(args.source_checkpoint):
        print(f"Error: --source-checkpoint not a file: "
              f"{args.source_checkpoint}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Pick the node. We require >= n_workers GPUs so every worker
    # gets its own GPU (no serialization).
    # ----------------------------------------------------------------
    chosen_node: Optional[str] = None
    if args.node == "any":
        pass
    elif args.node:
        chosen_node = args.node
    else:
        nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
        candidates = select_best_nodes(
            nodes, min_gpus=n_workers, allow_mix=False,
            exclude_self=False)
        if candidates:
            top = candidates[0]
            chosen_node = top.name
            print(f"[sinfo] picked node {top.name} "
                  f"(state={top.state}, gpu_count={top.gpu_count})")
        else:
            print(f"[sinfo] no idle nodes with >= {n_workers} GPUs; "
                  f"letting SLURM choose without --nodelist")

    if args.resume:
        if not args.output_root:
            print("ERROR: --resume requires --output-root pointing at "
                  "an existing atmos-transfer job dir.")
            sys.exit(1)
        if not os.path.isdir(args.output_root):
            print(f"ERROR: --resume --output-root ({args.output_root}) "
                  f"does not exist or is not a directory.")
            sys.exit(1)

    experiment_tag = (args.experiment_tag
                      or _derive_experiment_tag(args.train_script))
    run_id = args.run_id or (
        f"{_RUN_PREFIX}_{experiment_tag}_{int(time.time())}")
    # Default output root: absolute path under HPC_WORKDIR (work fs),
    # so dirs land in the right place regardless of where the launcher
    # is invoked from. Pass --output-root explicitly to override.
    output_root = args.output_root or os.path.join(
        HPC_WORKDIR, "atmos_finetuning", run_id)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "_logs"), exist_ok=True)

    script = _build_sbatch_script(
        args, run_id, output_root,
        node_name=chosen_node,
        n_workers=n_workers,
        per_worker_seeds=per_worker_seeds,
        train_script=args.train_script,
        resume=args.resume)

    print(f"Run id:             {run_id}")
    print(f"Experiment tag:     {experiment_tag}")
    print(f"Resume mode:        {args.resume}")
    print(f"Source checkpoint:  {args.source_checkpoint}")
    print(f"Training script:    {args.train_script}")
    print(f"Output root:        {output_root}")
    print(f"Atmosphere r0:      "
          f"{args.r0 if args.r0 is not None else '(parent default 0.12 m)'}")
    print(f"Node:               {chosen_node or '(SLURM chooses)'}")
    print(f"GPUs requested:     {n_workers} (--gres=gpu:{n_workers})")
    print(f"Workers:")
    for i, (n, s) in enumerate(zip(args.n_zernike, per_worker_seeds)):
        print(f"  worker {i}: n_zernike={n:4d}  seed={s:12d}  "
              f"GPU={i}  -> {output_root}/n{n}_seed{s}/")
    print(f"SLURM time:         {args.slurm_time}")
    print()

    if args.dry_run:
        print("Would submit:")
        print(textwrap.indent(script, "    "))
        return

    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch FAILED: {result.stderr.strip()}")
        sys.exit(1)
    job_id = result.stdout.strip()
    print(f"Submitted job {job_id}")
    print(f"Tail sbatch log: tail -f {output_root}/_logs/sbatch-*.out")
    print(f"Tail per-worker: tail -f {output_root}/n*/stdout.log")


if __name__ == "__main__":
    main()
