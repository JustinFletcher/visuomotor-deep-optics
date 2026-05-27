#!/usr/bin/env python
"""Per-phase launcher for the segment-PTT + Zernike-DM transfer-learning run.

For each of the 15 bootstrap phases, submits one sbatch job that
fine-tunes ``train_ppo_elf_segment_zernike_dm_transfer.py`` starting
from the corresponding source-agent phase checkpoint. The job grows
the policy head from 45 to 45 + n_zernike (default 32) and trains
under a random Kolmogorov atmosphere (r0 ~ U(0.10, 0.25) m by
default), so the resulting per-phase policies correct both segment
PTT misalignment AND the atmospheric wavefront via the DM in one
combined action vector.

Outputs land under <output_root>/phase_NN/, mirroring the
existing finetune output layout so the same packer / rollout tooling
can chew on the result. Each sbatch job is one node + one GPU
(no in-job fan-out); SLURM handles cross-phase parallelism. For
densely packing multiple phases on a single multi-GPU node, see the
launch_finetune_agent.py master/worker pattern -- this launcher
trades that density for substantial simplicity.

Usage from a login node::

    # First time -- 15 jobs, one per phase.
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7

    # Resume after a SLURM timeout / cluster outage. Re-runs only the
    # phases that have a prior latest.pt (others are submitted fresh).
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --resume \\
        --output-root /p/work/fletch/agent_finetuning/segment_dm_v2_...

    # Dry-run -- print the per-phase sbatch scripts without submitting.
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

from train.ppo.launch_static_dark_hole import (
    HPC_CODE_DIR, HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)
from train.ppo.finetune_sinfo import (
    query_sinfo_nodes, select_best_nodes)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py"
_RUN_PREFIX = "segment_dm_v2"


def _find_latest_checkpoint(phase_dir: str) -> Optional[str]:
    """Return the path to the most-recent ppo_optomech_*/checkpoints/
    latest.pt under ``phase_dir`` (the most recently modified one),
    or None when no prior run exists. Used by --resume to thread
    each phase's continuation through the existing training script's
    resume-in-place machinery."""
    if not os.path.isdir(phase_dir):
        return None
    candidates = []
    for run_dir in Path(phase_dir).glob("ppo_optomech_*"):
        ck = run_dir / "checkpoints" / "latest.pt"
        if ck.is_file():
            candidates.append(ck)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _build_sbatch(args, phase: int, source_ckpt: str,
                  output_dir: str, resume_ckpt: Optional[str],
                  seed: int) -> str:
    """One sbatch script per phase. Single node, single GPU. The
    training script's run_main + resume-in-place logic handles
    everything inside the job."""
    job_name = f"szdm-p{phase:02d}-{args.run_id[-6:]}"
    log_root = os.path.join(output_dir, "_logs")

    # Init-from vs resume-from: resume wins when present.
    ckpt_args = (
        f"--resume-from {resume_ckpt}" if resume_ckpt
        else f"--source-checkpoint {source_ckpt}")

    r0_args = (f"--r0-range {args.r0_range[0]} {args.r0_range[1]}"
               if args.r0_range else "")
    dm_scale_args = (f"--dm-action-scale {args.dm_action_scale}"
                     if args.dm_action_scale is not None else "")

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres=gpu:1
        #SBATCH --output={log_root}/sbatch-%j.out
        #SBATCH --error={log_root}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_root}
        cd {HPC_CODE_DIR}

        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            {ckpt_args} \\
            --phased-count {phase} \\
            --n-zernike {args.n_zernike} \\
            {dm_scale_args} \\
            {r0_args} \\
            --seed {seed} \\
            --run-dir {output_dir}
    """)


def _build_node_block_sbatch(args, node_name: str, gpu_count: int,
                             jobs: list[dict],
                             block_log_root: str) -> str:
    """One sbatch per node that fans K phases out across K GPUs via
    bash fork+wait + CUDA_VISIBLE_DEVICES, mirroring the bootstrap
    fine-tune master / launch_zernike_atm_transfer pattern.

    Each phase still runs the same train script with the same args;
    only the dispatch shape changes. Per-phase logs land under the
    phase's own dir (separate from the block-level slurm-%j logs).
    """
    phase_label = "-".join(f"p{j['phase']:02d}" for j in jobs)
    job_name = f"szdm-{node_name[-6:]}-{phase_label}-{args.run_id[-6:]}"

    r0_args = (f"--r0-range {args.r0_range[0]} {args.r0_range[1]}"
               if args.r0_range else "")
    dm_scale_args = (f"--dm-action-scale {args.dm_action_scale}"
                     if args.dm_action_scale is not None else "")

    # Build per-phase bash arrays for the inner loop.
    phases_str = " ".join(str(j["phase"]) for j in jobs)
    seeds_str = " ".join(str(j["seed"]) for j in jobs)
    src_str = " ".join(f'"{j["source_ckpt"]}"' for j in jobs)
    out_str = " ".join(f'"{j["output_dir"]}"' for j in jobs)
    # Per-phase resume ckpt -- empty string means "init from source".
    resume_str = " ".join(
        f'"{j["resume_ckpt"]}"' if j.get("resume_ckpt") else '""'
        for j in jobs)

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --nodelist={node_name}
        #SBATCH --gres=gpu:{gpu_count}
        #SBATCH --output={block_log_root}/sbatch-%j.out
        #SBATCH --error={block_log_root}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {block_log_root}
        cd {HPC_CODE_DIR}

        PHASES=({phases_str})
        SEEDS=({seeds_str})
        SOURCES=({src_str})
        OUTDIRS=({out_str})
        RESUME=({resume_str})

        for i in "${{!PHASES[@]}}"; do
            phase="${{PHASES[$i]}}"
            seed="${{SEEDS[$i]}}"
            src="${{SOURCES[$i]}}"
            outdir="${{OUTDIRS[$i]}}"
            resume="${{RESUME[$i]}}"
            mkdir -p "${{outdir}}/_logs"

            if [ -n "$resume" ]; then
                ckpt_flag="--resume-from $resume"
                echo "[launcher] phase $phase: resume from $resume"
            else
                ckpt_flag="--source-checkpoint $src"
                echo "[launcher] phase $phase: init from $src"
            fi

            CUDA_VISIBLE_DEVICES=$i \\
            poetry run python -u {_TRAIN_SCRIPT} \\
                --hpc \\
                ${{ckpt_flag}} \\
                --phased-count "$phase" \\
                --n-zernike {args.n_zernike} \\
                {dm_scale_args} \\
                {r0_args} \\
                --seed "$seed" \\
                --run-dir "$outdir" \\
                > "${{outdir}}/_logs/inner.out" \\
                2> "${{outdir}}/_logs/inner.err" &
        done

        wait
        echo "[launcher] node {node_name}: all phases finished"
    """)


def _distribute_phases_across_nodes(jobs: list[dict],
                                    nodes: list) -> list[tuple[object, list[dict]]]:
    """Round-robin assign jobs onto nodes, capped by each node's GPU
    count. Returns [(node, [job, ...]), ...] for every node that ends
    up with at least one job. Phases are assigned in given order so
    phase numbering stays roughly monotonic per node."""
    assignments = {n.name: {"node": n, "jobs": [],
                            "cap": n.gpu_count} for n in nodes}
    node_order = [n.name for n in nodes]
    cursor = 0
    for j in jobs:
        # Skip nodes that are already full; if all are full, drop the
        # remaining jobs (caller decides whether to error or warn).
        attempts = 0
        while attempts < len(node_order):
            name = node_order[cursor % len(node_order)]
            cursor += 1
            a = assignments[name]
            if len(a["jobs"]) < a["cap"]:
                a["jobs"].append(j)
                break
            attempts += 1
        else:
            return [(assignments[n]["node"], assignments[n]["jobs"])
                    for n in node_order if assignments[n]["jobs"]] + [
                (None, [j])]   # marker -- caller can detect overflow
    return [(assignments[n]["node"], assignments[n]["jobs"])
            for n in node_order if assignments[n]["jobs"]]


def main():
    p = argparse.ArgumentParser(
        description="Per-phase segment-PTT + Zernike-DM transfer launcher.")
    p.add_argument("--source-agent", required=True,
                   help="Source bootstrap agent dir, e.g. "
                        "agents/agent_20260419T211137Z_e3b7.")
    p.add_argument("--n-zernike", type=int, default=32,
                   help="Zernike modes for the DM head (default 32).")
    p.add_argument("--dm-action-scale", type=float, default=0.01,
                   help="Per-step DM delta cap. Default 0.01 matches "
                        "the dm_atmos training family.")
    p.add_argument("--r0-range", type=float, nargs=2, default=[0.10, 0.25],
                   metavar=("LOW", "HIGH"),
                   help="Atmosphere r0 sampling range in meters at "
                        "500 nm (default 0.10 0.25).")
    p.add_argument("--phases", type=int, nargs="+", default=None,
                   help="Subset of phase indices to (re)submit "
                        "(default: all 15 phases of the source agent).")
    p.add_argument("--output-root", type=str, default=None,
                   help="Output root dir. Default: $HPC_WORKDIR/"
                        "segment_dm_finetuning/<src_base>__<run_id>/. "
                        "Distinct from agent_finetuning/ (bootstrap "
                        "fine-tunes) and atmos_finetuning/ (atmos "
                        "Zernike transfer) so this experiment's runs "
                        "don't mingle with the other families' "
                        "output trees. On --resume this must point "
                        "at the existing run root.")
    p.add_argument("--run-id", type=str, default=None,
                   help=f"Unique run id (default: {_RUN_PREFIX}_<ts>).")
    p.add_argument("--slurm-time", type=str, default=SLURM_TIME,
                   help=f"SLURM --time per phase job (default {SLURM_TIME}).")
    p.add_argument("--nodes", type=int, default=None, metavar="N",
                   help="Total number of nodes to use. When set, the "
                        "launcher uses sinfo to pick the N best idle "
                        "GPU nodes and packs phases onto them via "
                        "bash fork+wait + CUDA_VISIBLE_DEVICES (one "
                        "sbatch per node, multiple phases per node). "
                        "Without this flag (default), submits one "
                        "1-GPU sbatch per phase and lets SLURM "
                        "schedule them.")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing per-phase run. For each "
                        "phase, finds the most-recent latest.pt under "
                        "<output-root>/phase_NN/ and passes it to the "
                        "training script as --resume-from (full model "
                        "+ optimizer + global_step resume). Phases "
                        "with no prior latest.pt fall back to init-"
                        "from-source so a partial cluster outage can "
                        "be recovered cleanly. --output-root must "
                        "point at the existing run root.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch scripts without submitting.")
    args = p.parse_args()

    if not os.path.isdir(args.source_agent):
        print(f"ERROR: --source-agent not a directory: {args.source_agent}")
        sys.exit(1)
    manifest_path = os.path.join(args.source_agent, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: missing source manifest: {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    src_phases = manifest.get("phases") or []
    if not src_phases:
        print(f"ERROR: source manifest has no phases: {manifest_path}")
        sys.exit(1)

    # Resolve which phases to (re)submit.
    all_phase_indices = [int(p["phase"]) for p in src_phases]
    if args.phases:
        target_indices = [i for i in args.phases if i in all_phase_indices]
        if len(target_indices) != len(args.phases):
            missing = set(args.phases) - set(all_phase_indices)
            print(f"ERROR: --phases {sorted(missing)} not in source manifest.")
            sys.exit(1)
    else:
        target_indices = all_phase_indices

    # Resolve run id + output root. Default output tree is a NEW
    # top-level sibling of agent_finetuning/ and atmos_finetuning/
    # so this experiment's runs are clearly distinct from the
    # bootstrap fine-tune and atmos Zernike transfer trees.
    args.run_id = args.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    src_base = os.path.basename(os.path.normpath(args.source_agent))
    output_root = (args.output_root
                   or os.path.join(HPC_WORKDIR, "segment_dm_finetuning",
                                   f"{src_base}__{args.run_id}"))

    if args.resume:
        if not os.path.isdir(output_root):
            print(f"ERROR: --resume requires --output-root to exist "
                  f"({output_root}).")
            sys.exit(1)

    os.makedirs(output_root, exist_ok=True)

    print(f"Source agent:    {args.source_agent}")
    print(f"Output root:     {output_root}")
    print(f"Run id:          {args.run_id}")
    print(f"Resume mode:     {args.resume}")
    print(f"n_zernike:       {args.n_zernike}")
    print(f"dm_action_scale: {args.dm_action_scale}")
    print(f"r0_range (m):    {args.r0_range}")
    print(f"Phases:          {target_indices}")
    print(f"SLURM time:      {args.slurm_time}")
    print()

    # Resolve per-phase job dicts up front (source path, output dir,
    # resume ckpt, seed). Used by both the per-phase and node-block
    # dispatch paths.
    jobs: list[dict] = []
    for idx in target_indices:
        src_meta = next(p for p in src_phases if int(p["phase"]) == idx)
        source_ckpt = os.path.join(args.source_agent, src_meta["bundle_path"])
        if not os.path.isfile(source_ckpt):
            print(f"  phase {idx:02d}: source checkpoint missing "
                  f"({source_ckpt}) -- skipping.")
            continue
        phase_dir = os.path.join(output_root, f"phase_{idx:02d}")
        os.makedirs(phase_dir, exist_ok=True)

        resume_ckpt = None
        if args.resume:
            resume_ckpt = _find_latest_checkpoint(phase_dir)
            if resume_ckpt:
                print(f"  phase {idx:02d}: resume from {resume_ckpt}")
            else:
                print(f"  phase {idx:02d}: no prior latest.pt; init from "
                      f"source {source_ckpt}")

        jobs.append({
            "phase": idx,
            "source_ckpt": source_ckpt,
            "output_dir": phase_dir,
            "resume_ckpt": resume_ckpt,
            "seed": secrets.randbelow(MAX_SEED),
        })

    if not jobs:
        print("No phases to submit.")
        return

    # --- Node-block dispatch ----------------------------------------
    if args.nodes is not None:
        print(f"\n[sinfo] picking {args.nodes} best idle GPU node(s) "
              f"to pack {len(jobs)} phase(s) onto...")
        try:
            all_nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
        except Exception as e:
            print(f"[sinfo] query failed: {e}; falling back to per-phase "
                  f"submission.")
            all_nodes = []
        candidates = select_best_nodes(
            all_nodes, min_gpus=1, allow_mix=False, exclude_self=False
        ) if all_nodes else []
        chosen = candidates[: args.nodes]
        if not chosen:
            print(f"[sinfo] no idle nodes; falling back to per-phase "
                  f"submission.")
        else:
            print(f"[sinfo] picked nodes:")
            for n in chosen:
                print(f"  - {n.name}  state={n.state}  gpus={n.gpu_count}")
            total_slots = sum(n.gpu_count for n in chosen)
            if len(jobs) > total_slots:
                print(f"  WARNING: {len(jobs)} phases > {total_slots} GPU "
                      f"slots across {len(chosen)} nodes; extras will "
                      f"be dropped. Either pass a larger --nodes or "
                      f"submit per-phase (omit --nodes).")
            assignments = _distribute_phases_across_nodes(jobs, chosen)
            # Check for overflow marker (None node entry at end).
            overflow = [j for n, jl in assignments
                        if n is None for j in jl]
            assignments = [(n, jl) for n, jl in assignments if n is not None]
            blocks_root = os.path.join(output_root, "_blocks")
            os.makedirs(blocks_root, exist_ok=True)
            submitted = []
            for bi, (node, jlist) in enumerate(assignments):
                block_log = os.path.join(
                    blocks_root,
                    f"block_{bi:02d}_{node.name}_"
                    + "-".join(f"p{j['phase']:02d}" for j in jlist))
                os.makedirs(block_log, exist_ok=True)
                script = _build_node_block_sbatch(
                    args, node.name, len(jlist), jlist, block_log)
                if args.dry_run:
                    print(f"\n--- node-block sbatch for {node.name} "
                          f"({len(jlist)} phases) ---")
                    print(textwrap.indent(script, "    "))
                    continue
                result = subprocess.run(
                    ["sbatch", "--parsable"],
                    input=script, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  block on {node.name}: sbatch FAILED -- "
                          f"{result.stderr.strip()}")
                    continue
                job_id = result.stdout.strip()
                submitted.append((node.name, job_id, jlist))
                print(f"  block on {node.name}: submitted job {job_id} "
                      f"(phases {[j['phase'] for j in jlist]})")
            if overflow:
                print(f"  {len(overflow)} phase(s) DID NOT FIT on "
                      f"--nodes={args.nodes}: "
                      f"{[j['phase'] for j in overflow]}")
            if args.dry_run:
                print(f"\nWould submit {len(assignments)} node-block "
                      f"sbatch(es) covering {sum(len(jl) for _, jl in assignments)} "
                      f"phases across {len(assignments)} node(s).")
                return
            print(f"\nSubmitted {len(submitted)} node-block(s) covering "
                  f"{sum(len(jl) for _, _, jl in submitted)} phase(s).")
            if submitted:
                print(f"Monitor:  squeue -u $USER | grep szdm-")
                print(f"Tail:     tail -f {output_root}/phase_*/_logs/inner.out")
                print(f"Per-node: tail -f {blocks_root}/block_*/sbatch-*.out")
            return

    # --- Per-phase dispatch (default) -------------------------------
    submitted = []
    for j in jobs:
        idx = j["phase"]
        script = _build_sbatch(args, idx, j["source_ckpt"], j["output_dir"],
                               j["resume_ckpt"], j["seed"])

        if args.dry_run:
            print(f"--- phase {idx:02d} sbatch script ---")
            print(textwrap.indent(script, "    "))
            continue

        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  phase {idx:02d}: sbatch FAILED -- "
                  f"{result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        submitted.append((idx, job_id, j["output_dir"]))
        print(f"  phase {idx:02d}: submitted job {job_id} -> {j['output_dir']}")

    if args.dry_run:
        print(f"\nWould submit {len(jobs)} job(s) under {output_root}")
        return

    print(f"\nSubmitted {len(submitted)}/{len(jobs)} job(s).")
    if submitted:
        print(f"Monitor:  squeue -u $USER | grep szdm-")
        print(f"Tail:     tail -f {output_root}/phase_*/_logs/sbatch-*.out")


if __name__ == "__main__":
    main()
