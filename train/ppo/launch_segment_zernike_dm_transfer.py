#!/usr/bin/env python
"""Thin wrapper around ``launch_finetune_agent.py`` that submits the
segment-PTT + Zernike-DM transfer training pipeline through the
existing fine-tune master/worker scheduling machinery.

Why a wrapper instead of a separate launcher? The fine-tune
master/worker (launch_finetune_agent -> finetune_agent_master ->
finetune_node_block_worker -> finetune_agent_worker) already does
sinfo-driven node selection, multi-node fan-out with per-node phase
packing, per-phase status sentinels, retry handling, --resume mode,
etc. Reinventing that for each new transfer pipeline is waste; the
master now accepts ``--train-script`` + ``--extra-args`` so any
training script with a CLI matching the fine-tune contract
(``--phased-count``, ``--source-checkpoint`` / ``--resume-from``,
``--run-dir``, ``--hpc``) can ride the existing scheduling.

This wrapper just sets the right --train-script (the segment-PTT +
Zernike-DM transfer training script) and a sensible --extra-args
string carrying the per-experiment knobs (n_zernike, dm_action_scale,
r0_range). Everything else (--source-agent, --max-nodes, --resume,
--max-phases-per-node, --output-root) is forwarded to the underlying
launcher.

Usage::

    # Default: 6-node spread, 32-mode DM head, 10-25 cm r0 range
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --max-nodes 6

    # Resume after a SLURM timeout
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --resume \\
        --output-root /p/work/fletch/segment_dm_finetuning/<run>
"""
from __future__ import annotations

import argparse
import os
import secrets
import shlex
import subprocess
import sys
import textwrap
import time


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py"
_MAX_SEED = 2**31 - 1


def _seed_sweep_sbatch(node_name: str, phase_idx: int, seed: int,
                       src_ckpt: str, out_dir: str, log_dir: str,
                       extra_args: str, slurm_time: str,
                       account: str, partition: str,
                       code_dir: str) -> str:
    """One sbatch script per (node, phase, seed) for seed-sweep mode.

    Pinned to the chosen node, 1 GPU, runs the train script directly
    (no master, no worker, no node-block fan-out). The training
    script's own resume-in-place + status logging handles the rest."""
    job_name = f"sweep-p{phase_idx:02d}-s{seed % 100000:05d}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={slurm_time}
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        #SBATCH --nodes=1
        #SBATCH --nodelist={node_name}
        #SBATCH --gres=gpu:1
        #SBATCH --output={log_dir}/sbatch-%j.out
        #SBATCH --error={log_dir}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_dir}
        cd {code_dir}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --phased-count {phase_idx} \\
            --source-checkpoint {src_ckpt} \\
            --seed {seed} \\
            --run-dir {out_dir} \\
            {extra_args}
    """)


def _run_seed_sweep(args, extra_args: str) -> None:
    """Submit N independent 1-GPU sbatches over the requested phases.

    No master, no retries, no status sentinels -- just fire N jobs
    at N idle nodes, round-robin across the phase list, each with a
    fresh seed. The user monitors via squeue and evaluates the
    results with rollout_per_phase.py.
    """
    # Imports localised so the module loads on machines that don't
    # have the launch_static_dark_hole constants available (e.g.
    # workstation lint checks). On HPC + login nodes these are
    # always present.
    from train.ppo.launch_static_dark_hole import (
        HPC_CODE_DIR, HPC_WORKDIR, SLURM_ACCOUNT, SLURM_PARTITION,
        SLURM_TIME)
    from train.ppo.finetune_sinfo import (
        query_sinfo_nodes, select_best_nodes)

    N = int(args.seed_sweep)
    if N < 1:
        print("ERROR: --seed-sweep must be >= 1")
        sys.exit(1)

    # Resolve phase list.
    if args.phases:
        phases = [int(x) for x in args.phases.split(",") if x.strip()]
    else:
        phases = list(range(15))
    if not phases:
        print("ERROR: empty --phases list")
        sys.exit(1)

    # Resolve output root.
    if args.output_root:
        output_root = args.output_root
    else:
        src_base = os.path.basename(os.path.normpath(args.source_agent))
        ts = int(time.time())
        output_root = os.path.join(
            HPC_WORKDIR, "segment_dm_finetuning",
            f"{src_base}__seed_sweep__{ts}")
    os.makedirs(output_root, exist_ok=True)
    blocks_log_root = os.path.join(output_root, "_seed_sweep_logs")
    os.makedirs(blocks_log_root, exist_ok=True)

    # Verify source checkpoints exist for the requested phases.
    src_dir = os.path.abspath(args.source_agent)
    src_ckpts: dict[int, str] = {}
    for ph in phases:
        ck = os.path.join(src_dir, "checkpoints", f"phase_{ph:02d}.pt")
        if not os.path.isfile(ck):
            print(f"ERROR: source checkpoint missing for phase {ph}: {ck}")
            sys.exit(1)
        src_ckpts[ph] = ck

    # Pick N idle GPU nodes via sinfo. If sinfo returns fewer than N,
    # the unmatched slots just don't get submitted (warn but
    # continue -- user can resubmit when more nodes free up).
    print(f"[sinfo] requesting {N} idle node(s) for seed-sweep "
          f"({len(phases)} phases, round-robin)...")
    try:
        all_nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
    except Exception as e:
        print(f"[sinfo] query failed: {e}; aborting.")
        sys.exit(1)
    candidates = select_best_nodes(
        all_nodes, min_gpus=1, allow_mix=False, exclude_self=False)
    nodes_for_sweep = candidates[:N]
    if len(nodes_for_sweep) < N:
        print(f"[sinfo] WARNING: only {len(nodes_for_sweep)} idle "
              f"node(s) available, asked for {N}. Submitting "
              f"{len(nodes_for_sweep)} jobs; resubmit later for the "
              f"rest.")

    # Build per-slot (node, phase, seed) plan.
    slots = []
    for i, node in enumerate(nodes_for_sweep):
        phase_idx = phases[i % len(phases)]
        seed = secrets.randbelow(_MAX_SEED)
        per_phase_dir = os.path.join(
            output_root, f"phase_{phase_idx:02d}", f"seed_{seed}")
        slots.append({
            "slot": i,
            "node": node.name,
            "phase": phase_idx,
            "seed": seed,
            "out_dir": per_phase_dir,
            "src_ckpt": src_ckpts[phase_idx],
        })

    # Header.
    print()
    print(f"=== Seed-sweep plan ===")
    print(f"Source agent:   {args.source_agent}")
    print(f"Output root:    {output_root}")
    print(f"Phases:         {phases}")
    print(f"Slots:          {len(slots)} (one per node, one GPU each)")
    for s in slots:
        print(f"  slot {s['slot']:2d}: node={s['node']:8s}  "
              f"phase={s['phase']:02d}  seed={s['seed']:>10d}  "
              f"-> {os.path.relpath(s['out_dir'], output_root)}")
    print()

    if args.dry_run:
        for s in slots[:2]:
            print(f"--- sbatch for slot {s['slot']} (sample) ---")
            print(_seed_sweep_sbatch(
                node_name=s["node"], phase_idx=s["phase"],
                seed=s["seed"], src_ckpt=s["src_ckpt"],
                out_dir=s["out_dir"],
                log_dir=os.path.join(blocks_log_root,
                                     f"slot_{s['slot']:02d}"),
                extra_args=extra_args,
                slurm_time=(args.slurm_time or SLURM_TIME),
                account=SLURM_ACCOUNT, partition=SLURM_PARTITION,
                code_dir=HPC_CODE_DIR))
        print(f"\nWould submit {len(slots)} job(s).")
        return

    # Submit.
    submitted = []
    for s in slots:
        slot_log = os.path.join(blocks_log_root, f"slot_{s['slot']:02d}")
        os.makedirs(slot_log, exist_ok=True)
        os.makedirs(s["out_dir"], exist_ok=True)
        script = _seed_sweep_sbatch(
            node_name=s["node"], phase_idx=s["phase"], seed=s["seed"],
            src_ckpt=s["src_ckpt"], out_dir=s["out_dir"],
            log_dir=slot_log, extra_args=extra_args,
            slurm_time=(args.slurm_time or SLURM_TIME),
            account=SLURM_ACCOUNT, partition=SLURM_PARTITION,
            code_dir=HPC_CODE_DIR)
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  slot {s['slot']:2d} on {s['node']}: sbatch "
                  f"FAILED -- {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        submitted.append((s, job_id))
        print(f"  slot {s['slot']:2d} on {s['node']}: submitted "
              f"job {job_id} (phase {s['phase']:02d}, seed {s['seed']})")

    print(f"\nSubmitted {len(submitted)}/{len(slots)} job(s).")
    if submitted:
        print(f"Monitor:  squeue -u $USER | grep sweep-")
        print(f"Tail:     tail -f {output_root}/phase_*/seed_*/"
              f"ppo_optomech_*/training.log")
        print(f"Eval:     poetry run python "
              f"train/ppo/rollout_per_phase.py --source {output_root}")


def main():
    p = argparse.ArgumentParser(
        description="Segment-PTT + Zernike-DM transfer launcher "
                    "(thin wrapper around launch_finetune_agent.py).")
    p.add_argument("--source-agent", required=True,
                   help="Source bootstrap agent dir, e.g. "
                        "agents/agent_20260419T211137Z_e3b7.")
    p.add_argument("--max-nodes", type=int, default=6,
                   help="Total node budget (including the master's "
                        "own node). Default 6.")
    p.add_argument("--max-phases-per-node", type=int, default=None,
                   help="Override per-node phase packing. Default: "
                        "auto = ceil(num_phases / max_nodes) so "
                        "phases spread as thinly as the node budget "
                        "allows.")
    p.add_argument("--n-zernike", type=int, default=32)
    p.add_argument("--dm-action-scale", type=float, default=0.01)
    p.add_argument("--r0-range", type=float, nargs=2,
                   default=[0.10, 0.25],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--output-root", type=str, default=None,
                   help="Optional explicit output root. Default: "
                        "$HPC_WORKDIR/segment_dm_finetuning/"
                        "<src>__train_ppo_elf_segment_zernike_dm_"
                        "transfer__<ts>/ (set by the underlying "
                        "launcher when --train-script is given).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing run; requires "
                        "--output-root to point at the existing dir.")
    p.add_argument("--master-time", type=str, default=None,
                   help="SLURM --time for the master sbatch.")
    p.add_argument("--slurm-time", type=str, default=None,
                   help="SLURM --time for worker sbatches.")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Per-phase retry budget on worker failure. "
                        "Default 5 (vs the upstream default of 1) -- "
                        "trades a few wasted submissions for "
                        "resilience to transient OOM / GPFS / node-"
                        "health hiccups overnight.")
    p.add_argument("--phases", type=str, default=None,
                   metavar="LIST",
                   help="Comma-separated phase indices to schedule "
                        "(e.g. '1,4,11'). Default: all 15 phases. "
                        "Use to retrain only the failed phases from "
                        "a previous run -- pair with the "
                        "failed_phases output of rollout_per_phase.py.")
    p.add_argument("--seed-sweep", type=int, default=None, metavar="N",
                   help="Seed-sweep mode: bypass the master and submit "
                        "exactly N independent 1-GPU sbatch jobs, "
                        "round-robin'd across the --phases list, each "
                        "with a fresh random seed. Use to dedupe "
                        "flaky-training failure modes by trying N "
                        "different seeds in parallel. Example: "
                        "--phases 3 --seed-sweep 6 runs 6 fresh seeds "
                        "of phase 3 on 6 nodes simultaneously. "
                        "--phases 3,7 --seed-sweep 6 runs 3 seeds of "
                        "each on 6 nodes. Output per job lives under "
                        "<output-root>/phase_NN/seed_SSSS/. Skips the "
                        "master/worker chain entirely -- no retry, no "
                        "status sentinels, just N straightforward "
                        "sbatch submissions you monitor via squeue.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the launch_finetune_agent.py command "
                        "without invoking it.")
    args = p.parse_args()

    # Compose the --extra-args string forwarded to every worker.
    extra_args = (
        f"--n-zernike {args.n_zernike} "
        f"--dm-action-scale {args.dm_action_scale} "
        f"--r0-range {args.r0_range[0]} {args.r0_range[1]}"
    )

    if args.seed_sweep is not None:
        _run_seed_sweep(args, extra_args)
        return

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    cmd = [
        sys.executable, "-u",
        os.path.join(repo_root, "train", "ppo",
                     "launch_finetune_agent.py"),
        "--source-agent", args.source_agent,
        "--train-script", _TRAIN_SCRIPT,
        "--extra-args", extra_args,
        "--max-nodes", str(args.max_nodes),
    ]
    if args.max_phases_per_node is not None:
        cmd += ["--max-phases-per-node", str(args.max_phases_per_node)]
    if args.output_root:
        cmd += ["--output-root", args.output_root]
    if args.resume:
        cmd += ["--resume"]
    if args.master_time:
        cmd += ["--master-time", args.master_time]
    if args.slurm_time:
        cmd += ["--slurm-time", args.slurm_time]
    cmd += ["--max-retries", str(args.max_retries)]
    if args.phases:
        cmd += ["--phases", args.phases]
    if args.dry_run:
        cmd += ["--dry-run"]

    print("Wrapping launch_finetune_agent.py with:")
    print(" ", " ".join(shlex.quote(c) for c in cmd))
    print()
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
