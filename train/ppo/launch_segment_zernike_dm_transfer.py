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
import shlex
import subprocess
import sys


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py"


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
