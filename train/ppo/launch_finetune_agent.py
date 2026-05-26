#!/usr/bin/env python
"""Login-node submitter for ``finetune_agent_master.py``.

Submits ONE SLURM job that runs the master on a compute node. The
master in turn:
  - runs one fine-tune worker as a local subprocess on its own node,
  - submits up to N (default 5) more workers as separate SLURM jobs,
  - polls until all 15 phases of the source composite agent are
    fine-tuned,
  - packs the result into a new agent bundle under agent_finetuning/.

Source agent is READ-ONLY; everything new lands under a fresh dir.

Usage (from a login node):
    poetry run python train/ppo/launch_finetune_agent.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --recipe train/ppo/finetune_recipes/piston_1mm.yaml

    # Tune concurrency
    poetry run python train/ppo/launch_finetune_agent.py \\
        --source-agent agents/... --recipe train/ppo/finetune_recipes/... \\
        --max-concurrent-slurm 7   # 8 total

    # Inspect the sbatch script without submitting:
    poetry run python train/ppo/launch_finetune_agent.py \\
        --source-agent agents/... --recipe train/ppo/finetune_recipes/... \\
        --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, SLURM_ACCOUNT, SLURM_GRES, SLURM_PARTITION, SLURM_TIME,
)


def _build_master_sbatch(args, run_id: str, output_root: str) -> str:
    job_name = f"ftm-{run_id[-8:]}"
    log_root = os.path.join(output_root, "master")
    # Request as many GPUs as the master will fan out to via
    # CUDA_VISIBLE_DEVICES on its local node. With --gpus-per-node 1
    # this comes out to the same --gres=gpu as before.
    gres_str = f"gpu:{args.gpus_per_node}" if args.gpus_per_node > 1 else SLURM_GRES
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.master_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={gres_str}
        #SBATCH --output={log_root}/master-%j.out
        #SBATCH --error={log_root}/master-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_root}
        cd {HPC_WORKDIR}
        poetry run python -u train/ppo/finetune_agent_master.py \\
            --source-agent {args.source_agent} \\
            --recipe {args.recipe} \\
            --output-root {output_root} \\
            --max-concurrent-slurm {args.max_concurrent_slurm} \\
            --gpus-per-node {args.gpus_per_node} \\
            --poll-interval-s {args.poll_interval_s} \\
            --max-retries {args.max_retries} \\
            --slurm-time {args.slurm_time}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a finetune_agent_master.py orchestration "
                    "job to SLURM. Source agent is read-only.")
    parser.add_argument("--source-agent", required=True,
                        help="Source agent dir, e.g. "
                             "agents/agent_20260419T211137Z_e3b7.")
    parser.add_argument("--recipe", required=True,
                        help="Recipe YAML, e.g. "
                             "train/ppo/finetune_recipes/piston_1mm.yaml.")
    parser.add_argument("--output-root", default=None,
                        help="Output dir for the master + per-phase runs. "
                             "Default: agent_finetuning/"
                             "<src_basename>_<recipe>_<ts>/.")
    parser.add_argument("--max-concurrent-slurm", type=int, default=5,
                        help="Max sbatch worker jobs the master will "
                             "have in flight (default 5).")
    parser.add_argument("--gpus-per-node", type=int, default=1,
                        help="GPUs to request for the master sbatch (passed "
                             "to --gres=gpu:N) AND to fan local workers "
                             "across via CUDA_VISIBLE_DEVICES on the "
                             "master's node. Default 1. With N>1 the "
                             "master runs N local workers concurrently in "
                             "addition to the sbatch pool, so peak "
                             "concurrency = N local + max_concurrent_slurm. "
                             "Each sbatch worker still requests gpu:1 and "
                             "SLURM packs them onto multi-GPU nodes if the "
                             "partition allows shared allocation.")
    parser.add_argument("--poll-interval-s", type=float, default=30.0,
                        help="Master poll interval (seconds).")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Per-phase retry budget on worker failure.")
    parser.add_argument("--master-time", type=str, default=SLURM_TIME,
                        help=f"SLURM --time for the master job itself "
                             f"(default {SLURM_TIME}). Must be at least "
                             f"as long as the worker --time times the "
                             f"number of rounds you expect.")
    parser.add_argument("--slurm-time", type=str, default="24:00:00",
                        help="SLURM --time for worker jobs (default "
                             "24:00:00).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch script without submitting.")
    args = parser.parse_args()

    if not os.path.isdir(args.source_agent):
        print(f"ERROR: --source-agent not a directory: {args.source_agent}")
        sys.exit(1)
    if not os.path.isfile(args.recipe):
        print(f"ERROR: --recipe not a file: {args.recipe}")
        sys.exit(1)

    # Default output dir name: <source>_<recipe>_<timestamp>.
    if args.output_root is None:
        src_base = os.path.basename(os.path.normpath(args.source_agent))
        recipe_base = os.path.basename(args.recipe).rsplit(".", 1)[0]
        ts = int(time.time())
        args.output_root = os.path.join(
            "agent_finetuning", f"{src_base}__{recipe_base}__{ts}")

    # Read-only invariant: refuse to clobber the source agent.
    if os.path.abspath(args.output_root).startswith(
            os.path.abspath(args.source_agent)):
        print(f"ERROR: --output-root ({args.output_root}) cannot be "
              f"inside --source-agent ({args.source_agent}).")
        sys.exit(1)

    run_id = os.path.basename(args.output_root.rstrip("/"))
    os.makedirs(os.path.join(args.output_root, "master"), exist_ok=True)

    script = _build_master_sbatch(args, run_id, args.output_root)

    print(f"Source agent:    {args.source_agent}")
    print(f"Recipe:          {args.recipe}")
    print(f"Output root:     {args.output_root}")
    print(f"Concurrency:     {args.gpus_per_node} local "
          f"(one per local GPU) + {args.max_concurrent_slurm} sbatch "
          f"= {args.gpus_per_node + args.max_concurrent_slurm} max")
    print(f"Master GPUs:     {args.gpus_per_node} "
          f"(--gres=gpu:{args.gpus_per_node})")
    print(f"Master time:     {args.master_time}")
    print(f"Worker time:     {args.slurm_time}")
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
    print(f"Submitted master job {job_id}")
    print(f"Tail the master log:  tail -f {args.output_root}/master/master-*.out")
    print(f"Or the orchestrator:   tail -f {args.output_root}/log.txt")
    print(f"Monitor children:      squeue -u $USER | grep ft-{run_id[-8:]}")


if __name__ == "__main__":
    main()
