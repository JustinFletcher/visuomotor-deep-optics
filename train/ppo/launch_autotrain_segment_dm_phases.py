#!/usr/bin/env python
"""Login-node submitter for ``autotrain_segment_dm_phases.py``.

Submits ONE SLURM job that runs the autotrainer on a compute node.
The autotrainer then drives all per-phase seed-sweeps from there
(submitting / cancelling worker sbatches the same way
finetune_agent_master.py does for the bootstrap finetune).

Matches the existing two-tier pattern:
  - login-node submitter (this file) -- thin, just forwards args
  - compute-node orchestrator (autotrain_segment_dm_phases.py) --
    long-running loop, polls squeue, evals checkpoints, merges
    winners

No long-running processes on the login node.

Usage (from a login node)::

    poetry run python train/ppo/launch_autotrain_segment_dm_phases.py \\
        --winners-agent /p/work/fletch/agents/segment_dm_agent_v2_winners \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --max-nodes 6
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
import time
from typing import Optional

from train.ppo.launch_static_dark_hole import (
    HPC_CODE_DIR, HPC_WORKDIR, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME)
from train.ppo.finetune_sinfo import (
    query_sinfo_nodes, select_best_nodes)


_AUTOTRAIN_SCRIPT = "train/ppo/autotrain_segment_dm_phases.py"


def _build_autotrain_sbatch(args, log_root: str,
                            node_name: Optional[str]) -> str:
    """One sbatch script for the long-running autotrainer.

    The autotrainer itself is single-process / single-node; the
    per-phase seed-sweep workers it spawns are separate sbatches
    that land on other nodes. Eval is CPU-only (50-step headless
    episodes), so the master node's GPU sits idle -- we still
    request --gres=gpu:1 because most partitions on this cluster
    require a GPU even for shell-only jobs.
    """
    job_name = f"autotrain-{args.run_id[-8:]}"
    nodelist_line = (f"#SBATCH --nodelist={node_name}\n        "
                     if node_name else "")

    # Forward every CLI knob that ``autotrain_segment_dm_phases.py``
    # accepts. Use shlex.quote on string fields so paths with
    # spaces survive the round-trip.
    forwards = [
        ("--winners-agent", args.winners_agent),
        ("--source-agent", args.source_agent),
        ("--max-nodes", str(args.max_nodes)),
        ("--max-steps-per-run", str(args.max_steps_per_run)),
        ("--eval-interval-s", str(args.eval_interval_s)),
        ("--eval-episodes", str(args.eval_episodes)),
        ("--eval-steps", str(args.eval_steps)),
        ("--strehl-threshold", str(args.strehl_threshold)),
        ("--n-zernike", str(args.n_zernike)),
        ("--dm-action-scale", str(args.dm_action_scale)),
        ("--slurm-time", args.slurm_time),
    ]
    if args.phases:
        forwards.append(("--phases", args.phases))
    if args.output_root:
        forwards.append(("--output-root", args.output_root))
    forward_str = " ".join(
        f"{k} {shlex.quote(v)}" for k, v in forwards)
    # --r0-range takes two floats; format separately.
    r0_str = (f"--r0-range {args.r0_range[0]} {args.r0_range[1]}")

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.master_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        {nodelist_line}#SBATCH --gres=gpu:1
        #SBATCH --output={log_root}/autotrain-%j.out
        #SBATCH --error={log_root}/autotrain-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_root}
        cd {HPC_CODE_DIR}
        poetry run python -u {_AUTOTRAIN_SCRIPT} \\
            {forward_str} {r0_str}
    """)


def main():
    p = argparse.ArgumentParser(
        description="Submit autotrain_segment_dm_phases.py as a "
                    "long-running compute-node job.")
    # Inputs forwarded to the compute-node autotrainer.
    p.add_argument("--winners-agent", required=True,
                   help="Writable agent dir produced by "
                        "pack_winner_phases.py. The autotrainer "
                        "fills in phases_missing one at a time.")
    p.add_argument("--source-agent", required=True,
                   help="Source bootstrap agent (read-only). Used "
                        "for per-phase --source-checkpoint init.")
    p.add_argument("--max-nodes", type=int, default=6,
                   help="Number of seeds run in parallel per "
                        "phase. Each gets its own node + 1 GPU.")
    p.add_argument("--max-steps-per-run", type=int,
                   default=50_000_000,
                   help="Per-seed total_timesteps cap (default 50M).")
    p.add_argument("--eval-interval-s", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--strehl-threshold", type=float, default=0.5)
    p.add_argument("--phases", type=str, default=None, metavar="LIST",
                   help="Override winners-agent's phases_missing "
                        "with an explicit list (e.g. '3,5,7').")
    p.add_argument("--n-zernike", type=int, default=32)
    p.add_argument("--dm-action-scale", type=float, default=0.01)
    p.add_argument("--r0-range", type=float, nargs=2,
                   default=[0.10, 0.25],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--slurm-time", type=str, default=SLURM_TIME,
                   help="SLURM --time for each spawned PER-SEED "
                        "training sbatch.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Per-run sweep output root.")
    # Master sbatch knobs.
    p.add_argument("--master-time", type=str, default="168:00:00",
                   help=f"SLURM --time for the autotrainer master "
                        f"sbatch (default 168:00:00 = 7 days; the "
                        f"loop is long-running and can wait through "
                        f"many seed sweeps).")
    p.add_argument("--master-node", type=str, default=None,
                   help="Pin the autotrainer to a specific node "
                        "(--nodelist). Default: pick best idle GPU "
                        "node via sinfo, or let SLURM choose if "
                        "sinfo finds nothing.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run id used in the autotrain sbatch's job "
                        "name + log dir. Default: ts-derived.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Where the autotrainer's sbatch-*.out|err "
                        "land. Default: $HPC_WORKDIR/"
                        "segment_dm_finetuning/_autotrain_logs/"
                        "<run-id>/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch script without submitting.")
    args = p.parse_args()

    # Resolve run id + log dir.
    args.run_id = args.run_id or f"autotrain_{int(time.time())}"
    if args.log_dir is None:
        args.log_dir = os.path.join(
            HPC_WORKDIR, "segment_dm_finetuning",
            "_autotrain_logs", args.run_id)
    os.makedirs(args.log_dir, exist_ok=True)

    # Pick master node (auto sinfo unless --master-node was set).
    chosen_node: Optional[str] = None
    if args.master_node:
        chosen_node = args.master_node
    else:
        try:
            all_nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
            best = select_best_nodes(
                all_nodes, min_gpus=1, allow_mix=False,
                exclude_self=False)
            if best:
                chosen_node = best[0].name
                print(f"[sinfo] picked autotrain node {best[0].name} "
                      f"(state={best[0].state}, "
                      f"gpus={best[0].gpu_count})")
            else:
                print("[sinfo] no idle GPU nodes found; letting "
                      "SLURM choose")
        except Exception as e:
            print(f"[sinfo] query failed: {e}; letting SLURM choose")

    script = _build_autotrain_sbatch(args, args.log_dir,
                                     node_name=chosen_node)

    print(f"Winners agent:   {args.winners_agent}")
    print(f"Source agent:    {args.source_agent}")
    print(f"Max nodes/phase: {args.max_nodes}")
    print(f"Max steps/run:   {args.max_steps_per_run:,}")
    print(f"Eval interval:   {args.eval_interval_s}s")
    print(f"Strehl thresh:   {args.strehl_threshold}")
    print(f"Phases override: {args.phases or '(auto from manifest)'}")
    print(f"Master node:     {chosen_node or '(SLURM chooses)'}")
    print(f"Master time:     {args.master_time}")
    print(f"Log dir:         {args.log_dir}")
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
    print(f"Submitted autotrain job {job_id}")
    print(f"Tail:  tail -f {args.log_dir}/autotrain-*.out")
    print(f"Per-seed workers:  squeue -u $USER | grep auto-")


if __name__ == "__main__":
    main()
