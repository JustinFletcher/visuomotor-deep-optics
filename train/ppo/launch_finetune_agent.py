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

    # Resume an existing fine-tune job (after SLURM timeout, OOM, or
    # after raising total_timesteps in the recipe). All 15 phases pick
    # up from their own latest.pt -- model + optimizer + global_step.
    poetry run python train/ppo/launch_finetune_agent.py \\
        --resume \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --recipe train/ppo/finetune_recipes/piston_1mm.yaml \\
        --output-root agent_finetuning/agent_..._piston_1mm__1779755759
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time
from typing import Optional

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, SLURM_ACCOUNT, SLURM_GRES, SLURM_PARTITION, SLURM_TIME,
)
from train.ppo.finetune_sinfo import (
    query_sinfo_nodes, select_best_nodes)


def _build_master_sbatch(args, run_id: str, output_root: str,
                         node_name: Optional[str],
                         gpus_per_node: int) -> str:
    job_name = f"ftm-{run_id[-8:]}"
    log_root = os.path.join(output_root, "master")
    gres_str = f"gpu:{gpus_per_node}" if gpus_per_node > 1 else SLURM_GRES
    nodelist_line = f"#SBATCH --nodelist={node_name}\n        " if node_name else ""
    resume_flag = " --resume" if args.resume else ""
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.master_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        {nodelist_line}#SBATCH --gres={gres_str}
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
            --max-nodes {args.max_nodes} \\
            --gpus-per-node {gpus_per_node} \\
            --poll-interval-s {args.poll_interval_s} \\
            --max-retries {args.max_retries} \\
            --slurm-time {args.slurm_time}{resume_flag}
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
    parser.add_argument("--max-nodes", type=int, default=4,
                        help="Total number of nodes used by the run, "
                             "INCLUDING the master's own node. Default "
                             "4 (1 master + up to 3 sbatch node-block "
                             "workers, each saturated with all its GPUs). "
                             "On a 4-GPU-per-node cluster, --max-nodes 4 "
                             "= 16 concurrent fine-tunes -- enough to "
                             "run all 15 phases of a bootstrap agent in "
                             "parallel.")
    parser.add_argument("--gpus-per-node", type=int, default=None,
                        help="GPUs to request for the master sbatch. "
                             "Default: pick the largest free node via "
                             "sinfo and use its full GPU count (so the "
                             "master saturates one node). Pass an explicit "
                             "value to override that auto-selection.")
    parser.add_argument("--master-node", type=str, default=None,
                        help="Explicit --nodelist for the master sbatch. "
                             "Default: pick the best idle node via sinfo. "
                             "Pass 'auto' (or omit) for auto-pick, "
                             "'any' to let SLURM choose without "
                             "--nodelist, or a node name to pin.")
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
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing fine-tune job. --output-"
                             "root must point at an existing dir created "
                             "by a prior submission of this launcher. The "
                             "master scans each phase's prior latest.pt "
                             "and continues training (full model + "
                             "optimizer + global_step resume). New "
                             "ppo_optomech_* sub-run dirs are created for "
                             "the resumed runs; prior TB events are left "
                             "in place. Use after a SLURM timeout, OOM, "
                             "or manual cancellation, or to continue past "
                             "an earlier total_timesteps cap once it has "
                             "been raised in the recipe.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch script without submitting.")
    args = parser.parse_args()

    if not os.path.isdir(args.source_agent):
        print(f"ERROR: --source-agent not a directory: {args.source_agent}")
        sys.exit(1)
    if not os.path.isfile(args.recipe):
        print(f"ERROR: --recipe not a file: {args.recipe}")
        sys.exit(1)

    # --resume requires an explicit --output-root pointing at an
    # existing job; we do NOT auto-name on resume.
    if args.resume:
        if args.output_root is None:
            print("ERROR: --resume requires --output-root pointing at an "
                  "existing fine-tune job dir.")
            sys.exit(1)
        if not os.path.isdir(args.output_root):
            print(f"ERROR: --resume --output-root ({args.output_root}) "
                  f"does not exist or is not a directory.")
            sys.exit(1)
        phases_dir = os.path.join(args.output_root, "phases")
        if not os.path.isdir(phases_dir):
            print(f"ERROR: --resume target has no phases/ subdir: "
                  f"{phases_dir}. Is this a finetune_agent run root?")
            sys.exit(1)
    else:
        # Default output dir name: <source>_<recipe>_<timestamp>.
        # Anchored at HPC_WORKDIR (work filesystem) so SLURM jobs
        # writing into it satisfy the "must live under /p/work"
        # cluster policy regardless of where you launch from.
        if args.output_root is None:
            src_base = os.path.basename(os.path.normpath(args.source_agent))
            recipe_base = os.path.basename(args.recipe).rsplit(".", 1)[0]
            ts = int(time.time())
            args.output_root = os.path.join(
                HPC_WORKDIR, "agent_finetuning",
                f"{src_base}__{recipe_base}__{ts}")

    # Read-only invariant: refuse to clobber the source agent.
    if os.path.abspath(args.output_root).startswith(
            os.path.abspath(args.source_agent)):
        print(f"ERROR: --output-root ({args.output_root}) cannot be "
              f"inside --source-agent ({args.source_agent}).")
        sys.exit(1)

    run_id = os.path.basename(args.output_root.rstrip("/"))
    os.makedirs(os.path.join(args.output_root, "master"), exist_ok=True)

    # ----------------------------------------------------------------
    # sinfo-driven master-node selection.
    #
    # We want the master to land on the biggest idle GPU node so it
    # can saturate that node with local workers. Three modes:
    #   --master-node any        -> no --nodelist, no auto-detect
    #   --master-node <name>     -> pin to that node
    #   --master-node auto / unset -> sinfo, pick best
    # When auto-picking and --gpus-per-node is also unset, the chosen
    # node's GPU count drives --gres=gpu:N.
    # ----------------------------------------------------------------
    chosen_node: Optional[str] = None
    gpus_per_node: int = args.gpus_per_node or 0
    if args.master_node == "any":
        pass
    elif args.master_node and args.master_node != "auto":
        chosen_node = args.master_node
    else:
        nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
        best = select_best_nodes(
            nodes, min_gpus=1, allow_mix=False, exclude_self=False)
        if best:
            top = best[0]
            chosen_node = top.name
            if gpus_per_node <= 0:
                gpus_per_node = top.gpu_count
            print(f"[sinfo] picked master node {top.name} "
                  f"(state={top.state}, gpu_count={top.gpu_count})")
        else:
            print("[sinfo] no idle GPU nodes found; letting SLURM "
                  "choose without --nodelist")

    if gpus_per_node <= 0:
        gpus_per_node = 1

    script = _build_master_sbatch(args, run_id, args.output_root,
                                  node_name=chosen_node,
                                  gpus_per_node=gpus_per_node)

    print(f"Source agent:    {args.source_agent}")
    print(f"Recipe:          {args.recipe}")
    print(f"Output root:     {args.output_root}")
    print(f"Resume mode:     {args.resume}")
    print(f"Node budget:     {args.max_nodes} total "
          f"(1 master + up to {args.max_nodes - 1} sbatch blocks)")
    print(f"Master node:     {chosen_node or '(SLURM chooses)'}")
    print(f"Master GPUs:     {gpus_per_node} (--gres=gpu:{gpus_per_node})")
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
