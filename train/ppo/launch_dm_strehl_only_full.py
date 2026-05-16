#!/usr/bin/env python
"""Launch the full-DM strehl-only sanity-check experiment on SLURM.

Companion to ``train_ppo_elf_dm_strehl_only_full.py``: bilateral
wrapper DISABLED (1225-dim full-DM action), no dark hole, no
curriculum, plain Strehl reward only. The simplest "can PPO close
the loop on this hardware" control for the bilateral-DM dark-hole
train-eval gap.

Unlike ``launch_static_dark_hole.py`` there's no grid of targets to
sweep -- the task has a single, fixed configuration. The only thing
that varies between submitted jobs is the PPO seed.

--num-jobs N submits N independent jobs, each with its own random
seed. The default N=1 still picks a random seed unless --ppo-seed is
explicitly set (matches the "should be random by default even in the
single-job case" requirement). Each job lands in
``dark_hole_runs/<run_id>/seed_<seed>/``.

Usage:
    # Single random-seed job:
    python train/ppo/launch_dm_strehl_only_full.py

    # Five independent random-seed jobs:
    python train/ppo/launch_dm_strehl_only_full.py --num-jobs 5

    # Single job, explicit seed (reproducible):
    python train/ppo/launch_dm_strehl_only_full.py --ppo-seed 42

    # Override the DM init perturbation:
    python train/ppo/launch_dm_strehl_only_full.py --init-dm-micron-std 0.1

    # Dry-run to inspect the sbatch scripts without submitting:
    python train/ppo/launch_dm_strehl_only_full.py --num-jobs 3 --dry-run

    # Local sequential execution (smoke test; do not use for real runs):
    python train/ppo/launch_dm_strehl_only_full.py --local
"""
import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dm_strehl_only_full.py"
_RUN_PREFIX = "dm_strehl_only_full"


def make_sbatch_script(run_id, run_dir, seed, init_dm_micron_std,
                       wall_time=SLURM_TIME):
    """Build an sbatch script for a single full-DM strehl-only job.

    The training script accepts --seed and --init-dm-micron-std via its
    own pre-parser, then forwards remaining flags (notably --hpc and
    --run-dir) to run_main(). Same pattern as the dark-hole launcher.
    """
    # Short job suffix mirrors the dark-hole launcher's `dhs-...` style
    # so existing squeue / monitoring habits keep working.
    job_name = f"dmf-{run_id[-8:]}-{seed % 10000:04d}"
    extra = ""
    if init_dm_micron_std is not None:
        extra = (f"\\\n            --init-dm-micron-std "
                 f"{init_dm_micron_std:.4f} ")
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-seed{seed}-%j.out
        #SBATCH --error=slurm-{run_id}-seed{seed}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --seed {seed} {extra}\\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch full-DM strehl-only training jobs on SLURM.")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help=f"Unique run ID (default: {_RUN_PREFIX}_<timestamp>).")
    parser.add_argument(
        "--num-jobs", type=int, default=1,
        help="Number of independent jobs to submit; each gets its own "
             "random PPO seed unless --ppo-seed is set (default: 1).")
    parser.add_argument(
        "--ppo-seed", type=int, default=None,
        help="If set, every launched job uses this same PPO seed; "
             "otherwise each job gets a distinct random seed (even when "
             "--num-jobs 1).")
    parser.add_argument(
        "--init-dm-micron-std", type=float, default=None,
        help="Override the per-actuator init std (microns).")
    parser.add_argument(
        "--local", action="store_true",
        help="Run jobs sequentially in-process instead of submitting "
             "sbatch jobs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch scripts without executing.")
    parser.add_argument(
        "--time", type=str, default=SLURM_TIME,
        help=f"SLURM wall time (default: {SLURM_TIME}).")
    cli = parser.parse_args()

    if cli.num_jobs < 1:
        print("Error: --num-jobs must be >= 1")
        sys.exit(1)

    run_id = cli.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    run_dir_base = os.path.join("dark_hole_runs", run_id)

    if cli.ppo_seed is not None:
        seeds = [int(cli.ppo_seed)] * cli.num_jobs
        seed_mode = f"explicit ({cli.ppo_seed}, replicated)"
    else:
        # Random seed per job, even when num-jobs=1. ``secrets`` is
        # used (not ``random``) to avoid global RNG-state coupling.
        seeds = [secrets.randbelow(MAX_SEED) for _ in range(cli.num_jobs)]
        seed_mode = "random per-job"

    print(f"Run ID:       {run_id}")
    print(f"Output dir:   {run_dir_base}")
    print(f"Num jobs:     {cli.num_jobs}")
    print(f"PPO seeds:    {seed_mode}")
    print(f"Init DM std:  "
          f"{cli.init_dm_micron_std if cli.init_dm_micron_std is not None else '(default 0.05 um)'}")
    print(f"Mode:         {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()
    print(f"{'job':>3}  {'seed':>12}  run_dir")
    for i, seed in enumerate(seeds):
        run_dir = os.path.join(run_dir_base, f"seed_{seed}")
        print(f"{i:3d}  {seed:12d}  {run_dir}")
    print()

    if cli.local:
        for i, seed in enumerate(seeds):
            run_dir = os.path.join(run_dir_base, f"seed_{seed}")
            cmd = [
                sys.executable, _TRAIN_SCRIPT,
                "--seed", str(seed),
                "--run-dir", run_dir,
            ]
            if cli.init_dm_micron_std is not None:
                cmd.extend(["--init-dm-micron-std",
                            f"{cli.init_dm_micron_std:.4f}"])
            print(f"Job {i:2d}: {' '.join(cmd)}")
            if not cli.dry_run:
                os.makedirs(run_dir, exist_ok=True)
                subprocess.run(cmd, check=True)
        return

    job_ids = []
    for i, seed in enumerate(seeds):
        run_dir = os.path.join(run_dir_base, f"seed_{seed}")
        script = make_sbatch_script(
            run_id=run_id, run_dir=run_dir, seed=seed,
            init_dm_micron_std=cli.init_dm_micron_std,
            wall_time=cli.time)
        if cli.dry_run:
            print(f"Job {i:2d}: would submit sbatch job (seed={seed})")
            print(textwrap.indent(script, "    "))
            continue
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Job {i:2d}: FAILED - {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        job_ids.append((seed, job_id))
        print(f"Job {i:2d}: submitted job {job_id} (seed={seed})")

    if job_ids and not cli.dry_run:
        print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
        print(f"Monitor: squeue -u $USER | grep dmf-{run_id[-8:]}")


if __name__ == "__main__":
    main()
