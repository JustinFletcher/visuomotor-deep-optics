#!/usr/bin/env python
"""Launch the strehl-only DM sanity-check run (single SLURM job).

Single job: bilateral DM in fixed_vertical mode, small symmetric
init perturbation, centre-weighted Strehl reward only. The point is
to verify that the 612-dim policy can do closed-loop wavefront
control on this hardware at all before re-engaging the dark-hole
task with a curriculum. See
``train_ppo_elf_dm_strehl_only.py`` for the experiment rationale.

Usage:
    python train/ppo/launch_dm_strehl_only.py
    python train/ppo/launch_dm_strehl_only.py --local
    python train/ppo/launch_dm_strehl_only.py --dry-run
    python train/ppo/launch_dm_strehl_only.py --init-dm-micron-std 0.1
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


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dm_strehl_only.py"
_RUN_PREFIX = "dm_strehl_only"


def make_sbatch_script(run_id, run_dir, seed, init_dm_micron_std,
                       wall_time=SLURM_TIME):
    job_name = f"dms-{run_id[-8:]}"
    extra = ""
    if init_dm_micron_std is not None:
        extra = f"\\\n            --init-dm-micron-std {init_dm_micron_std:.4f} "
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-%j.out
        #SBATCH --error=slurm-{run_id}-%j.err

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
        description="Launch the strehl-only DM sanity-check run.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--ppo-seed", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time", type=str, default=SLURM_TIME)
    parser.add_argument(
        "--init-dm-micron-std", type=float, default=None,
        help="Override the per-actuator init std (microns).")
    cli = parser.parse_args()

    run_id = cli.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    run_dir = os.path.join("dark_hole_runs", run_id)

    seed = cli.ppo_seed if cli.ppo_seed is not None else secrets.randbelow(MAX_SEED)

    print(f"Run ID:       {run_id}")
    print(f"Output dir:   {run_dir}")
    print(f"PPO seed:     {seed}")
    print(f"Init DM std:  {cli.init_dm_micron_std if cli.init_dm_micron_std is not None else '(default 0.05 um)'}")
    print(f"Mode:         {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()

    if cli.local:
        cmd = [sys.executable, _TRAIN_SCRIPT,
               "--seed", str(seed),
               "--run-dir", run_dir]
        if cli.init_dm_micron_std is not None:
            cmd.extend(["--init-dm-micron-std", f"{cli.init_dm_micron_std:.4f}"])
        print("Local cmd:", " ".join(cmd))
        if not cli.dry_run:
            os.makedirs(run_dir, exist_ok=True)
            subprocess.run(cmd, check=True)
        return

    script = make_sbatch_script(
        run_id=run_id, run_dir=run_dir, seed=seed,
        init_dm_micron_std=cli.init_dm_micron_std, wall_time=cli.time)
    if cli.dry_run:
        print("Would submit sbatch job:")
        print(textwrap.indent(script, "    "))
        return

    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED - {result.stderr.strip()}")
        sys.exit(1)
    job_id = result.stdout.strip()
    print(f"Submitted job {job_id}")
    print(f"Monitor: squeue -u $USER | grep dms-{run_id[-8:]}")


if __name__ == "__main__":
    main()
