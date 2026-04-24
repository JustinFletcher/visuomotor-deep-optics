#!/usr/bin/env python
"""Launch a single dynamic (target-randomised) dark-hole training run.

Unlike the static and grid launchers, this one submits one SLURM job
that trains a single target-aware policy where the dark-hole geometry
is resampled from a uniform envelope at the start of every episode.
The policy's LSTM input includes a 4-dim target vector
[sin(θ), cos(θ), radius_frac, size_frac] so it learns to generalise
across the envelope rather than overfitting one geometry.

Sampling envelope (matches the static-grid rings):
    angle:       [0, 360) degrees
    radius_frac: [0.16, 0.32]
    size_frac:   0.08 (fixed)

Env version is pinned to v4 because v5's hole mask is currently a
shared batched tensor and per-episode resampling would require a
per-env mask refactor. V4 is slower but correct.

Usage:
    python train/ppo/launch_dynamic_dark_hole.py
    python train/ppo/launch_dynamic_dark_hole.py --ppo-seed 42
    python train/ppo/launch_dynamic_dark_hole.py --dry-run
    python train/ppo/launch_dynamic_dark_hole.py --local  # run in-process
"""
import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time

MAX_SEED = 2**31 - 1
SLURM_ACCOUNT = "MHPCC38870258"
SLURM_PARTITION = "standard"
SLURM_TIME = "72:00:00"
SLURM_GRES = "gpu"
HPC_WORKDIR = "/p/home/fletch/visuomotor-deep-optics"


def make_sbatch_script(run_id, run_dir, seed, wall_time=SLURM_TIME):
    job_name = f"dhd-{run_id[-8:]}"
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
        poetry run python train/ppo/train_ppo_elf_dark_hole_dynamic.py \\
            --hpc \\
            --seed {seed} \\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch a single dynamic dark-hole training run.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run ID (default: dark_hole_dynamic_{ts})")
    parser.add_argument("--ppo-seed", type=int, default=None,
                        help="PPO seed (default: fresh random).")
    parser.add_argument("--local", action="store_true",
                        help="Run in-process instead of submitting sbatch.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch script without executing.")
    parser.add_argument("--time", type=str, default=SLURM_TIME,
                        help=f"SLURM wall time (default: {SLURM_TIME})")
    cli = parser.parse_args()

    run_id = cli.run_id or f"dark_hole_dynamic_{int(time.time())}"
    run_dir = os.path.join("dark_hole_runs", run_id)
    seed = (int(cli.ppo_seed) if cli.ppo_seed is not None
            else secrets.randbelow(MAX_SEED))

    print(f"Run ID:      {run_id}")
    print(f"Output dir:  {run_dir}")
    print(f"PPO seed:    {seed}")
    print(f"Mode:        {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()

    if cli.local:
        cmd = [
            sys.executable,
            "train/ppo/train_ppo_elf_dark_hole_dynamic.py",
            "--seed", str(seed),
            "--run-dir", run_dir,
        ]
        print(" ".join(cmd))
        if not cli.dry_run:
            os.makedirs(run_dir, exist_ok=True)
            subprocess.run(cmd, check=True)
        return

    script = make_sbatch_script(run_id, run_dir, seed, wall_time=cli.time)
    if cli.dry_run:
        print(textwrap.indent(script, "    "))
        return

    result = subprocess.run(
        ["sbatch", "--parsable"], input=script,
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED - {result.stderr.strip()}")
        sys.exit(1)
    job_id = result.stdout.strip()
    print(f"Submitted job {job_id}")
    print(f"Monitor: squeue -u $USER | grep dhd-{run_id[-8:]}")


if __name__ == "__main__":
    main()
