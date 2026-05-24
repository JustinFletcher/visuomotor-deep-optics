#!/usr/bin/env python
"""SLURM sweep over realistic Mt. Teide r0 values, atmospheric disturbance.

One job per r0. Trains the Zernike-action / atmosphere-disturbance
strehl-only experiment with eval images enabled so the eval rollout
filmstrips are visible in TB at every eval interval (1 episode/eval).

Default r0 grid spans Teide-typical seeing:
    0.06 m  -- below median
    0.09 m  -- median
    0.12 m  -- median-good (Teide-typical)
    0.18 m  -- good
    0.25 m  -- excellent (1st percentile night)

Each job lands at
    dark_hole_runs/<run_id>/r0_<NN>cm_seed<seed>/
so per-r0 outputs don't collide and post-hoc sweep analysis can sort
by both r0 and seed.

Usage:
    poetry run python train/ppo/launch_dm_strehl_only_zernike_atm_sweep.py
    poetry run python train/ppo/launch_dm_strehl_only_zernike_atm_sweep.py --dry-run
    poetry run python train/ppo/launch_dm_strehl_only_zernike_atm_sweep.py \\
        --r0-grid 0.08 0.12 0.20 --n-zernike 64
"""
import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm.py"
_RUN_PREFIX = "dm_strehl_only_zernike_atm_sweep"

# Realistic Teide site-quality grid (r0 at 500 nm).
_DEFAULT_R0_GRID = [0.06, 0.09, 0.12, 0.18, 0.25]


def make_sbatch_script(run_id, run_dir, seed, r0_m, n_zernike,
                       L0_m, wall_time=SLURM_TIME):
    # Job label encodes r0 in cm for at-a-glance squeue scanning.
    r0_cm = int(round(r0_m * 100))
    job_name = f"atms-{run_id[-8:]}-r{r0_cm:02d}"
    extras = [f"--r0 {r0_m:.4f}"]
    if L0_m is not None:
        extras.append(f"--L0 {L0_m:.2f}")
    if n_zernike is not None:
        extras.append(f"--n-zernike {int(n_zernike)}")
    extras_str = " \\\n            " + " ".join(extras) + " "
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-r{r0_cm:02d}-seed{seed}-%j.out
        #SBATCH --error=slurm-{run_id}-r{r0_cm:02d}-seed{seed}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --seed {seed}{extras_str}\\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch one SLURM job per r0 value (atmosphere "
                    "disturbance + Zernike action + eval images).")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help=f"Unique run ID (default: {_RUN_PREFIX}_<timestamp>).")
    parser.add_argument(
        "--r0-grid", type=float, nargs="+", default=_DEFAULT_R0_GRID,
        help=f"r0 values to sweep, meters at 500 nm (default: "
             f"{_DEFAULT_R0_GRID}).")
    parser.add_argument(
        "--L0", type=float, default=None,
        help="Override the von Karman outer scale L0 (m) on every job.")
    parser.add_argument(
        "--n-zernike", type=int, default=None,
        help="Override Zernike basis size (inherited default = 24).")
    parser.add_argument(
        "--ppo-seeds", type=int, nargs="+", default=None,
        help="If set, one seed per r0 (lengths must match); otherwise "
             "each job draws a fresh random seed.")
    parser.add_argument(
        "--local", action="store_true",
        help="Run sequentially in-process (smoke test only).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch scripts without executing.")
    parser.add_argument(
        "--time", type=str, default=SLURM_TIME,
        help=f"SLURM wall time (default: {SLURM_TIME}).")
    cli = parser.parse_args()

    r0_grid = list(cli.r0_grid)
    n_jobs = len(r0_grid)
    if n_jobs < 1:
        print("Error: --r0-grid must have at least one value")
        sys.exit(1)
    if any(r <= 0 for r in r0_grid):
        print("Error: --r0-grid values must be positive (meters)")
        sys.exit(1)

    if cli.ppo_seeds is not None:
        if len(cli.ppo_seeds) != n_jobs:
            print(f"Error: --ppo-seeds must have {n_jobs} values to "
                  f"match --r0-grid ({n_jobs} values)")
            sys.exit(1)
        seeds = [int(s) for s in cli.ppo_seeds]
        seed_mode = "explicit per-job"
    else:
        seeds = [secrets.randbelow(MAX_SEED) for _ in range(n_jobs)]
        seed_mode = "random per-job"

    run_id = cli.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    run_dir_base = os.path.join("dark_hole_runs", run_id)

    print(f"Run ID:       {run_id}")
    print(f"Output dir:   {run_dir_base}")
    print(f"Num jobs:     {n_jobs}")
    print(f"PPO seeds:    {seed_mode}")
    print(f"L0 override:  "
          f"{cli.L0 if cli.L0 is not None else '(default 25 m)'}")
    print(f"n_zernike:    "
          f"{cli.n_zernike if cli.n_zernike is not None else '(inherit 24)'}")
    print(f"Mode:         {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()
    print(f"{'job':>3}  {'r0 (m)':>7}  {'r0 (cm)':>8}  {'seed':>12}  run_dir")
    for i, (r0, seed) in enumerate(zip(r0_grid, seeds)):
        r0_cm = int(round(r0 * 100))
        run_dir = os.path.join(run_dir_base,
                               f"r0_{r0_cm:02d}cm_seed{seed}")
        print(f"{i:3d}  {r0:7.3f}  {r0_cm:8d}  {seed:12d}  {run_dir}")
    print()

    if cli.local:
        for i, (r0, seed) in enumerate(zip(r0_grid, seeds)):
            r0_cm = int(round(r0 * 100))
            run_dir = os.path.join(run_dir_base,
                                   f"r0_{r0_cm:02d}cm_seed{seed}")
            cmd = [sys.executable, _TRAIN_SCRIPT,
                   "--seed", str(seed),
                   "--r0", f"{r0:.4f}",
                   "--run-dir", run_dir]
            if cli.L0 is not None:
                cmd.extend(["--L0", f"{cli.L0:.2f}"])
            if cli.n_zernike is not None:
                cmd.extend(["--n-zernike", str(int(cli.n_zernike))])
            print(f"Job {i:2d}: {' '.join(cmd)}")
            if not cli.dry_run:
                os.makedirs(run_dir, exist_ok=True)
                subprocess.run(cmd, check=True)
        return

    job_ids = []
    for i, (r0, seed) in enumerate(zip(r0_grid, seeds)):
        r0_cm = int(round(r0 * 100))
        run_dir = os.path.join(run_dir_base,
                               f"r0_{r0_cm:02d}cm_seed{seed}")
        script = make_sbatch_script(
            run_id=run_id, run_dir=run_dir, seed=seed, r0_m=r0,
            n_zernike=cli.n_zernike, L0_m=cli.L0,
            wall_time=cli.time)
        if cli.dry_run:
            print(f"Job {i:2d}: would submit (r0={r0:.3f} m, seed={seed})")
            print(textwrap.indent(script, "    "))
            continue
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Job {i:2d}: FAILED - {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        job_ids.append((r0, seed, job_id))
        print(f"Job {i:2d}: submitted job {job_id} "
              f"(r0={r0:.3f} m, seed={seed})")

    if job_ids and not cli.dry_run:
        print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
        print(f"Monitor: squeue -u $USER | grep atms-{run_id[-8:]}")


if __name__ == "__main__":
    main()
