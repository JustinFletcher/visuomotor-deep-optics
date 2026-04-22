#!/usr/bin/env python
"""Launch a sweep of dark-hole shaping training runs.

Submits one sbatch job per target. Each job runs
``train_ppo_elf_dark_hole.py`` with a different randomly-drawn
dark-hole geometry (angular location, radial location, size). The
geometry is fixed for the whole run, so every episode within a single
training job shares the same target hole.

Geometry sampling:
  * angular location:           uniform on [0, 360) degrees
  * radial location fraction:   uniform on [0.10, 0.40]
  * size radius:                uniform on [0.02, 0.08]

Directory structure:
    dark_hole_runs/
      dark_hole_{run_id}/
        target_00/  (first geometry)
          ppo_optomech_.../
        target_01/  (second geometry)
        ...

Usage:
    # Launch 8 random targets on HPC
    python train/ppo/launch_dark_hole.py --num-targets 8

    # Reproduce a specific sweep by fixing the geometry RNG seed
    python train/ppo/launch_dark_hole.py --num-targets 8 --geom-seed 42

    # Local (one-target-at-a-time) test run
    python train/ppo/launch_dark_hole.py --num-targets 2 --local

    # Dry run: print what would be submitted
    python train/ppo/launch_dark_hole.py --num-targets 8 --dry-run
"""

import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time

import numpy as np

# 32-bit cap so the seed survives PyTorch / numpy seed-setting paths
# that internally call seed % 2**32.
MAX_SEED = 2**31 - 1

# SLURM defaults (match run_elf_bootstrap.sbatch)
SLURM_ACCOUNT = "MHPCC38870258"
SLURM_PARTITION = "standard"
SLURM_TIME = "72:00:00"
SLURM_GRES = "gpu"

# HPC working directory
HPC_WORKDIR = "/p/home/fletch/visuomotor-deep-optics"

# Geometry-sampling envelope.
ANGLE_RANGE_DEG = (0.0, 360.0)
RADIUS_FRAC_RANGE = (0.10, 0.40)
SIZE_RADIUS_RANGE = (0.02, 0.08)


def sample_geometry(rng):
    """Draw one (angle, radius_frac, size_radius) tuple."""
    angle = float(rng.uniform(*ANGLE_RANGE_DEG))
    radius_frac = float(rng.uniform(*RADIUS_FRAC_RANGE))
    size_radius = float(rng.uniform(*SIZE_RADIUS_RANGE))
    return angle, radius_frac, size_radius


def make_sbatch_script(target_idx, run_id, run_dir_base, seed,
                       angle_deg, radius_frac, size_radius,
                       wall_time=SLURM_TIME):
    """Generate sbatch script contents for one dark-hole target."""
    run_dir = os.path.join(run_dir_base, f"target_{target_idx:02d}")
    job_name = f"dh-{run_id[-8:]}-{target_idx:02d}"

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-target{target_idx:02d}-%j.out
        #SBATCH --error=slurm-{run_id}-target{target_idx:02d}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python train/ppo/train_ppo_elf_dark_hole.py \\
            --hpc \\
            --dark-hole-angle {angle_deg:.4f} \\
            --dark-hole-radius-frac {radius_frac:.4f} \\
            --dark-hole-size {size_radius:.4f} \\
            --seed {seed} \\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch a sweep of dark-hole shaping training runs")
    parser.add_argument(
        "--num-targets", type=int, required=True,
        help="Number of training jobs to launch, each with a distinct "
             "randomly-sampled dark-hole geometry.")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Unique run ID (default: dark_hole_{timestamp})")
    parser.add_argument(
        "--geom-seed", type=int, default=None,
        help="RNG seed for the geometry sampler (default: random). "
             "Use this to reproduce the same set of holes across "
             "two launches.")
    parser.add_argument(
        "--ppo-seed", type=int, default=None,
        help="If set, every launched job uses this same PPO seed; "
             "otherwise each job gets a fresh random seed.")
    parser.add_argument(
        "--local", action="store_true",
        help="Run jobs sequentially in this process instead of "
             "submitting sbatch jobs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands / scripts without executing.")
    parser.add_argument(
        "--time", type=str, default=SLURM_TIME,
        help=f"SLURM wall time (default: {SLURM_TIME})")
    cli = parser.parse_args()

    if cli.num_targets < 1:
        print("Error: --num-targets must be >= 1")
        sys.exit(1)

    run_id = cli.run_id or f"dark_hole_{int(time.time())}"
    run_dir_base = os.path.join("dark_hole_runs", run_id)

    geom_rng = np.random.default_rng(
        cli.geom_seed if cli.geom_seed is not None
        else secrets.randbelow(MAX_SEED))
    geometries = [sample_geometry(geom_rng) for _ in range(cli.num_targets)]

    if cli.ppo_seed is not None:
        ppo_seeds = [int(cli.ppo_seed)] * cli.num_targets
        seed_mode = f"explicit ({cli.ppo_seed})"
    else:
        ppo_seeds = [secrets.randbelow(MAX_SEED) for _ in range(cli.num_targets)]
        seed_mode = "random per-target"

    print(f"Run ID:      {run_id}")
    print(f"Output dir:  {run_dir_base}")
    print(f"Targets:     {cli.num_targets}")
    print(f"Geom seed:   {cli.geom_seed if cli.geom_seed is not None else 'random'}")
    print(f"PPO seeds:   {seed_mode}")
    print(f"Mode:        {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()
    print(f"{'#':>3}  {'angle (deg)':>11}  {'radius_frac':>12}  "
          f"{'size_radius':>12}  {'ppo_seed':>10}")
    for i, ((a, rf, sr), s) in enumerate(zip(geometries, ppo_seeds)):
        print(f"{i:3d}  {a:11.3f}  {rf:12.3f}  {sr:12.3f}  {s:10d}")
    print()

    if cli.local:
        for i, ((angle, radius_frac, size_radius), seed) in enumerate(
                zip(geometries, ppo_seeds)):
            run_dir = os.path.join(run_dir_base, f"target_{i:02d}")
            cmd = [
                sys.executable,
                "train/ppo/train_ppo_elf_dark_hole.py",
                "--dark-hole-angle", f"{angle:.4f}",
                "--dark-hole-radius-frac", f"{radius_frac:.4f}",
                "--dark-hole-size", f"{size_radius:.4f}",
                "--seed", str(seed),
                "--run-dir", run_dir,
            ]
            print(f"Target {i:2d}: {' '.join(cmd)}")
            if not cli.dry_run:
                os.makedirs(run_dir, exist_ok=True)
                subprocess.run(cmd, check=True)
        return

    # SLURM path.
    job_ids = []
    for i, ((angle, radius_frac, size_radius), seed) in enumerate(
            zip(geometries, ppo_seeds)):
        script = make_sbatch_script(
            target_idx=i, run_id=run_id, run_dir_base=run_dir_base,
            seed=seed, angle_deg=angle, radius_frac=radius_frac,
            size_radius=size_radius, wall_time=cli.time)

        if cli.dry_run:
            print(f"Target {i:2d}: would submit sbatch job")
            print(textwrap.indent(script, "    "))
            continue

        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Target {i:2d}: FAILED - {result.stderr.strip()}")
            continue

        job_id = result.stdout.strip()
        job_ids.append((i, job_id))
        print(f"Target {i:2d}: submitted job {job_id}")

    if job_ids and not cli.dry_run:
        print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
        print(f"Monitor: squeue -u $USER | grep dh-{run_id[-8:]}")


if __name__ == "__main__":
    main()
