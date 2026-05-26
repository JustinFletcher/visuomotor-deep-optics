#!/usr/bin/env python
"""Sweep ``n_zernike`` across one SLURM job per basis size.

Companion to ``launch_dm_strehl_only_zernike.py`` which holds n_zernike
fixed across N independent jobs. This launcher instead submits one job
per basis size, each with its own random PPO seed. Default sweep is
``[32, 64, 128, 256, 512, 1024]`` -- six jobs covering a 32x range so
the n_zernike-vs-strehl curve is resolvable in a single submission.

Premise: dm_strehl_only_zernike_1779505485 trained cleanly at
n_zernike=24 (sensible loss curves, no NaN, no kl blowup) but plateaued
at low strehl. The hypothesis is that 24 modes is too coarse a basis
to invert the per-actuator-iid init perturbation, which has energy
distributed across high spatial frequencies. The sweep tests whether
increasing the basis size unblocks the agent before we redesign the
task.

Each job lands under
``dark_hole_runs/<run_id>/n<n_zernike>_seed<seed>/`` so the per-mode
output dirs don't collide and post-hoc analysis can sort by basis size.

Usage:
    poetry run python train/ppo/launch_dm_strehl_only_zernike_sweep.py
    poetry run python train/ppo/launch_dm_strehl_only_zernike_sweep.py --dry-run
    poetry run python train/ppo/launch_dm_strehl_only_zernike_sweep.py \\
        --n-zernike 16 32 64 128

    # Reproducible seeds:
    poetry run python train/ppo/launch_dm_strehl_only_zernike_sweep.py \\
        --ppo-seeds 1 2 3 4 5 6
"""
import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time

from train.ppo.launch_static_dark_hole import (
    HPC_CODE_DIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dm_strehl_only_zernike.py"
_RUN_PREFIX = "dm_strehl_only_zernike_sweep"
_DEFAULT_N_ZERNIKE = [32, 64, 128, 256, 512, 1024]


def make_sbatch_script(run_id, run_dir, seed, init_dm_micron_std,
                       n_zernike, wall_time=SLURM_TIME):
    """sbatch script for one (n_zernike, seed) cell of the sweep."""
    job_name = f"dmzs-{run_id[-8:]}-n{n_zernike:04d}"
    extras = []
    if init_dm_micron_std is not None:
        extras.append(
            f"--init-dm-micron-std {init_dm_micron_std:.4f}")
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
        #SBATCH --output=slurm-{run_id}-n{n_zernike:04d}-seed{seed}-%j.out
        #SBATCH --error=slurm-{run_id}-n{n_zernike:04d}-seed{seed}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_CODE_DIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --seed {seed}{extras_str}\\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep n_zernike across SLURM jobs (one job per size).")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help=f"Unique run ID (default: {_RUN_PREFIX}_<timestamp>).")
    parser.add_argument(
        "--n-zernike", type=int, nargs="+", default=_DEFAULT_N_ZERNIKE,
        help=f"Sweep grid of Zernike mode counts (default: "
             f"{_DEFAULT_N_ZERNIKE}).")
    parser.add_argument(
        "--ppo-seeds", type=int, nargs="+", default=None,
        help="If set, one seed per --n-zernike value (lengths must "
             "match). Otherwise each job gets a fresh random seed.")
    parser.add_argument(
        "--init-dm-micron-std", type=float, default=None,
        help="Override the per-actuator init std (microns) for every "
             "job in the sweep.")
    parser.add_argument(
        "--local", action="store_true",
        help="Run jobs sequentially in-process instead of submitting "
             "sbatch jobs (smoke testing only -- each job is hours).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch scripts without executing.")
    parser.add_argument(
        "--time", type=str, default=SLURM_TIME,
        help=f"SLURM wall time (default: {SLURM_TIME}).")
    cli = parser.parse_args()

    n_zernike_list = list(cli.n_zernike)
    n_jobs = len(n_zernike_list)
    if n_jobs < 1:
        print("Error: --n-zernike must have at least one value")
        sys.exit(1)

    # Resolve seeds.
    if cli.ppo_seeds is not None:
        if len(cli.ppo_seeds) != n_jobs:
            print(f"Error: --ppo-seeds must have {n_jobs} values to "
                  f"match --n-zernike ({n_jobs} values)")
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
    print(f"Init DM std:  "
          f"{cli.init_dm_micron_std if cli.init_dm_micron_std is not None else '(default 0.05 um)'}")
    print(f"Mode:         {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()
    print(f"{'job':>3}  {'n_zernike':>9}  {'seed':>12}  run_dir")
    for i, (n_z, seed) in enumerate(zip(n_zernike_list, seeds)):
        run_dir = os.path.join(run_dir_base, f"n{n_z:04d}_seed{seed}")
        print(f"{i:3d}  {n_z:9d}  {seed:12d}  {run_dir}")
    print()

    if cli.local:
        for i, (n_z, seed) in enumerate(zip(n_zernike_list, seeds)):
            run_dir = os.path.join(run_dir_base, f"n{n_z:04d}_seed{seed}")
            cmd = [
                sys.executable, _TRAIN_SCRIPT,
                "--seed", str(seed),
                "--n-zernike", str(int(n_z)),
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
    for i, (n_z, seed) in enumerate(zip(n_zernike_list, seeds)):
        run_dir = os.path.join(run_dir_base, f"n{n_z:04d}_seed{seed}")
        script = make_sbatch_script(
            run_id=run_id, run_dir=run_dir, seed=seed,
            init_dm_micron_std=cli.init_dm_micron_std,
            n_zernike=n_z, wall_time=cli.time)
        if cli.dry_run:
            print(f"Job {i:2d}: would submit (n_zernike={n_z}, seed={seed})")
            print(textwrap.indent(script, "    "))
            continue
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Job {i:2d}: FAILED - {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        job_ids.append((n_z, seed, job_id))
        print(f"Job {i:2d}: submitted job {job_id} "
              f"(n_zernike={n_z}, seed={seed})")

    if job_ids and not cli.dry_run:
        print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
        print(f"Monitor: squeue -u $USER | grep dmzs-{run_id[-8:]}")


if __name__ == "__main__":
    main()
