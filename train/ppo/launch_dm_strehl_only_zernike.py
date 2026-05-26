#!/usr/bin/env python
"""Launch the Zernike-coefficient strehl-only sanity-check on SLURM.

Companion to ``train_ppo_elf_dm_strehl_only_zernike.py``: same env as
the full-DM run (no bilateral wrapper, no dark hole, plain strehl
reward) but the policy operates in a low-dim Zernike-coefficient
action space via the ZernikeDMVectorEnv wrapper. Same multi-seed
launcher pattern as launch_dm_strehl_only_full.py.

Usage:
    poetry run python train/ppo/launch_dm_strehl_only_zernike.py
    poetry run python train/ppo/launch_dm_strehl_only_zernike.py --num-jobs 6
    poetry run python train/ppo/launch_dm_strehl_only_zernike.py --n-zernike 36
    poetry run python train/ppo/launch_dm_strehl_only_zernike.py --num-jobs 3 --dry-run
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
_RUN_PREFIX = "dm_strehl_only_zernike"


def make_sbatch_script(run_id, run_dir, seed, init_dm_micron_std,
                       n_zernike, wall_time=SLURM_TIME):
    job_name = f"dmz-{run_id[-8:]}-{seed % 10000:04d}"
    extras = []
    if init_dm_micron_std is not None:
        extras.append(
            f"--init-dm-micron-std {init_dm_micron_std:.4f}")
    if n_zernike is not None:
        extras.append(f"--n-zernike {int(n_zernike)}")
    extras_str = (" \\\n            " + " ".join(extras) + " "
                  if extras else "")
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

        cd {HPC_CODE_DIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --seed {seed}{extras_str}\\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Zernike-coefficient strehl-only jobs on SLURM.")
    parser.add_argument("--run-id", type=str, default=None,
                        help=f"Unique run ID (default: {_RUN_PREFIX}_<ts>).")
    parser.add_argument("--num-jobs", type=int, default=1,
                        help="Number of independent jobs.")
    parser.add_argument("--ppo-seed", type=int, default=None,
                        help="If set, every job uses this PPO seed.")
    parser.add_argument("--init-dm-micron-std", type=float, default=None)
    parser.add_argument("--n-zernike", type=int, default=None,
                        help="Override the Zernike basis size (default 24).")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time", type=str, default=SLURM_TIME)
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
        seeds = [secrets.randbelow(MAX_SEED) for _ in range(cli.num_jobs)]
        seed_mode = "random per-job"

    print(f"Run ID:       {run_id}")
    print(f"Output dir:   {run_dir_base}")
    print(f"Num jobs:     {cli.num_jobs}")
    print(f"PPO seeds:    {seed_mode}")
    print(f"n_zernike:    {cli.n_zernike if cli.n_zernike is not None else '(default 24)'}")
    print(f"Init DM std:  "
          f"{cli.init_dm_micron_std if cli.init_dm_micron_std is not None else '(default 0.05 um)'}")
    print(f"Mode:         {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()

    if cli.local:
        for i, seed in enumerate(seeds):
            run_dir = os.path.join(run_dir_base, f"seed_{seed}")
            cmd = [sys.executable, _TRAIN_SCRIPT,
                   "--seed", str(seed), "--run-dir", run_dir]
            if cli.init_dm_micron_std is not None:
                cmd.extend(["--init-dm-micron-std",
                            f"{cli.init_dm_micron_std:.4f}"])
            if cli.n_zernike is not None:
                cmd.extend(["--n-zernike", str(int(cli.n_zernike))])
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
            n_zernike=cli.n_zernike,
            wall_time=cli.time)
        if cli.dry_run:
            print(f"Job {i:2d}: would submit (seed={seed})")
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
        print(f"Monitor: squeue -u $USER | grep dmz-{run_id[-8:]}")


if __name__ == "__main__":
    main()
