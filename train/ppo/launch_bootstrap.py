#!/usr/bin/env python
"""Launch incremental bootstrapping training runs.

Submits 15 sbatch jobs (one per ELF segment phase) or runs locally.
Each launch gets a unique run ID; phases are organized underneath it.

Directory structure:
    bootstrap_runs/
      bootstrap_{run_id}/
        bootstrap_phase_00/
          ppo_optomech_.../ (training output)
        bootstrap_phase_01/
        ...

Usage:
    # Launch all 15 on HPC
    python train/ppo/launch_bootstrap.py

    # Launch subset
    python train/ppo/launch_bootstrap.py --phases 0,3,7

    # Custom run ID
    python train/ppo/launch_bootstrap.py --run-id my_experiment_v2

    # Dry run (print commands only)
    python train/ppo/launch_bootstrap.py --dry-run

    # Run phase 0 locally (no sbatch)
    python train/ppo/launch_bootstrap.py --local --phases 0
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time

NUM_PHASES = 15

# SLURM defaults (match run_elf_bootstrap.sbatch)
SLURM_ACCOUNT = "MHPCC38870258"
SLURM_PARTITION = "standard"
SLURM_TIME = "72:00:00"
SLURM_GRES = "gpu"

# HPC working directory
HPC_WORKDIR = "/p/home/fletch/visuomotor-deep-optics"


def parse_phases(phases_str):
    """Parse '0,3,5-7' into sorted list of ints."""
    result = set()
    for part in phases_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def make_sbatch_script(phase, run_id, run_dir_base, wall_time=SLURM_TIME):
    """Generate sbatch script contents for one phase."""
    run_dir = os.path.join(run_dir_base, f"bootstrap_phase_{phase:02d}")
    job_name = f"boot-{run_id[-8:]}-{phase:02d}"

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-phase{phase:02d}-%j.out
        #SBATCH --error=slurm-{run_id}-phase{phase:02d}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python train/ppo/train_ppo_elf_bootstrap.py \\
            --hpc --no-eval \\
            --phased-count {phase} \\
            --run-dir {run_dir}
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Launch incremental bootstrapping training runs")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Unique run ID (default: bootstrap_{timestamp})")
    parser.add_argument(
        "--phases", type=str, default=None,
        help="Comma-separated phases to launch, e.g. '0,3,5-7' "
             "(default: all 0-14)")
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of submitting sbatch jobs")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing")
    parser.add_argument(
        "--time", type=str, default=SLURM_TIME,
        help=f"SLURM wall time (default: {SLURM_TIME})")
    cli = parser.parse_args()

    # Generate or use provided run ID
    run_id = cli.run_id or f"bootstrap_{int(time.time())}"
    run_dir_base = os.path.join("bootstrap_runs", run_id)

    # Parse phases
    phases = parse_phases(cli.phases) if cli.phases else list(range(NUM_PHASES))
    for p in phases:
        if p < 0 or p >= NUM_PHASES:
            print(f"Error: phase {p} out of range [0, {NUM_PHASES - 1}]")
            sys.exit(1)

    wall_time = cli.time

    print(f"Run ID:     {run_id}")
    print(f"Output dir: {run_dir_base}")
    print(f"Phases:     {phases}")
    print(f"Mode:       {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()

    if cli.local:
        # Run one phase at a time locally
        for phase in phases:
            run_dir = os.path.join(run_dir_base, f"bootstrap_phase_{phase:02d}")
            cmd = [
                sys.executable,
                "train/ppo/train_ppo_elf_bootstrap.py",
                "--phased-count", str(phase),
                "--run-dir", run_dir,
            ]
            print(f"Phase {phase:2d}: {' '.join(cmd)}")
            if not cli.dry_run:
                os.makedirs(run_dir, exist_ok=True)
                subprocess.run(cmd, check=True)
    else:
        # Submit sbatch jobs
        job_ids = []
        for phase in phases:
            script = make_sbatch_script(phase, run_id, run_dir_base,
                                       wall_time=wall_time)

            if cli.dry_run:
                print(f"Phase {phase:2d}: would submit sbatch job")
                print(textwrap.indent(script, "    "))
                continue

            # Submit via sbatch --parsable with stdin script
            result = subprocess.run(
                ["sbatch", "--parsable"],
                input=script, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Phase {phase:2d}: FAILED - {result.stderr.strip()}")
                continue

            job_id = result.stdout.strip()
            job_ids.append((phase, job_id))
            print(f"Phase {phase:2d}: submitted job {job_id}")

        if job_ids and not cli.dry_run:
            print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
            print(f"Monitor: squeue -u $USER | grep boot-{run_id[-8:]}")


if __name__ == "__main__":
    main()
