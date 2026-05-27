#!/usr/bin/env python
"""Per-phase launcher for the segment-PTT + Zernike-DM transfer-learning run.

For each of the 15 bootstrap phases, submits one sbatch job that
fine-tunes ``train_ppo_elf_segment_zernike_dm_transfer.py`` starting
from the corresponding source-agent phase checkpoint. The job grows
the policy head from 45 to 45 + n_zernike (default 32) and trains
under a random Kolmogorov atmosphere (r0 ~ U(0.10, 0.25) m by
default), so the resulting per-phase policies correct both segment
PTT misalignment AND the atmospheric wavefront via the DM in one
combined action vector.

Outputs land under <output_root>/phase_NN/, mirroring the
existing finetune output layout so the same packer / rollout tooling
can chew on the result. Each sbatch job is one node + one GPU
(no in-job fan-out); SLURM handles cross-phase parallelism. For
densely packing multiple phases on a single multi-GPU node, see the
launch_finetune_agent.py master/worker pattern -- this launcher
trades that density for substantial simplicity.

Usage from a login node::

    # First time -- 15 jobs, one per phase.
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7

    # Resume after a SLURM timeout / cluster outage. Re-runs only the
    # phases that have a prior latest.pt (others are submitted fresh).
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --resume \\
        --output-root /p/work/fletch/agent_finetuning/segment_dm_v2_...

    # Dry-run -- print the per-phase sbatch scripts without submitting.
    poetry run python train/ppo/launch_segment_zernike_dm_transfer.py \\
        --source-agent agents/agent_20260419T211137Z_e3b7 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

from train.ppo.launch_static_dark_hole import (
    HPC_CODE_DIR, HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py"
_RUN_PREFIX = "segment_dm_v2"


def _find_latest_checkpoint(phase_dir: str) -> Optional[str]:
    """Return the path to the most-recent ppo_optomech_*/checkpoints/
    latest.pt under ``phase_dir`` (the most recently modified one),
    or None when no prior run exists. Used by --resume to thread
    each phase's continuation through the existing training script's
    resume-in-place machinery."""
    if not os.path.isdir(phase_dir):
        return None
    candidates = []
    for run_dir in Path(phase_dir).glob("ppo_optomech_*"):
        ck = run_dir / "checkpoints" / "latest.pt"
        if ck.is_file():
            candidates.append(ck)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _build_sbatch(args, phase: int, source_ckpt: str,
                  output_dir: str, resume_ckpt: Optional[str],
                  seed: int) -> str:
    """One sbatch script per phase. Single node, single GPU. The
    training script's run_main + resume-in-place logic handles
    everything inside the job."""
    job_name = f"szdm-p{phase:02d}-{args.run_id[-6:]}"
    log_root = os.path.join(output_dir, "_logs")

    # Init-from vs resume-from: resume wins when present.
    ckpt_args = (
        f"--resume-from {resume_ckpt}" if resume_ckpt
        else f"--source-checkpoint {source_ckpt}")

    r0_args = (f"--r0-range {args.r0_range[0]} {args.r0_range[1]}"
               if args.r0_range else "")
    dm_scale_args = (f"--dm-action-scale {args.dm_action_scale}"
                     if args.dm_action_scale is not None else "")

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={args.slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres=gpu:1
        #SBATCH --output={log_root}/sbatch-%j.out
        #SBATCH --error={log_root}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_root}
        cd {HPC_CODE_DIR}

        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            {ckpt_args} \\
            --phased-count {phase} \\
            --n-zernike {args.n_zernike} \\
            {dm_scale_args} \\
            {r0_args} \\
            --seed {seed} \\
            --run-dir {output_dir}
    """)


def main():
    p = argparse.ArgumentParser(
        description="Per-phase segment-PTT + Zernike-DM transfer launcher.")
    p.add_argument("--source-agent", required=True,
                   help="Source bootstrap agent dir, e.g. "
                        "agents/agent_20260419T211137Z_e3b7.")
    p.add_argument("--n-zernike", type=int, default=32,
                   help="Zernike modes for the DM head (default 32).")
    p.add_argument("--dm-action-scale", type=float, default=0.01,
                   help="Per-step DM delta cap. Default 0.01 matches "
                        "the dm_atmos training family.")
    p.add_argument("--r0-range", type=float, nargs=2, default=[0.10, 0.25],
                   metavar=("LOW", "HIGH"),
                   help="Atmosphere r0 sampling range in meters at "
                        "500 nm (default 0.10 0.25).")
    p.add_argument("--phases", type=int, nargs="+", default=None,
                   help="Subset of phase indices to (re)submit "
                        "(default: all 15 phases of the source agent).")
    p.add_argument("--output-root", type=str, default=None,
                   help="Output root dir. Default: $HPC_WORKDIR/"
                        "agent_finetuning/<run_id>/. On --resume this "
                        "must point at the existing run root.")
    p.add_argument("--run-id", type=str, default=None,
                   help=f"Unique run id (default: {_RUN_PREFIX}_<ts>).")
    p.add_argument("--slurm-time", type=str, default=SLURM_TIME,
                   help=f"SLURM --time per phase job (default {SLURM_TIME}).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing per-phase run. For each "
                        "phase, finds the most-recent latest.pt under "
                        "<output-root>/phase_NN/ and passes it to the "
                        "training script as --resume-from (full model "
                        "+ optimizer + global_step resume). Phases "
                        "with no prior latest.pt fall back to init-"
                        "from-source so a partial cluster outage can "
                        "be recovered cleanly. --output-root must "
                        "point at the existing run root.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch scripts without submitting.")
    args = p.parse_args()

    if not os.path.isdir(args.source_agent):
        print(f"ERROR: --source-agent not a directory: {args.source_agent}")
        sys.exit(1)
    manifest_path = os.path.join(args.source_agent, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: missing source manifest: {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    src_phases = manifest.get("phases") or []
    if not src_phases:
        print(f"ERROR: source manifest has no phases: {manifest_path}")
        sys.exit(1)

    # Resolve which phases to (re)submit.
    all_phase_indices = [int(p["phase"]) for p in src_phases]
    if args.phases:
        target_indices = [i for i in args.phases if i in all_phase_indices]
        if len(target_indices) != len(args.phases):
            missing = set(args.phases) - set(all_phase_indices)
            print(f"ERROR: --phases {sorted(missing)} not in source manifest.")
            sys.exit(1)
    else:
        target_indices = all_phase_indices

    # Resolve run id + output root.
    args.run_id = args.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    src_base = os.path.basename(os.path.normpath(args.source_agent))
    output_root = (args.output_root
                   or os.path.join(HPC_WORKDIR, "agent_finetuning",
                                   f"{src_base}__{args.run_id}"))

    if args.resume:
        if not os.path.isdir(output_root):
            print(f"ERROR: --resume requires --output-root to exist "
                  f"({output_root}).")
            sys.exit(1)

    os.makedirs(output_root, exist_ok=True)

    print(f"Source agent:    {args.source_agent}")
    print(f"Output root:     {output_root}")
    print(f"Run id:          {args.run_id}")
    print(f"Resume mode:     {args.resume}")
    print(f"n_zernike:       {args.n_zernike}")
    print(f"dm_action_scale: {args.dm_action_scale}")
    print(f"r0_range (m):    {args.r0_range}")
    print(f"Phases:          {target_indices}")
    print(f"SLURM time:      {args.slurm_time}")
    print()

    submitted = []
    for idx in target_indices:
        src_meta = next(p for p in src_phases if int(p["phase"]) == idx)
        source_ckpt = os.path.join(args.source_agent, src_meta["bundle_path"])
        if not os.path.isfile(source_ckpt):
            print(f"  phase {idx:02d}: source checkpoint missing "
                  f"({source_ckpt}) -- skipping.")
            continue
        phase_dir = os.path.join(output_root, f"phase_{idx:02d}")
        os.makedirs(phase_dir, exist_ok=True)

        resume_ckpt = None
        if args.resume:
            resume_ckpt = _find_latest_checkpoint(phase_dir)
            if resume_ckpt:
                print(f"  phase {idx:02d}: resume from {resume_ckpt}")
            else:
                print(f"  phase {idx:02d}: no prior latest.pt; init from "
                      f"source {source_ckpt}")

        seed = secrets.randbelow(MAX_SEED)
        script = _build_sbatch(args, idx, source_ckpt, phase_dir,
                               resume_ckpt, seed)

        if args.dry_run:
            print(f"--- phase {idx:02d} sbatch script ---")
            print(textwrap.indent(script, "    "))
            continue

        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  phase {idx:02d}: sbatch FAILED -- "
                  f"{result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        submitted.append((idx, job_id, phase_dir))
        print(f"  phase {idx:02d}: submitted job {job_id} -> {phase_dir}")

    if args.dry_run:
        print(f"\nWould submit {len(target_indices)} job(s) under {output_root}")
        return

    print(f"\nSubmitted {len(submitted)}/{len(target_indices)} job(s).")
    if submitted:
        print(f"Monitor:  squeue -u $USER | grep szdm-")
        print(f"Tail:     tail -f {output_root}/phase_*/_logs/sbatch-*.out")


if __name__ == "__main__":
    main()
