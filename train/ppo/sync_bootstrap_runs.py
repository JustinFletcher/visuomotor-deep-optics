#!/usr/bin/env python
"""Sync bootstrap training outputs into runs/ for local inspection.

Copies only TensorBoard logs and the latest checkpoint from
bootstrap_runs/{run_id}/ into runs/, keeping the bulk training data
out of the main runs directory.

Usage:
    # Sync a specific run
    python train/ppo/sync_bootstrap_runs.py --run-id bootstrap_1742845200

    # Sync all bootstrap runs
    python train/ppo/sync_bootstrap_runs.py --all

    # Dry run
    python train/ppo/sync_bootstrap_runs.py --run-id bootstrap_1742845200 --dry-run
"""

import argparse
import glob
import os
import re
import shutil
import sys


BOOTSTRAP_ROOT = "bootstrap_runs"
RUNS_ROOT = "runs"


def find_training_subdir(phase_dir):
    """Find the ppo_optomech_* subdir inside a bootstrap phase dir."""
    candidates = glob.glob(os.path.join(phase_dir, "ppo_optomech_*"))
    if not candidates:
        return None
    # If multiple, take the most recently modified
    return max(candidates, key=os.path.getmtime)


def find_latest_checkpoint(ckpt_dir):
    """Find the highest-numbered update_N.pt checkpoint."""
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r"update_(\d+)\.pt$")
    best_num = -1
    best_path = None
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            num = int(m.group(1))
            if num > best_num:
                best_num = num
                best_path = os.path.join(ckpt_dir, f)
    return best_path


def sync_phase(phase_dir, dest_dir, dry_run=False):
    """Sync one phase's TB logs and latest checkpoint into dest_dir.

    Returns dict with sync status info.
    """
    info = {
        "phase_dir": phase_dir,
        "dest_dir": dest_dir,
        "status": "missing",
        "latest_ckpt": None,
        "has_best": False,
        "tb_synced": False,
    }

    train_dir = find_training_subdir(phase_dir)
    if train_dir is None:
        return info

    info["status"] = "found"

    # Sync TensorBoard logs
    tb_src = os.path.join(train_dir, "tensorboard")
    tb_dst = os.path.join(dest_dir, "tensorboard")
    if os.path.isdir(tb_src):
        if not dry_run:
            os.makedirs(tb_dst, exist_ok=True)
            # Copy all event files (overwrite if newer)
            for f in os.listdir(tb_src):
                src = os.path.join(tb_src, f)
                dst = os.path.join(tb_dst, f)
                if os.path.isfile(src):
                    if (not os.path.exists(dst) or
                            os.path.getmtime(src) > os.path.getmtime(dst)):
                        shutil.copy2(src, dst)
        info["tb_synced"] = True

    # Sync checkpoints: best.pt + latest update_N.pt
    ckpt_src = os.path.join(train_dir, "checkpoints")
    ckpt_dst = os.path.join(dest_dir, "checkpoints")

    if os.path.isdir(ckpt_src):
        if not dry_run:
            os.makedirs(ckpt_dst, exist_ok=True)

        # best.pt
        best_src = os.path.join(ckpt_src, "best.pt")
        if os.path.isfile(best_src):
            info["has_best"] = True
            if not dry_run:
                shutil.copy2(best_src, os.path.join(ckpt_dst, "best.pt"))

        # Latest numbered checkpoint
        latest = find_latest_checkpoint(ckpt_src)
        if latest:
            info["latest_ckpt"] = os.path.basename(latest)
            if not dry_run:
                shutil.copy2(latest, os.path.join(ckpt_dst,
                             os.path.basename(latest)))

    return info


def sync_run(run_id, dry_run=False):
    """Sync all phases of a bootstrap run."""
    run_dir = os.path.join(BOOTSTRAP_ROOT, run_id)
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} does not exist")
        return False

    # Find all bootstrap_phase_XX dirs
    phase_dirs = sorted(glob.glob(os.path.join(run_dir, "bootstrap_phase_*")))
    if not phase_dirs:
        print(f"No phases found in {run_dir}")
        return False

    print(f"Syncing run '{run_id}' ({len(phase_dirs)} phases found)")
    if dry_run:
        print("  (dry run — no files will be copied)")
    print()

    # Header
    print(f"  {'Phase':>7}  {'Status':>8}  {'TB':>4}  {'Best':>5}  "
          f"{'Latest Checkpoint':<25}")
    print(f"  {'-----':>7}  {'------':>8}  {'--':>4}  {'----':>5}  "
          f"{'-' * 25}")

    results = []
    for phase_dir in phase_dirs:
        phase_name = os.path.basename(phase_dir)
        # Extract phase number for dest naming
        dest_name = f"{run_id}_{phase_name}"
        dest_dir = os.path.join(RUNS_ROOT, dest_name)

        info = sync_phase(phase_dir, dest_dir, dry_run=dry_run)
        results.append(info)

        # Extract short phase label
        phase_num = phase_name.replace("bootstrap_phase_", "")
        tb_mark = "yes" if info["tb_synced"] else "no"
        best_mark = "yes" if info["has_best"] else "no"
        ckpt_name = info["latest_ckpt"] or "-"

        print(f"  {phase_num:>7}  {info['status']:>8}  {tb_mark:>4}  "
              f"{best_mark:>5}  {ckpt_name:<25}")

    found = sum(1 for r in results if r["status"] == "found")
    print(f"\n  {found}/{len(phase_dirs)} phases synced to {RUNS_ROOT}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync bootstrap runs into runs/ for local inspection")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Bootstrap run ID to sync")
    parser.add_argument(
        "--all", action="store_true",
        help="Sync all bootstrap runs found in bootstrap_runs/")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be synced without copying files")
    cli = parser.parse_args()

    if not cli.run_id and not cli.all:
        # List available runs
        if os.path.isdir(BOOTSTRAP_ROOT):
            runs = sorted(os.listdir(BOOTSTRAP_ROOT))
            runs = [r for r in runs if os.path.isdir(
                os.path.join(BOOTSTRAP_ROOT, r))]
        else:
            runs = []

        if runs:
            print(f"Available runs in {BOOTSTRAP_ROOT}/:")
            for r in runs:
                phases = glob.glob(os.path.join(
                    BOOTSTRAP_ROOT, r, "bootstrap_phase_*"))
                print(f"  {r}  ({len(phases)} phases)")
            print(f"\nUse --run-id <name> or --all to sync.")
        else:
            print(f"No bootstrap runs found in {BOOTSTRAP_ROOT}/")
        sys.exit(0)

    if cli.all:
        if not os.path.isdir(BOOTSTRAP_ROOT):
            print(f"No {BOOTSTRAP_ROOT}/ directory found")
            sys.exit(1)
        run_ids = sorted([
            d for d in os.listdir(BOOTSTRAP_ROOT)
            if os.path.isdir(os.path.join(BOOTSTRAP_ROOT, d))
        ])
        for run_id in run_ids:
            sync_run(run_id, dry_run=cli.dry_run)
            print()
    else:
        sync_run(cli.run_id, dry_run=cli.dry_run)


if __name__ == "__main__":
    main()
