#!/usr/bin/env python3
"""
Walk a directory tree, find every TensorBoard event file
(``events.out.tfevents.*``), and write a *scalar-only* copy alongside
it under a ``tb_scalars/`` subdir. The result is typically 10–100x
smaller than the original because PPO training scripts log
``add_figure`` PNGs and ``add_histogram`` payloads that dominate
event-file size but are useless for monitoring training curves.

Pair with ``utils/sync_remote_runs.py --dark-hole --scalars-only``
(once that flag is wired in) to only transfer the slim files.

Usage:
    # Extract scalars under one sweep dir
    python utils/extract_tb_scalars.py dark_hole_runs/dark_hole_1776xxxx

    # Extract under multiple roots
    python utils/extract_tb_scalars.py dark_hole_runs/ bootstrap_runs/

    # Force regeneration even if scalars file is newer than source
    python utils/extract_tb_scalars.py --force dark_hole_runs/

    # Dry run: list source -> destination pairs without writing
    python utils/extract_tb_scalars.py --dry-run dark_hole_runs/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Tensorboard reader + writer.
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch.utils.tensorboard import SummaryWriter


def find_event_files(root: Path):
    """Yield every events.out.tfevents.* file under root, skipping any
    that already live inside a tb_scalars/ output directory."""
    for path in root.rglob("events.out.tfevents.*"):
        if "tb_scalars" in path.parts:
            continue
        yield path


def extract_one(src: Path, force: bool, dry_run: bool):
    """Write a scalar-only event file to ``<src.parent>/tb_scalars/``.

    Skips work if the destination already exists and is newer than the
    source (unless ``force`` is set)."""
    dst_dir = src.parent / "tb_scalars"
    # Use a stable destination filename so re-runs overwrite.
    dst = dst_dir / "events.out.tfevents.scalars"

    if dry_run:
        print(f"  {src}\n    -> {dst}")
        return

    if (
        not force
        and dst.exists()
        and dst.stat().st_mtime >= src.stat().st_mtime
    ):
        print(f"  skip (up-to-date): {src}")
        return

    print(f"  extract: {src}")
    # Read scalars only. size_guidance=0 for scalars means "load all".
    acc = EventAccumulator(
        str(src), size_guidance={"scalars": 0, "tensors": 0,
                                 "histograms": 1, "images": 1,
                                 "audio": 1, "compressedHistograms": 1,
                                 "graph": 0, "meta_graph": 0,
                                 "run_metadata": 0}
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if not tags:
        print(f"    (no scalars; skipping)")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any prior partial output so the writer creates one clean file.
    for old in dst_dir.glob("events.out.tfevents.*"):
        old.unlink()

    writer = SummaryWriter(str(dst_dir), filename_suffix=".scalars")
    for tag in tags:
        for ev in acc.Scalars(tag):
            writer.add_scalar(tag, ev.value, ev.step, walltime=ev.wall_time)
    writer.flush()
    writer.close()

    # Rename writer's auto-generated file to the canonical destination.
    written = sorted(dst_dir.glob("events.out.tfevents.*"))
    if not written:
        print(f"    (writer produced nothing; skipping)")
        return
    if len(written) > 1:
        # Keep the newest, delete others.
        written.sort(key=lambda p: p.stat().st_mtime)
        for old in written[:-1]:
            old.unlink()
        written = [written[-1]]
    written[0].rename(dst)
    sz_src = src.stat().st_size
    sz_dst = dst.stat().st_size
    ratio = sz_src / max(sz_dst, 1)
    print(f"    {sz_src/1e6:.1f} MB -> {sz_dst/1e6:.2f} MB  ({ratio:.0f}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Directories to walk for tfevents files.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the destination is up to date.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written, don't write.")
    args = parser.parse_args()

    n_total = 0
    for root in args.roots:
        if not root.exists():
            print(f"WARN: {root} does not exist; skipping.", file=sys.stderr)
            continue
        print(f"\n[{root}]")
        files = list(find_event_files(root))
        if not files:
            print("  (no event files found)")
            continue
        for src in files:
            extract_one(src, force=args.force, dry_run=args.dry_run)
            n_total += 1

    print(f"\nProcessed {n_total} event file(s).")


if __name__ == "__main__":
    main()
