#!/usr/bin/env python
"""Snapshot the passing per-phase checkpoints into a new agent dir.

Reads the ``summary.json`` produced by ``rollout_per_phase.py``,
copies each ``passed_phases`` entry's source checkpoint into
``<dst>/checkpoints/phase_NN.pt``, and writes a manifest +
partial-spec README documenting which phases are present and
which are still missing.

The partial agent is a holding state -- it preserves the winning
phases so the next training pass (which retrains only the failed
phases) doesn't risk clobbering them. Once the retrain produces
working failed-phase checkpoints, re-run this script with the new
eval + source pointing at the retrain dir, and it will merge the
new checkpoints into the same agent dir alongside the existing
winners.

Usage::

    poetry run python train/ppo/pack_winner_phases.py \\
        --eval-summary test_output/per_phase_eval_<ts>/summary.json \\
        --source agents/segment_dm_agent_v2 \\
        --dst agents/segment_dm_agent_v2_winners

    # After retraining the failed phases, merge new winners into the
    # same dst (skip already-present phases unless --force):
    poetry run python train/ppo/pack_winner_phases.py \\
        --eval-summary test_output/per_phase_eval_<ts2>/summary.json \\
        --source segment_dm_finetuning/<retrain-run>/ \\
        --dst agents/segment_dm_agent_v2_winners
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


_NUM_PHASES = 15


def _find_source_checkpoint(source: str, phase_idx: int) -> Path | None:
    """Locate the source checkpoint for ``phase_idx`` under ``source``.

    Supports both agent-dir layout (checkpoints/phase_NN.pt) and
    run-dir layout (phases/phase_NN/ppo_optomech_*/checkpoints/
    best.pt). Returns None when nothing matches; caller decides
    whether to skip or error.
    """
    src = Path(source).resolve()

    flat = src / "checkpoints" / f"phase_{phase_idx:02d}.pt"
    if flat.is_file():
        return flat

    deep_dir = src / "phases" / f"phase_{phase_idx:02d}"
    if deep_dir.is_dir():
        run_dirs = sorted(deep_dir.glob("ppo_optomech_*"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)
        for rd in run_dirs:
            for candidate in ("best.pt", "latest.pt"):
                ck = rd / "checkpoints" / candidate
                if ck.is_file():
                    return ck

    return None


def main():
    p = argparse.ArgumentParser(
        description="Snapshot passing per-phase checkpoints into a "
                    "winner-only agent dir.")
    p.add_argument("--eval-summary", required=True,
                   help="Path to summary.json from rollout_per_phase.py.")
    p.add_argument("--source", required=True,
                   help="Directory to pull checkpoints from "
                        "(agent-dir or run-dir layout, auto-detected).")
    p.add_argument("--dst", required=True,
                   help="Destination agent dir. Created if missing; "
                        "merged into if it already exists (skip phases "
                        "already present unless --force).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite phase checkpoints already in --dst.")
    args = p.parse_args()

    with open(args.eval_summary) as f:
        summary = json.load(f)
    passed = list(summary.get("passed_phases", []))
    failed = list(summary.get("failed_phases", []))
    print(f"Eval summary {args.eval_summary}:")
    print(f"  passed: {passed}")
    print(f"  failed: {failed}")

    if not passed:
        print("\nNo passed phases to pack. Exiting.")
        sys.exit(0)

    dst = Path(args.dst).resolve()
    ck_dir = dst / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)

    # Load any pre-existing manifest so we can merge incrementally.
    manifest_path = dst / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "agent_name": dst.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
            "description": (
                "Winner-only segment_dm_v2 agent. Contains only the "
                "per-phase checkpoints that passed the per-phase Strehl "
                "eval. Incrementally merged: re-run pack_winner_phases "
                "with a new eval summary + source dir to add more "
                "phases."),
            "num_phases_target": _NUM_PHASES,
            "phases_present": [],
            "phases_missing": list(range(_NUM_PHASES)),
            "pack_history": [],
        }

    present = set(manifest.get("phases_present", []))
    missing = set(range(_NUM_PHASES)) - present

    copied = []
    skipped_existing = []
    not_found = []
    for phase_idx in passed:
        dst_ck = ck_dir / f"phase_{phase_idx:02d}.pt"
        if dst_ck.is_file() and not args.force:
            skipped_existing.append(phase_idx)
            continue
        src_ck = _find_source_checkpoint(args.source, phase_idx)
        if src_ck is None:
            not_found.append(phase_idx)
            print(f"  phase_{phase_idx:02d}: source ckpt NOT FOUND under "
                  f"{args.source}")
            continue
        shutil.copyfile(src_ck, dst_ck)
        size = dst_ck.stat().st_size
        copied.append((phase_idx, str(src_ck), size))
        present.add(phase_idx)
        missing.discard(phase_idx)
        print(f"  phase_{phase_idx:02d}: copied {src_ck} -> {dst_ck} "
              f"({size / 1024 / 1024:.1f} MB)")

    if skipped_existing:
        print(f"\nSkipped (already present, --force to overwrite): "
              f"{skipped_existing}")

    # Update manifest.
    manifest["phases_present"] = sorted(present)
    manifest["phases_missing"] = sorted(missing)
    manifest["pack_history"].append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_summary": str(Path(args.eval_summary).resolve()),
        "source": str(Path(args.source).resolve()),
        "passed_phases_in_eval": passed,
        "failed_phases_in_eval": failed,
        "phases_copied": [c[0] for c in copied],
        "phases_skipped_existing": skipped_existing,
        "phases_source_not_found": not_found,
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    readme = f"""# {dst.name}

Winner-only segment_dm_v2 agent (incrementally merged).

phases present ({len(present)}/{_NUM_PHASES}): {sorted(present)}
phases missing ({len(missing)}/{_NUM_PHASES}): {sorted(missing)}

This agent is INCOMPLETE while phases are missing. Rollouts via the
standard composite spec will fail on any missing phase. Use only
after every phase is present, or run rollout_per_phase.py against
this dir to test the present ones individually.

To complete: re-train the missing phases and re-run
pack_winner_phases.py with the new eval summary + retrain dir as
--source.
"""
    with open(dst / "README.md", "w") as f:
        f.write(readme)

    print(f"\n{'=' * 60}")
    print(f"  agent dir:        {dst}")
    print(f"  phases present:   {sorted(present)}")
    print(f"  phases missing:   {sorted(missing)}")
    print(f"{'=' * 60}")
    if missing:
        print(f"\nNext step: retrain missing phases. Pass them to "
              f"launch_segment_zernike_dm_transfer with --phases:")
        print(f"  --phases {','.join(str(p) for p in sorted(missing))}")


if __name__ == "__main__":
    main()
