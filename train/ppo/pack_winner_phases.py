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
from typing import Optional


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


def _phase_from_seed_dir(seed_dir: Path) -> int:
    """Infer the phase index from an autotrain seed-dir path. Expects
    the layout ``.../phase_NN/seed_SSSS/``. Returns the int NN."""
    parent = seed_dir.parent.name
    if not parent.startswith("phase_"):
        raise ValueError(
            f"--seed-dir parent must be named phase_NN, got "
            f"{parent!r} (full path: {seed_dir})")
    try:
        return int(parent.split("_", 1)[1])
    except ValueError as e:
        raise ValueError(
            f"could not parse phase index from {parent!r}: {e}")


def _seed_from_seed_dir(seed_dir: Path) -> Optional[int]:
    """Pull the seed value out of ``seed_SSSS`` for manifest
    bookkeeping. None when the dir isn't named that way."""
    name = seed_dir.name
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def _best_or_latest_in_seed_dir(seed_dir: Path,
                                prefer: str = "best") -> Optional[Path]:
    """Find the most-recently-modified ``best.pt`` (or ``latest.pt``)
    under ``<seed_dir>/ppo_optomech_*/checkpoints/``. Returns None if
    the dir layout is unexpected or no candidates exist."""
    cands = []
    primary = prefer
    fallback = "latest" if prefer == "best" else "best"
    for rd in seed_dir.glob("ppo_optomech_*"):
        for name in (f"{primary}.pt", f"{fallback}.pt"):
            ck = rd / "checkpoints" / name
            if ck.is_file():
                cands.append(ck)
                break
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _update_manifest_for_phase(manifest_path: Path, dst_dir: Path,
                               phase: int,
                               src_ckpt: Path,
                               history_entry: dict) -> dict:
    """Read (or seed) the manifest, mark ``phase`` as present, append
    a history entry, write back. Returns the updated manifest dict."""
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "agent_name": dst_dir.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
            "description": (
                "Winner-only segment_dm_v2 agent. Contains only the "
                "per-phase checkpoints that passed evaluation, "
                "incrementally merged."),
            "num_phases_target": _NUM_PHASES,
            "phases_present": [],
            "phases_missing": list(range(_NUM_PHASES)),
            "pack_history": [],
        }
    present = set(int(x) for x in manifest.get("phases_present", []))
    missing = set(int(x) for x in manifest.get("phases_missing", []))
    present.add(phase)
    missing.discard(phase)
    manifest["phases_present"] = sorted(present)
    manifest["phases_missing"] = sorted(missing)
    manifest.setdefault("pack_history", []).append(history_entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _write_readme(dst_dir: Path, manifest: dict) -> None:
    """Refresh the dst README with the current present/missing
    counts. Called after every successful promote."""
    present = manifest.get("phases_present", [])
    missing = manifest.get("phases_missing", [])
    readme = f"""# {dst_dir.name}

Winner-only segment_dm_v2 agent (incrementally merged).

phases present ({len(present)}/{_NUM_PHASES}): {sorted(present)}
phases missing ({len(missing)}/{_NUM_PHASES}): {sorted(missing)}

This agent is INCOMPLETE while phases are missing. Rollouts via
the standard composite spec will fail on any missing phase. Use
only after every phase is present, or run rollout_per_phase.py
against this dir to test the present ones individually.

To complete: re-train the missing phases (e.g. via the autotrainer)
and promote the winning seed for each with::

    poetry run python train/ppo/pack_winner_phases.py \\
        --seed-dir <autotrain_run>/phase_NN/seed_SSSS/ \\
        --dst {dst_dir} \\
        --reason "<why this seed was accepted>"
"""
    with open(dst_dir / "README.md", "w") as f:
        f.write(readme)


def _promote_one_seed(seed_dir: Path, dst_dir: Path,
                      phase_override: Optional[int],
                      prefer: str, reason: Optional[str],
                      force: bool) -> None:
    """One-seed promote: copy <seed_dir>/ppo_optomech_*/checkpoints/
    {best,latest}.pt into the dst's checkpoints/phase_NN.pt and
    update the manifest. Phase is inferred from the seed_dir's
    parent dir name unless ``phase_override`` is set."""
    seed_dir = seed_dir.resolve()
    if not seed_dir.is_dir():
        print(f"ERROR: --seed-dir not a directory: {seed_dir}")
        sys.exit(1)
    phase = (phase_override if phase_override is not None
             else _phase_from_seed_dir(seed_dir))
    seed_val = _seed_from_seed_dir(seed_dir)
    src_ckpt = _best_or_latest_in_seed_dir(seed_dir, prefer=prefer)
    if src_ckpt is None:
        print(f"ERROR: no {prefer}.pt or fallback in "
              f"{seed_dir}/ppo_optomech_*/checkpoints/")
        sys.exit(1)

    dst_dir = dst_dir.resolve()
    ck_dir = dst_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    dst_ck = ck_dir / f"phase_{phase:02d}.pt"
    if dst_ck.is_file() and not force:
        print(f"ERROR: phase {phase} already promoted in {dst_dir} "
              f"({dst_ck}). Pass --force to overwrite.")
        sys.exit(1)
    shutil.copyfile(src_ckpt, dst_ck)
    size_mb = dst_ck.stat().st_size / 1024 / 1024
    print(f"  copied {src_ckpt} -> {dst_ck} ({size_mb:.1f} MB)")

    history_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "promote_seed",
        "phase_winners": [{
            "phase": phase,
            "seed": seed_val,
            "source_checkpoint": str(src_ckpt),
            "source_seed_dir": str(seed_dir),
            "reason": reason or "(not provided)",
        }],
    }
    manifest = _update_manifest_for_phase(
        dst_dir / "manifest.json", dst_dir, phase, src_ckpt,
        history_entry)
    _write_readme(dst_dir, manifest)

    print()
    print(f"  phase {phase} promoted (seed {seed_val}).")
    print(f"  phases present: {manifest['phases_present']}")
    print(f"  phases missing: {manifest['phases_missing']}")


def _dir_size_bytes(path: Path) -> int:
    """Recursive byte total of a directory. Used for the cleanup
    summary so the user knows what they're about to free."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _clean_promoted(scan_dir: Path, winners_dir: Path,
                    dry_run: bool) -> None:
    """Delete every ``<scan_dir>/phase_NN/`` whose NN appears in
    ``<winners_dir>/manifest.json`` phases_present.

    Defensive: refuses to run if winners_dir's manifest can't be
    read, since the safest interpretation of "no manifest" is "no
    phases promoted" -- which would delete nothing -- but we'd
    rather fail loudly than silently no-op.
    """
    scan_dir = scan_dir.resolve()
    winners_dir = winners_dir.resolve()

    if not scan_dir.is_dir():
        print(f"ERROR: --clean-promoted dir not a directory: {scan_dir}")
        sys.exit(1)

    manifest_path = winners_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: winners manifest not found at {manifest_path}. "
              f"Refusing to clean -- without the manifest we don't "
              f"know which phases are safe to delete.")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    present = set(int(x) for x in manifest.get("phases_present", []))
    if not present:
        print(f"Winners agent has no phases promoted yet "
              f"(phases_present empty). Nothing to clean.")
        return

    print(f"Cleanup mode  ({'DRY-RUN' if dry_run else 'LIVE'})")
    print(f"  scan dir:          {scan_dir}")
    print(f"  winners manifest:  {manifest_path}")
    print(f"  phases_present:    {sorted(present)}")
    print()

    # Iterate over every phase_NN dir under the scan dir. Match by
    # numeric suffix; ignore non-conforming entries.
    candidates: list[tuple[int, Path, int]] = []
    skipped: list[Path] = []
    for entry in sorted(scan_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("phase_"):
            continue
        try:
            idx = int(name.split("_", 1)[1])
        except ValueError:
            continue
        if idx not in present:
            skipped.append(entry)
            continue
        size = _dir_size_bytes(entry)
        candidates.append((idx, entry, size))

    if skipped:
        print(f"  Skipped (not yet promoted):")
        for s in skipped:
            print(f"    {s.name}")
        print()

    if not candidates:
        print("  No promoted-phase dirs found under scan dir. "
              "Nothing to do.")
        return

    total = sum(s for _, _, s in candidates)
    print(f"  Candidates to remove ({len(candidates)}):")
    for idx, path, size in candidates:
        print(f"    phase_{idx:02d}   {_fmt_size(size):>10}   {path}")
    print(f"  TOTAL would free: {_fmt_size(total)}")
    print()

    if dry_run:
        print("  --dry-run set; not deleting. Re-run without "
              "--dry-run to apply.")
        return

    for idx, path, _ in candidates:
        print(f"  rm -rf {path}")
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"    FAILED: {e}")
    print(f"\n  Done. Reclaimed ~{_fmt_size(total)}.")


def main():
    p = argparse.ArgumentParser(
        description="Snapshot per-phase checkpoints into a winner-"
                    "only agent dir.\n\n"
                    "Two modes:\n"
                    "  1. Bulk promote from an eval summary (default).\n"
                    "  2. Single-seed promote via --seed-dir (for one-"
                    "off accepts after a fail-but-close autotrain "
                    "run).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-summary", default=None,
                   help="Path to summary.json from rollout_per_phase.py "
                        "(bulk mode). Use with --source.")
    p.add_argument("--source", default=None,
                   help="Bulk mode: dir to pull checkpoints from "
                        "(agent-dir or run-dir layout, auto-detected).")
    p.add_argument("--seed-dir", default=None,
                   help="Single-seed promote mode: path to one "
                        "autotrain seed dir, e.g. <autotrain_run>/"
                        "phase_03/seed_167681808/. Phase index is "
                        "auto-detected from the parent dir name "
                        "(override with --phase). The most recently "
                        "modified best.pt under <seed-dir>/ppo_"
                        "optomech_*/checkpoints/ is promoted; falls "
                        "back to latest.pt if best.pt isn't there.")
    p.add_argument("--phase", type=int, default=None,
                   help="Explicit phase index for --seed-dir mode. "
                        "Default: inferred from seed-dir parent.")
    p.add_argument("--prefer", choices=["best", "latest"],
                   default="best",
                   help="Which checkpoint flavour to prefer in "
                        "--seed-dir mode (default: best, fall back "
                        "to latest if missing).")
    p.add_argument("--reason", default=None,
                   help="Optional one-line note logged into the "
                        "manifest's pack_history for this promotion. "
                        "Use it to record why a sub-threshold seed "
                        "was accepted.")
    p.add_argument("--dst", required=True,
                   help="Destination agent dir. Created if missing; "
                        "merged into if it already exists (skip "
                        "phases already present unless --force).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite phase checkpoints already in --dst.")
    p.add_argument("--clean-promoted", default=None, metavar="DIR",
                   help="Cleanup mode: scan DIR (an autotrain run "
                        "root with phase_NN/ subdirs) and delete each "
                        "phase subdir whose index is already in the "
                        "--dst winners agent's phases_present list. "
                        "Reclaims the disk used by losing seeds + "
                        "stale checkpoints once a phase is locked "
                        "in. Combine with --dry-run to preview.")
    p.add_argument("--dry-run", action="store_true",
                   help="In --clean-promoted mode, list what would be "
                        "removed without actually removing.")
    args = p.parse_args()

    # Mode dispatch: cleanup mode is exclusive (no promotion);
    # --seed-dir takes precedence over bulk mode for promotions.
    if args.clean_promoted:
        _clean_promoted(
            scan_dir=Path(args.clean_promoted),
            winners_dir=Path(args.dst),
            dry_run=args.dry_run)
        return

    if args.seed_dir:
        if args.eval_summary or args.source:
            print("WARN: --eval-summary / --source ignored in "
                  "--seed-dir mode.")
        _promote_one_seed(
            Path(args.seed_dir),
            Path(args.dst),
            phase_override=args.phase,
            prefer=args.prefer,
            reason=args.reason,
            force=args.force)
        return

    if not args.eval_summary or not args.source:
        p.error("either --seed-dir, --clean-promoted, or both "
                "--eval-summary and --source, must be provided.")

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
