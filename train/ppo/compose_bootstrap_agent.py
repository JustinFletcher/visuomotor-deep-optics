#!/usr/bin/env python
"""Compose a CompositeAgent from 15 trained bootstrap phase models.

Copies checkpoints from bootstrap_runs/ into a single directory and
writes a YAML policy spec for use with rollout.py and sweep_tiptilt.py.

Supports cherry-picking phases from different bootstrap runs.

Usage:
    # Auto-detect the most recent bootstrap run in bootstrap_runs/
    python train/ppo/compose_bootstrap_agent.py

    # All 15 from one run (explicit)
    python train/ppo/compose_bootstrap_agent.py --run-id bootstrap_1742845200

    # Cherry-pick phases from different runs
    python train/ppo/compose_bootstrap_agent.py \\
        --phase 0-7:bootstrap_1742845200 \\
        --phase 8-14:bootstrap_1742850000

    # Custom trigger and output
    python train/ppo/compose_bootstrap_agent.py \\
        --run-id bootstrap_1742845200 \\
        --trigger "metric_above strehl 0.8" \\
        --output-dir train/ppo/specs/my_composed_agent

    # Allow partial (fewer than 15 phases)
    python train/ppo/compose_bootstrap_agent.py \\
        --run-id bootstrap_1742845200 --allow-partial

    # Per-phase checkpoint name override (default is --checkpoint-name
    # for all phases; override bakes into the spec so the result is
    # portable without further flags at rollout time):
    python train/ppo/compose_bootstrap_agent.py \\
        --phase-checkpoint 0=history_update_2500.pt \\
        --phase-checkpoint 7=history_update_9800.pt
"""

import argparse
import glob
import os
import re
import shutil
import sys

import yaml

NUM_PHASES = 15
BOOTSTRAP_ROOT = "bootstrap_runs"
DEFAULT_OUTPUT_DIR = "train/ppo/specs/bootstrap_composed"


def find_training_subdir(phase_dir):
    """Find the ppo_optomech_* subdir inside a bootstrap phase dir."""
    candidates = glob.glob(os.path.join(phase_dir, "ppo_optomech_*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def discover_default_run_id():
    """Return the most recently modified bootstrap run id, or None.

    Looks inside ``bootstrap_runs/`` for directories matching
    ``bootstrap_*`` and picks the one with the latest mtime. If there
    is exactly one run the choice is unambiguous; with multiple, the
    newest wins and a warning is printed so the caller can override
    with ``--run-id`` if needed.
    """
    if not os.path.isdir(BOOTSTRAP_ROOT):
        return None
    candidates = [
        d for d in os.listdir(BOOTSTRAP_ROOT)
        if d.startswith("bootstrap_")
        and os.path.isdir(os.path.join(BOOTSTRAP_ROOT, d))
    ]
    if not candidates:
        return None
    paths = [os.path.join(BOOTSTRAP_ROOT, d) for d in candidates]
    latest = max(paths, key=os.path.getmtime)
    return os.path.basename(latest)


def find_checkpoint(phase_dir, checkpoint_name="best.pt"):
    """Locate the requested checkpoint in a phase directory."""
    train_dir = find_training_subdir(phase_dir)
    if train_dir is None:
        return None
    ckpt_path = os.path.join(train_dir, "checkpoints", checkpoint_name)
    if os.path.isfile(ckpt_path):
        return ckpt_path

    # Fall back: if checkpoint_name is "best.pt" and not found, try latest
    if checkpoint_name == "best.pt":
        ckpt_dir = os.path.join(train_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None
        pattern = re.compile(r"update_(\d+)\.pt$")
        best_num = -1
        best_path = None
        for f in os.listdir(ckpt_dir):
            m = pattern.match(f)
            if m and int(m.group(1)) > best_num:
                best_num = int(m.group(1))
                best_path = os.path.join(ckpt_dir, f)
        return best_path
    return None


def parse_trigger(trigger_str):
    """Parse trigger string into YAML-compatible dict.

    Formats:
        "step 32"                    -> {"step": 32}
        "metric_above strehl 0.8"    -> {"metric_above": {"strehl": 0.8}}
        "metric_below mse 0.01"     -> {"metric_below": {"mse": 0.01}}
        "episode_fraction 0.5"      -> {"episode_fraction": 0.5}
    """
    parts = trigger_str.strip().split()
    kind = parts[0]

    if kind == "step":
        return {"step": int(parts[1])}
    elif kind in ("metric_above", "metric_below"):
        return {kind: {parts[1]: float(parts[2])}}
    elif kind == "episode_fraction":
        return {"episode_fraction": float(parts[1])}
    else:
        raise ValueError(f"Unknown trigger type: {kind}")


def parse_phase_spec(spec_str):
    """Parse 'RANGE:RUN_ID' into (list_of_ints, run_id).

    Examples:
        '0-7:bootstrap_123' -> ([0,1,2,3,4,5,6,7], 'bootstrap_123')
        '3:bootstrap_456'   -> ([3], 'bootstrap_456')
    """
    range_str, run_id = spec_str.rsplit(":", 1)
    phases = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            phases.update(range(int(lo), int(hi) + 1))
        else:
            phases.add(int(part))
    return sorted(phases), run_id


def main():
    parser = argparse.ArgumentParser(
        description="Compose a CompositeAgent from bootstrap phase models")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Use all 15 phases from this bootstrap run")
    parser.add_argument(
        "--phase", type=str, action="append", default=[],
        help="Cherry-pick phases: RANGE:RUN_ID (repeatable). "
             "E.g., '0-7:bootstrap_123' or '8-14:bootstrap_456'")
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for checkpoints + YAML (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--trigger", type=str, default="step 32",
        help="Phase transition trigger (default: 'step 32'). "
             "Options: 'step N', 'metric_above KEY VAL', "
             "'metric_below KEY VAL', 'episode_fraction F'")
    parser.add_argument(
        "--checkpoint-name", type=str, default="best.pt",
        help="Default checkpoint filename per phase (default: best.pt). "
             "Per-phase overrides via --phase-checkpoint.")
    parser.add_argument(
        "--phase-checkpoint", action="append", default=[], metavar="N=NAME",
        help="Override the checkpoint filename for phase N. NAME is "
             "resolved inside that phase's checkpoints/ directory "
             "(e.g. 0=history_update_2500.pt) — or an absolute/"
             "repo-relative path to use that file verbatim. Repeatable. "
             "Bakes the selection into the output spec so rollouts are "
             "portable without further flags.")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Proceed even if some phases are missing")
    cli = parser.parse_args()

    # Parse --phase-checkpoint entries into {int: str}.
    phase_ckpt_overrides: dict[int, str] = {}
    for entry in cli.phase_checkpoint:
        if "=" not in entry:
            parser.error(
                f"--phase-checkpoint expects N=NAME, got {entry!r}")
        k, v = entry.split("=", 1)
        try:
            idx = int(k)
        except ValueError:
            parser.error(f"Phase index must be int, got {k!r}")
        if not 0 <= idx < NUM_PHASES:
            parser.error(
                f"Phase index {idx} out of range [0, {NUM_PHASES-1}]")
        phase_ckpt_overrides[idx] = v

    # Build phase → run_id mapping
    phase_map = {}  # phase_num -> run_id

    if cli.run_id and cli.phase:
        parser.error("Use --run-id OR --phase, not both")

    if cli.run_id:
        for p in range(NUM_PHASES):
            phase_map[p] = cli.run_id
    elif cli.phase:
        for spec in cli.phase:
            phases, run_id = parse_phase_spec(spec)
            for p in phases:
                if p in phase_map:
                    print(f"Warning: phase {p} specified multiple times, "
                          f"using {run_id}")
                phase_map[p] = run_id
    else:
        # No --run-id or --phase provided: auto-detect from bootstrap_runs/
        auto_run_id = discover_default_run_id()
        if auto_run_id is None:
            parser.error(
                f"No --run-id or --phase given and no bootstrap runs "
                f"found in {BOOTSTRAP_ROOT}/. Sync a run first or pass "
                f"one of --run-id / --phase explicitly.")
        print(f"Auto-detected run id: {auto_run_id}")
        for p in range(NUM_PHASES):
            phase_map[p] = auto_run_id

    # Parse trigger
    try:
        trigger = parse_trigger(cli.trigger)
    except (ValueError, IndexError) as e:
        print(f"Error parsing trigger '{cli.trigger}': {e}")
        sys.exit(1)

    # Resolve checkpoints
    print(f"Output directory: {cli.output_dir}")
    print(f"Trigger: {trigger}")
    print(f"Checkpoint: {cli.checkpoint_name}")
    print()

    resolved = {}  # phase_num -> source_path
    missing = []

    for phase in sorted(phase_map.keys()):
        run_id = phase_map[phase]
        phase_dir = os.path.join(
            BOOTSTRAP_ROOT, run_id, f"bootstrap_phase_{phase:02d}")

        # Pick the checkpoint name for this phase: override wins over
        # the global default.
        ckpt_name = phase_ckpt_overrides.get(phase, cli.checkpoint_name)
        tag = "" if phase not in phase_ckpt_overrides else " [override]"

        # Override value may be an explicit path (absolute or containing
        # a directory separator); otherwise treat as a bare filename
        # inside the phase's checkpoints/ directory.
        if os.path.sep in ckpt_name or os.path.isabs(ckpt_name):
            ckpt = ckpt_name if os.path.isfile(ckpt_name) else None
        else:
            ckpt = find_checkpoint(phase_dir, ckpt_name)

        if ckpt:
            resolved[phase] = ckpt
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"  Phase {phase:2d}: {ckpt} ({size_mb:.1f} MB){tag}")
        else:
            missing.append(phase)
            print(f"  Phase {phase:2d}: MISSING (run={run_id}, "
                  f"wanted {ckpt_name}){tag}")

    print()

    # Check completeness
    all_phases = set(range(NUM_PHASES))
    covered = set(phase_map.keys())
    unspecified = all_phases - covered
    if unspecified:
        print(f"Warning: phases {sorted(unspecified)} not specified")
        missing.extend(sorted(unspecified))

    if missing and not cli.allow_partial:
        print(f"Error: {len(missing)} phase(s) missing: {sorted(set(missing))}")
        print("Use --allow-partial to proceed anyway")
        sys.exit(1)

    if not resolved:
        print("Error: no phases resolved")
        sys.exit(1)

    # Create output directory
    os.makedirs(cli.output_dir, exist_ok=True)

    # Copy checkpoints
    for phase, src_path in sorted(resolved.items()):
        dst_path = os.path.join(cli.output_dir, f"phase_{phase:02d}.pt")
        shutil.copy2(src_path, dst_path)

    # Write YAML spec
    phases_yaml = []
    for phase in sorted(resolved.keys()):
        entry = {
            "name": f"phase-{phase:02d}",
            "checkpoint": os.path.join(cli.output_dir, f"phase_{phase:02d}.pt"),
        }
        # Last phase has no trigger (terminal)
        if phase < max(resolved.keys()):
            entry["until"] = trigger
        phases_yaml.append(entry)

    spec = {
        "type": "composite",
        "phases": phases_yaml,
    }

    yaml_path = cli.output_dir.rstrip("/") + ".yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(resolved)} phase checkpoints to {cli.output_dir}/")
    print(f"Wrote composite spec to {yaml_path}")
    print()
    print("Rollout:")
    print(f"  poetry run python train/ppo/rollout.py \\")
    print(f"      --policy-spec {yaml_path} \\")
    print(f"      --env-version v4 --num-episodes 4")
    print()
    print("Sweep:")
    print(f"  poetry run python train/ppo/sweep_tiptilt.py \\")
    print(f"      --policy-spec {yaml_path} \\")
    print(f"      --env-version v4 --tt-min 0.0 --tt-max 2.0 --tt-steps 8 \\")
    print(f"      --num-episodes 4")


if __name__ == "__main__":
    main()
