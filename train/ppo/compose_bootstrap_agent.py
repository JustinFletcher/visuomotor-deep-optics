#!/usr/bin/env python
"""Compose a CompositeAgent from 15 trained bootstrap phase models.

Copies checkpoints from bootstrap_runs/ into a single directory and
writes a YAML policy spec for use with rollout.py and sweep_tiptilt.py.

Supports cherry-picking phases from different bootstrap runs.

Usage:
    # All 15 from one run
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
RUNS_ROOT = "runs"
DEFAULT_OUTPUT_DIR = "train/ppo/specs/bootstrap_composed"


def find_training_subdir(phase_dir):
    """Find the ppo_optomech_* subdir inside a bootstrap phase dir."""
    candidates = glob.glob(os.path.join(phase_dir, "ppo_optomech_*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _search_checkpoint(ckpt_dir, checkpoint_name):
    """Return path to requested checkpoint in ``ckpt_dir``, or the latest
    ``update_N.pt`` as a fallback when asked for ``best.pt``."""
    if not os.path.isdir(ckpt_dir):
        return None
    ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
    if os.path.isfile(ckpt_path):
        return ckpt_path
    if checkpoint_name == "best.pt":
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


def find_checkpoint(run_id, phase, checkpoint_name="best.pt"):
    """Locate the requested checkpoint for (run_id, phase).

    Tries two layouts in order:
      1. Nested bootstrap layout:
         ``bootstrap_runs/{run_id}/bootstrap_phase_NN/ppo_optomech_*/checkpoints/``
      2. Flattened layout produced by ``sync_bootstrap_runs.py``:
         ``runs/{run_id}_bootstrap_phase_NN/checkpoints/``
    """
    phase_name = f"bootstrap_phase_{phase:02d}"

    # Layout 1: original nested bootstrap_runs structure
    phase_dir = os.path.join(BOOTSTRAP_ROOT, run_id, phase_name)
    train_dir = find_training_subdir(phase_dir)
    if train_dir is not None:
        ckpt = _search_checkpoint(
            os.path.join(train_dir, "checkpoints"), checkpoint_name)
        if ckpt:
            return ckpt

    # Layout 2: flattened runs/ structure (post sync_bootstrap_runs.py)
    flat_dir = os.path.join(RUNS_ROOT, f"{run_id}_{phase_name}", "checkpoints")
    ckpt = _search_checkpoint(flat_dir, checkpoint_name)
    if ckpt:
        return ckpt

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
        help="Checkpoint filename to use per phase (default: best.pt)")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Proceed even if some phases are missing")
    cli = parser.parse_args()

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
        parser.error("Specify --run-id or --phase")

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
        ckpt = find_checkpoint(run_id, phase, cli.checkpoint_name)

        if ckpt:
            resolved[phase] = ckpt
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"  Phase {phase:2d}: {ckpt} ({size_mb:.1f} MB)")
        else:
            missing.append(phase)
            print(f"  Phase {phase:2d}: MISSING (run={run_id})")

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
