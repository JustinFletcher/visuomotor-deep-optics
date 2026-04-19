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
import datetime as _dt
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys

import yaml

NUM_PHASES = 15
BOOTSTRAP_ROOT = "bootstrap_runs"
DEFAULT_OUTPUT_DIR = "train/ppo/specs/bootstrap_composed"
DEFAULT_EXPORT_ROOT = "agents"


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
    parser.add_argument(
        "--export", action="store_true",
        help="Build a self-contained, relocatable agent bundle under "
             f"{DEFAULT_EXPORT_ROOT}/ instead of editing in place. The "
             "bundle directory contains checkpoints, a YAML spec with "
             "RELATIVE checkpoint paths so the directory can be moved "
             "anywhere, a manifest.json with full provenance "
             "(per-phase source run / checkpoint, git commit, command "
             "line, env_kwargs from the first phase's training "
             "config), and a README.md with rollout instructions.")
    parser.add_argument(
        "--export-name", type=str, default=None,
        help="Directory name under --export-root for the exported "
             "bundle. Default: agent_<UTC>_<short-id>. Refuses to "
             "overwrite an existing directory.")
    parser.add_argument(
        "--export-root", type=str, default=DEFAULT_EXPORT_ROOT,
        help=f"Parent directory for exports (default: {DEFAULT_EXPORT_ROOT})")
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

    if cli.export:
        export_agent_bundle(resolved, phase_map, trigger, cli)
    else:
        write_in_place(resolved, trigger, cli)


def write_in_place(resolved, trigger, cli):
    """Original behaviour: copy checkpoints to --output-dir and write
    a sibling .yaml spec next to it (paths are absolute / repo-relative,
    not portable)."""
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
    print(f"  poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \\")
    print(f"      --policy-spec {yaml_path} \\")
    print(f"      --num-episodes 4 --steps-per-phase 64 --lowres-gifs")


def _git_commit_short():
    """Return the current repo's short SHA, or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _read_env_kwargs(checkpoint_path):
    """Pull env_kwargs out of a checkpoint's stored config. Loaded
    lazily so the composer doesn't pay the torch import cost when
    --export isn't used."""
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        return cfg.get("env_kwargs", {}) or {}
    except Exception as e:
        print(f"  Warning: could not read env_kwargs from "
              f"{checkpoint_path}: {e}")
        return {}


def export_agent_bundle(resolved, phase_map, trigger, cli):
    """Build a self-contained, relocatable agent directory.

    Layout:
        <export_root>/<name>/
          composed.yaml            # spec with RELATIVE checkpoint paths
          checkpoints/
            phase_00.pt
            ...
          manifest.json            # provenance, env_kwargs, git, command
          README.md                # rollout instructions
    """
    # Resolve target directory.
    if cli.export_name:
        bundle_name = cli.export_name
    else:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bundle_name = f"agent_{ts}_{secrets.token_hex(2)}"
    bundle_dir = os.path.join(cli.export_root, bundle_name)

    if os.path.exists(bundle_dir):
        print(f"Error: export directory already exists: {bundle_dir}")
        print("Pass a different --export-name or remove the existing dir.")
        sys.exit(1)
    ckpt_dir = os.path.join(bundle_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=False)

    print(f"Exporting agent bundle to: {bundle_dir}")

    # Copy checkpoints into the bundle.
    bundle_phases = []
    total_bytes = 0
    for phase, src_path in sorted(resolved.items()):
        rel_ckpt = f"checkpoints/phase_{phase:02d}.pt"
        dst_path = os.path.join(bundle_dir, rel_ckpt)
        shutil.copy2(src_path, dst_path)
        size = os.path.getsize(dst_path)
        total_bytes += size
        bundle_phases.append({
            "phase": phase,
            "name": f"phase-{phase:02d}",
            "source_run": phase_map.get(phase),
            "source_path": src_path,
            "bundle_path": rel_ckpt,
            "size_bytes": size,
        })

    # YAML spec with paths RELATIVE to the spec file. The loader
    # tries spec-relative resolution first, so the bundle stays
    # portable across moves.
    phases_yaml = []
    last_phase = max(resolved.keys())
    for phase in sorted(resolved.keys()):
        entry = {
            "name": f"phase-{phase:02d}",
            "checkpoint": f"checkpoints/phase_{phase:02d}.pt",
        }
        if phase < last_phase:
            entry["until"] = trigger
        phases_yaml.append(entry)
    spec = {"type": "composite", "phases": phases_yaml}
    yaml_path = os.path.join(bundle_dir, "composed.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    # Pull env_kwargs from phase 0's checkpoint (or the lowest phase
    # we have) so the rollout consumer can verify the env config.
    first_phase = min(resolved.keys())
    env_kwargs = _read_env_kwargs(resolved[first_phase])

    manifest = {
        "agent_name": bundle_name,
        "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_commit_short(),
        "command": " ".join(sys.argv),
        "trigger": trigger,
        "num_phases": len(resolved),
        "phases": bundle_phases,
        "total_checkpoint_bytes": total_bytes,
        "training_env_kwargs": env_kwargs,
        "spec_path": "composed.yaml",
        "rollout_entry": "train/ppo/rollout_elf_bootstrap_ptt.py",
    }
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # README the user (or future you) can read after `cd` into the
    # bundle. Phase-source table is inlined so the bundle is
    # self-explanatory without opening manifest.json.
    readme_lines = [
        f"# {bundle_name}",
        "",
        f"Self-contained ELF bootstrap composite agent, exported on "
        f"{manifest['created_utc']}.",
        "",
        f"- Phases: {len(resolved)}",
        f"- Total checkpoint size: {total_bytes / (1024 * 1024):.1f} MB",
        f"- Source git commit: `{manifest['git_commit']}`",
        f"- Phase transition trigger: `{trigger}`",
        "",
        "## Provenance",
        "",
        "| Phase | Source run | Source checkpoint |",
        "|-------|------------|-------------------|",
    ]
    for ph in bundle_phases:
        src = ph["source_path"]
        # Show the basename if the source path is long.
        readme_lines.append(
            f"| {ph['phase']:>5} | `{ph['source_run']}` | `{src}` |"
        )
    readme_lines += [
        "",
        "## Rollout",
        "",
        "From the visuomotor-deep-optics repo root:",
        "",
        "```bash",
        f"poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \\",
        f"    --policy-spec {yaml_path} \\",
        "    --num-episodes 4 --steps-per-phase 64 --lowres-gifs",
        "```",
        "",
        "The YAML spec uses paths RELATIVE to itself, so this bundle "
        "directory can be moved anywhere on disk and the rollout "
        "command above will continue to work as long as you point "
        "`--policy-spec` at the new location of `composed.yaml`.",
        "",
        "## Files",
        "",
        "- `composed.yaml` — composite policy spec",
        "- `checkpoints/phase_NN.pt` — per-phase PPO checkpoints",
        "- `manifest.json` — full provenance (env_kwargs, git, command, "
        "per-phase source paths)",
        "",
    ]
    readme_path = os.path.join(bundle_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines))

    print()
    print(f"Bundle ready: {bundle_dir}")
    print(f"  spec:     {yaml_path}")
    print(f"  manifest: {manifest_path}")
    print(f"  readme:   {readme_path}")
    print(f"  size:     {total_bytes / (1024 * 1024):.1f} MB across "
          f"{len(resolved)} checkpoints")
    print()
    print("Rollout:")
    print(f"  poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \\")
    print(f"      --policy-spec {yaml_path} \\")
    print(f"      --num-episodes 4 --steps-per-phase 64 --lowres-gifs")


if __name__ == "__main__":
    main()
