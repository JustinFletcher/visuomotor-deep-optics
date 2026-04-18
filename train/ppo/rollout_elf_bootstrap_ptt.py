#!/usr/bin/env python
"""
End-to-end rollout for the ELF PTT bootstrap composite agent.

Owns the ELF bootstrap env configuration so that `rollout.run_rollouts`
is built with the same aperture / focal plane / action bounds used during
training. Mirrors the pattern of `optimize/multi_stage/ms_elf_smaes_bootstrap.py`
where a thin experiment script provides static config and delegates to a
generic runner.

The composite spec produced by `compose_bootstrap_agent.py` defines N
phases with per-phase step triggers; this script reads the phase count
and trigger steps from the spec to size `max_episode_steps` so the full
15-phase stack actually runs.

Usage:
    # Default: auto-detect composed spec, 4 episodes, 256 steps/phase
    poetry run python train/ppo/rollout_elf_bootstrap_ptt.py

    # Explicit spec + custom episode count
    poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \\
        --policy-spec train/ppo/specs/bootstrap_composed.yaml \\
        --num-episodes 8

    # Override steps-per-phase (also rewrites the spec in memory)
    poetry run python train/ppo/rollout_elf_bootstrap_ptt.py --steps-per-phase 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.rollout import (
    run_rollouts,
    save_episode_gifs,
    save_summary_figures,
    _DEFAULT_OUTPUT_DIR,
)
from train.ppo.train_ppo_elf_bootstrap import ELF_BOOTSTRAP_ENV_KWARGS


# End-to-end composite rollout env:
#   - bootstrap_phased_count=0 keeps the reset state as "everything
#     off-axis" so the composite starts from the same condition phase 0
#     was trained on.
#   - bootstrap_reference_phased_count=15 forces the reference image /
#     reference flux to be computed with all 15 segments aligned, so
#     Strehl actually reads 0..1 against the ultimate goal state rather
#     than against phase 0's intermediate "1 aligned + 14 off-axis"
#     reference (which would let Strehl exceed 1 as more segments
#     co-phase during the rollout).
ROLLOUT_ENV_KWARGS = {
    **ELF_BOOTSTRAP_ENV_KWARGS,
    "bootstrap_phased_count": 0,
    "bootstrap_reference_phased_count": 15,
    # During composite rollout the CompositeAgent applies a per-phase
    # DOF mask built from each checkpoint's training config, so the env
    # itself must NOT mask. A single env-side mask (tied to a single
    # phased_count) would clobber the dynamic per-phase masks.
    "bootstrap_mask_nontarget": False,
}

DEFAULT_SPEC = "train/ppo/specs/bootstrap_composed.yaml"


def _count_phases_and_steps(spec_path: str, override_steps: int | None):
    """Inspect a composite spec to decide max_episode_steps.

    Returns (num_phases, steps_per_phase, max_episode_steps). If
    override_steps is given it replaces the per-phase step count
    everywhere; otherwise the per-phase count is read from the spec.
    """
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    if spec.get("type") == "single":
        # Single-model eval falls back to a standard episode length.
        return 1, override_steps or 256, override_steps or 256

    phases = spec["phases"]
    spec_steps = None
    for ph in phases:
        until = ph.get("until") or {}
        if "step" in until:
            spec_steps = int(until["step"])
            break
    per_phase = override_steps if override_steps is not None else (spec_steps or 256)
    return len(phases), per_phase, per_phase * len(phases)


def _maybe_rewrite_spec(spec_path: str,
                        steps_per_phase: int | None,
                        run_through_phase: int | None) -> str:
    """Rewrite the composite spec with optional overrides.

    - ``steps_per_phase`` (int): force every phase's ``step`` trigger
      to this value.
    - ``run_through_phase`` (int): truncate the spec to phases
      0..run_through_phase inclusive (drop later phases entirely).

    Returns the path to use (original if no rewrite needed).
    """
    if steps_per_phase is None and run_through_phase is None:
        return spec_path

    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)
    if spec.get("type") != "composite":
        return spec_path

    suffix = ""
    if run_through_phase is not None:
        N = len(spec["phases"])
        if not 0 <= run_through_phase < N:
            raise ValueError(
                f"--run-through-phase {run_through_phase} out of range "
                f"[0, {N-1}] for spec with {N} phases")
        spec["phases"] = spec["phases"][: run_through_phase + 1]
        suffix += f"_p{run_through_phase}"

    if steps_per_phase is not None:
        for ph in spec["phases"]:
            until = ph.get("until")
            if until and "step" in until:
                until["step"] = steps_per_phase
        suffix += f"_s{steps_per_phase}"

    tmp_dir = os.path.join(_REPO_ROOT, "test_output", "_tmp_specs")
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.basename(spec_path).replace(".yaml", "")
    tmp_path = os.path.join(tmp_dir, f"{base}{suffix}.yaml")
    with open(tmp_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    print(f"  Rewrote spec ({suffix.lstrip('_')}): {tmp_path}")
    return tmp_path


def main():
    parser = argparse.ArgumentParser(
        description="Rollout the ELF PTT bootstrap composite agent.")
    parser.add_argument(
        "--policy-spec", type=str, default=DEFAULT_SPEC,
        help=f"Composite (or single) policy spec YAML (default: {DEFAULT_SPEC})")
    parser.add_argument(
        "--num-episodes", type=int, default=4)
    parser.add_argument(
        "--steps-per-phase", type=int, default=None,
        help="Override per-phase step trigger in the spec. When omitted, "
             "use whatever the spec says.")
    parser.add_argument(
        "--run-through-phase", type=int, default=None,
        help="Stop the rollout at the end of this phase index (0-based, "
             "inclusive). E.g. --run-through-phase 7 runs phases 0..7 "
             "and ends. Cuts both wall time and GIF render time.")
    parser.add_argument(
        "--env-version", type=str, default="v4",
        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-gifs", action="store_true")
    parser.add_argument(
        "--lowres-gifs", action="store_true",
        help="Render GIFs at lower resolution and tighter layout. ~5-10x "
             "smaller files and ~5-8x faster rendering. Keeps every frame.")
    args = parser.parse_args()

    if not os.path.isabs(args.policy_spec):
        args.policy_spec = os.path.join(_REPO_ROOT, args.policy_spec)

    if not os.path.isfile(args.policy_spec):
        parser.error(f"Policy spec not found: {args.policy_spec}")

    # Figure out how long an episode needs to be (size against the
    # ORIGINAL spec; we'll apply --run-through-phase truncation below).
    num_phases, per_phase, _ = _count_phases_and_steps(
        args.policy_spec, args.steps_per_phase)
    effective_phases = (args.run_through_phase + 1
                        if args.run_through_phase is not None
                        else num_phases)
    max_steps = per_phase * effective_phases

    print(f"Spec:              {args.policy_spec}")
    print(f"Phases (spec):     {num_phases}")
    print(f"Phases (this run): {effective_phases}")
    print(f"Steps per phase:   {per_phase}")
    print(f"Max episode steps: {max_steps}")
    print(f"Env version:       {args.env_version}")
    print(f"Aperture:          {ROLLOUT_ENV_KWARGS['aperture_type']}")
    print(f"Focal plane:       {ROLLOUT_ENV_KWARGS['focal_plane_image_size_pixels']}px")
    print()

    # Rewrite the spec when either steps-per-phase or run-through-phase
    # is overridden. CompositeAgent's triggers and phase list must
    # match the evaluation horizon.
    effective_spec = _maybe_rewrite_spec(
        args.policy_spec,
        steps_per_phase=args.steps_per_phase,
        run_through_phase=args.run_through_phase,
    )

    # Timestamped output directory.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"bootstrap_rollout_{timestamp}_{int(time.time()) % 10000}"
    test_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output directory: {test_dir}\n")

    episodes, metrics = run_rollouts(
        policy_spec_path=effective_spec,
        env_kwargs=ROLLOUT_ENV_KWARGS,
        env_version=args.env_version,
        num_episodes=args.num_episodes,
        max_episode_steps=max_steps,
    )

    print(f"\n{'=' * 60}")
    print(f"  Episodes: {metrics['num_episodes']}")
    print(f"  Mean return:       {metrics['mean_return']:.4f} "
          f"+/- {metrics['std_return']:.4f}")
    print(f"  Median return:     {metrics['median_return']:.4f}")
    print(f"  Best / Worst:      {metrics['best_return']:.4f} / "
          f"{metrics['worst_return']:.4f}")
    print(f"  Zero baseline:     {metrics['mean_zero_return']:.4f}")
    print(f"  Mean gap:          {metrics['mean_improvement_gap']:+.4f}")
    print(f"  Mean final Strehl: {metrics['mean_final_strehl']:.4f} "
          f"+/- {metrics['std_final_strehl']:.4f}")
    print(f"{'=' * 60}")

    metrics_path = os.path.join(test_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")

    if not args.no_gifs:
        save_episode_gifs(episodes, os.path.join(test_dir, "gifs"),
                          lowres=args.lowres_gifs)

    save_summary_figures(episodes, metrics, os.path.join(test_dir, "figures"))

    print(f"\nAll outputs in: {test_dir}")


if __name__ == "__main__":
    main()
