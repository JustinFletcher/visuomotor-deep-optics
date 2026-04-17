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


# The bootstrap env perturbs segments 0..14 off-axis at reset when
# bootstrap_phased_count=0, which is the starting condition for an
# end-to-end composite rollout. Explicitly pin the env to phase 0 here
# so the eval doesn't inherit any training-time override.
ROLLOUT_ENV_KWARGS = {
    **ELF_BOOTSTRAP_ENV_KWARGS,
    "bootstrap_phased_count": 0,
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


def _maybe_rewrite_spec_steps(spec_path: str, steps_per_phase: int) -> str:
    """If the user overrode steps-per-phase, write a temp spec with new triggers."""
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)
    if spec.get("type") != "composite":
        return spec_path

    changed = False
    for ph in spec["phases"]:
        until = ph.get("until")
        if until and "step" in until and int(until["step"]) != steps_per_phase:
            until["step"] = steps_per_phase
            changed = True
    if not changed:
        return spec_path

    tmp_dir = os.path.join(_REPO_ROOT, "test_output", "_tmp_specs")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(
        tmp_dir, f"{os.path.basename(spec_path).replace('.yaml', '')}"
                 f"_s{steps_per_phase}.yaml")
    with open(tmp_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    print(f"  Rewrote spec with steps-per-phase={steps_per_phase}: {tmp_path}")
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
        "--env-version", type=str, default="v4",
        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-gifs", action="store_true")
    args = parser.parse_args()

    if not os.path.isabs(args.policy_spec):
        args.policy_spec = os.path.join(_REPO_ROOT, args.policy_spec)

    if not os.path.isfile(args.policy_spec):
        parser.error(f"Policy spec not found: {args.policy_spec}")

    # Figure out how long an episode needs to be.
    num_phases, per_phase, max_steps = _count_phases_and_steps(
        args.policy_spec, args.steps_per_phase)

    print(f"Spec:              {args.policy_spec}")
    print(f"Phases:            {num_phases}")
    print(f"Steps per phase:   {per_phase}")
    print(f"Max episode steps: {max_steps}")
    print(f"Env version:       {args.env_version}")
    print(f"Aperture:          {ROLLOUT_ENV_KWARGS['aperture_type']}")
    print(f"Focal plane:       {ROLLOUT_ENV_KWARGS['focal_plane_image_size_pixels']}px")
    print()

    # If the user overrode steps-per-phase, rewrite a temp spec so the
    # CompositeAgent's triggers match the evaluation horizon.
    effective_spec = args.policy_spec
    if args.steps_per_phase is not None:
        effective_spec = _maybe_rewrite_spec_steps(
            args.policy_spec, args.steps_per_phase)

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
        save_episode_gifs(episodes, os.path.join(test_dir, "gifs"))

    save_summary_figures(episodes, metrics, os.path.join(test_dir, "figures"))

    print(f"\nAll outputs in: {test_dir}")


if __name__ == "__main__":
    main()
