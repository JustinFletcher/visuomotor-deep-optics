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
#   - bootstrap_reference_phased_count is left UNSET so the env
#     defaults to (phased_count + 1) — the same reference each phase
#     was trained against. This matters for two reasons:
#       1. obs is normalised by _reference_fpi_max, so a different
#          reference produces obs values orders of magnitude off-
#          distribution from what the CNN encoder was trained on.
#          (Old override ref=15 made phase-0 rollout obs ~225x dimmer
#          than training, so the policy never recognised convergence.)
#       2. The bootstrap_rescale_reward formula cancels out the
#          reference choice for the rescaled cs_val, so prior_reward
#          remains in the correct training distribution.
#     Trade-off: Strehl readout becomes per-phase relative (always
#     [0, 1] against that phase's reference), not a global Strehl.
#   - bootstrap_mask_nontarget=False because CompositeAgent applies
#     a per-phase DOF mask externally — a single env-side mask tied
#     to a fixed phased_count would clobber the dynamic per-phase
#     masking when phases switch.
ROLLOUT_ENV_KWARGS = {
    **ELF_BOOTSTRAP_ENV_KWARGS,
    "bootstrap_phased_count": 0,
    "bootstrap_mask_nontarget": False,
}

DEFAULT_SPEC = "train/ppo/specs/bootstrap_composed.yaml"


def _count_phases_and_steps(spec_path: str, override_steps: int | None):
    """Inspect a composite spec to decide max_episode_steps.

    Returns (num_phases, steps_per_phase, max_episode_steps).

    For HOMOGENEOUS specs (every phase shares the same step trigger)
    ``steps_per_phase`` is that shared value and max_episode_steps is
    ``steps_per_phase * num_phases``. For HETEROGENEOUS specs (e.g.
    segment_dm_agent with 16-step segment phases interleaved with
    32-step dm_atmos phases) we sum each phase's own ``until.step``
    so the episode actually has room to run every phase. The terminal
    phase usually has no ``until`` -- we give it the median per-phase
    budget so it runs for a reasonable tail.

    ``override_steps`` replaces every per-phase count uniformly when
    set (legacy --steps-per-phase behaviour). Use of --steps-per-phase
    on a heterogeneous spec will collapse every phase to the same
    budget, which probably isn't what the user wants -- log a warning
    in that case.
    """
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    if spec.get("type") == "single":
        # Single-model eval falls back to a standard episode length.
        return 1, override_steps or 256, override_steps or 256

    phases = spec["phases"]
    n = len(phases)

    # Collect each phase's explicit step budget; None marks phases
    # with non-step triggers (metric, fraction, missing). Final phase
    # frequently has no `until` block (terminal "runs to end").
    per_phase_steps: list[int | None] = []
    for ph in phases:
        until = ph.get("until") or {}
        if "step" in until:
            per_phase_steps.append(int(until["step"]))
        else:
            per_phase_steps.append(None)

    # Resolve a per-phase summary (used for printing and legacy
    # callers that want a single number). Prefer the median of
    # explicit step values; fall back to 256.
    explicit = [s for s in per_phase_steps if s is not None]
    median_steps = (sorted(explicit)[len(explicit) // 2]
                    if explicit else 256)

    if override_steps is not None:
        # Uniform override -- collapse every phase to this width.
        if len(set(explicit)) > 1:
            print(f"  [warn] --steps-per-phase {override_steps} "
                  f"collapses {sorted(set(explicit))} into a uniform "
                  f"budget; per-phase variation in the spec is lost.")
        per_phase_steps = [override_steps] * n
        median_steps = override_steps

    # max_episode_steps is the SUM of every phase's budget. Phases
    # with no explicit step trigger (typically the terminal "runs to
    # end" phase) get the MAX explicit step value -- preserves the
    # rhythm of the longest cadence so e.g. a terminal dm_atmos
    # phase in a 32-step-cadence spec gets a full 32 steps rather
    # than something shorter from a different cadence.
    max_explicit = max(explicit) if explicit else median_steps
    total = sum((s if s is not None else max_explicit)
                for s in per_phase_steps)
    return n, median_steps, total


def _maybe_rewrite_spec(spec_path: str,
                        steps_per_phase: int | None,
                        start_at_phase: int | None,
                        run_through_phase: int | None,
                        phase_checkpoints: dict[int, str] | None = None) -> str:
    """Rewrite the composite spec with optional overrides.

    - ``steps_per_phase`` (int): force every phase's ``step`` trigger
      to this value.
    - ``start_at_phase`` (int): drop phases 0..start_at_phase-1 so
      the composite begins at phase ``start_at_phase``.
    - ``run_through_phase`` (int): truncate the spec to phases
      0..run_through_phase inclusive (drop later phases entirely).
    - ``phase_checkpoints`` (dict[int, str]): per-phase checkpoint
      path overrides, keyed by ORIGINAL spec phase index (so phase 0
      in the original spec is key 0, even if --start-at-phase slices
      it away).

    The slicing rules are evaluated against the ORIGINAL spec phase
    indices (so --start-at-phase 3 --run-through-phase 7 keeps phases
    3, 4, 5, 6, 7 in their original named order).

    Returns the path to use (original if no rewrite needed).
    """
    if (steps_per_phase is None
            and start_at_phase is None
            and run_through_phase is None
            and not phase_checkpoints):
        return spec_path

    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)
    if spec.get("type") != "composite":
        return spec_path

    # Rewritten spec lives in test_output/_tmp_specs/, NOT next to
    # the source. Any spec-relative checkpoint path in the original
    # spec (e.g. "checkpoints/phase_00.pt" from an exported agent
    # bundle) would silently break against the temp dir, so resolve
    # every checkpoint path to the original spec's directory now.
    src_spec_dir = os.path.dirname(os.path.abspath(spec_path))
    for ph in spec["phases"]:
        ck = ph.get("checkpoint")
        if ck and not os.path.isabs(ck):
            ph["checkpoint"] = os.path.normpath(
                os.path.join(src_spec_dir, ck))

    N = len(spec["phases"])
    lo = 0 if start_at_phase is None else start_at_phase
    hi = (N - 1) if run_through_phase is None else run_through_phase
    if not 0 <= lo < N:
        raise ValueError(
            f"--start-at-phase {lo} out of range [0, {N-1}] for spec "
            f"with {N} phases")
    if not 0 <= hi < N:
        raise ValueError(
            f"--run-through-phase {hi} out of range [0, {N-1}] for spec "
            f"with {N} phases")
    if lo > hi:
        raise ValueError(
            f"--start-at-phase ({lo}) must be <= --run-through-phase ({hi})")

    suffix = ""

    # Apply per-phase checkpoint overrides FIRST (against original
    # indices), then slice.
    if phase_checkpoints:
        bad = [k for k in phase_checkpoints if not 0 <= k < N]
        if bad:
            raise ValueError(
                f"Phase override index {bad} out of range [0, {N-1}] "
                f"for spec with {N} phases")
        for idx, ckpt_path in phase_checkpoints.items():
            spec["phases"][idx]["checkpoint"] = ckpt_path
        suffix += "_ck" + "-".join(str(k) for k in sorted(phase_checkpoints))

    if start_at_phase is not None or run_through_phase is not None:
        spec["phases"] = spec["phases"][lo : hi + 1]
        suffix += f"_p{lo}-{hi}"

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
        "--start-at-phase", type=int, default=None,
        help="Begin the rollout at this phase index (0-based) instead "
             "of phase 0. The env starts in phase N's training-time "
             "start state: segs 0..N-1 already aligned, seg N perturbed, "
             "segs N+1..14 off-axis. Phases 0..N-1 are dropped from the "
             "composite. Combine with --run-through-phase to inspect a "
             "specific slice.")
    parser.add_argument(
        "--run-through-phase", type=int, default=None,
        help="Stop the rollout at the end of this phase index (0-based, "
             "inclusive). E.g. --run-through-phase 7 runs phases 0..7 "
             "and ends. Cuts both wall time and GIF render time.")
    parser.add_argument(
        "--env-version", type=str, default="v4",
        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument(
        "--env-recipe", type=str, default=None,
        help="Optional path to a fine-tune recipe YAML "
             "(train/ppo/finetune_recipes/*.yaml). When set, the "
             "recipe's env_kwarg_overrides block is applied on top "
             "of ROLLOUT_ENV_KWARGS so the eval env matches the "
             "regime the agent was fine-tuned for. Without this flag "
             "fine-tuned agents run against the ORIGINAL training "
             "env, which usually makes them look better than they are.")
    parser.add_argument(
        "--env-kwarg", action="append", default=[], metavar="KEY=VALUE",
        help="Override a single env_kwarg by hand. Repeatable. Useful "
             "for ad-hoc probes without writing a recipe. Values are "
             "parsed as YAML scalars (so '1e-4', '1000.0', 'true' all "
             "work). Applied AFTER --env-recipe so manual overrides win.")
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-gifs", action="store_true")
    parser.add_argument(
        "--lowres-gifs", action="store_true",
        help="Render GIFs at lower resolution and tighter layout. ~5-10x "
             "smaller files and ~5-8x faster rendering. Keeps every frame.")
    parser.add_argument(
        "--frame-stride", type=int, default=1, metavar="N",
        help="Keep every Nth frame in the GIF (default 1 = every "
             "frame). First and last frames are always retained so "
             "the start and end states are visible. Combine with "
             "--lowres-gifs for the smallest, fastest renders.")
    parser.add_argument(
        "--phase-checkpoint", action="append", default=[], metavar="N=PATH",
        help="Override the checkpoint for phase N (0-based, original "
             "spec index). May be passed multiple times to override "
             "several phases. Example: --phase-checkpoint "
             "0=runs/my_phase0/best.pt --phase-checkpoint "
             "3=bootstrap_runs/.../history_update_5000.pt")
    args = parser.parse_args()

    # Parse --phase-checkpoint entries into {int: str}.
    phase_checkpoints: dict[int, str] = {}
    for entry in args.phase_checkpoint:
        if "=" not in entry:
            parser.error(
                f"--phase-checkpoint expects N=PATH, got {entry!r}")
        k, v = entry.split("=", 1)
        try:
            idx = int(k)
        except ValueError:
            parser.error(f"Phase index must be int, got {k!r}")
        if not os.path.isabs(v):
            v = os.path.join(_REPO_ROOT, v)
        if not os.path.isfile(v):
            parser.error(f"Phase {idx} checkpoint not found: {v}")
        phase_checkpoints[idx] = v

    if not os.path.isabs(args.policy_spec):
        args.policy_spec = os.path.join(_REPO_ROOT, args.policy_spec)

    if not os.path.isfile(args.policy_spec):
        parser.error(f"Policy spec not found: {args.policy_spec}")

    # Figure out how long an episode needs to be (size against the
    # ORIGINAL spec; slicing flags are applied below).
    num_phases, per_phase, spec_total_steps = _count_phases_and_steps(
        args.policy_spec, args.steps_per_phase)
    lo = 0 if args.start_at_phase is None else args.start_at_phase
    hi = (num_phases - 1 if args.run_through_phase is None
          else args.run_through_phase)
    effective_phases = hi - lo + 1
    if effective_phases <= 0:
        parser.error(
            f"--start-at-phase ({lo}) must be <= --run-through-phase ({hi})")
    # Heterogeneous specs (e.g. segment_dm_agent: 16-step segment
    # phases interleaved with 32-step dm_atmos phases) need
    # max_episode_steps = sum of every active phase's budget, not
    # per_phase * effective_phases. _count_phases_and_steps returned
    # the SUM across all phases; when --start-at-phase or
    # --run-through-phase trims the spec, recompute on the trimmed
    # range. For full-spec runs the trimming branch is a no-op and
    # max_steps equals spec_total_steps verbatim.
    if (args.start_at_phase is None
            and args.run_through_phase is None):
        max_steps = spec_total_steps
    else:
        # Recompute the sum on the active slice [lo, hi+1).
        import yaml as _yaml
        with open(args.policy_spec) as _f:
            _all_phases = _yaml.safe_load(_f).get("phases", [])
        max_steps = 0
        for _ph in _all_phases[lo:hi + 1]:
            _u = (_ph.get("until") or {})
            max_steps += int(_u.get("step",
                args.steps_per_phase
                if args.steps_per_phase is not None else per_phase))

    # Env starts in phase ``lo``'s training-time condition. With
    # bootstrap_phased_count = lo: segs 0..lo-1 already aligned, seg lo
    # perturbed, segs lo+1..14 off-axis. Reference image stays at all-
    # 15-aligned so Strehl still reads against the ultimate goal.
    env_kwargs = dict(ROLLOUT_ENV_KWARGS)
    env_kwargs["bootstrap_phased_count"] = lo

    # --- Composite-spec env_kwarg overrides ---
    # A composite agent bundle can declare env_kwarg_overrides at the
    # top of its spec to describe the env it needs (e.g. command_dm:
    # true to enable DM commanding so a Zernike-DM phase has somewhere
    # for its output to go). Applied BEFORE --env-recipe and
    # --env-kwarg so user overrides still win at the CLI.
    from train.ppo.policy_spec import read_env_kwarg_overrides
    spec_env_overrides = read_env_kwarg_overrides(args.policy_spec)
    if spec_env_overrides:
        env_kwargs.update(spec_env_overrides)
        print(f"  spec env_kwarg overrides ({args.policy_spec}): "
              f"applied {len(spec_env_overrides)}")
        for k, v in spec_env_overrides.items():
            print(f"    {k} = {v}")

    # --- Recipe env_kwarg overrides (e.g., fine-tune target regime) ---
    if args.env_recipe:
        with open(args.env_recipe) as f:
            recipe = yaml.safe_load(f) or {}
        rec_overrides = recipe.get("env_kwarg_overrides", {}) or {}
        if rec_overrides:
            env_kwargs.update(rec_overrides)
            print(f"  env_recipe {args.env_recipe}: applied "
                  f"{len(rec_overrides)} env_kwarg overrides")
            for k, v in rec_overrides.items():
                print(f"    {k} = {v}")

    # --- Per-key manual overrides (--env-kwarg KEY=VALUE) ---
    for entry in args.env_kwarg:
        if "=" not in entry:
            parser.error(f"--env-kwarg expects KEY=VALUE, got {entry!r}")
        k, raw = entry.split("=", 1)
        v = yaml.safe_load(raw)
        env_kwargs[k] = v
        print(f"  env_kwarg override: {k} = {v!r}")

    print(f"Spec:              {args.policy_spec}")
    print(f"Phases (spec):     {num_phases}")
    print(f"Phases (this run): {effective_phases}  [{lo}..{hi}]")
    print(f"Steps per phase:   {per_phase}")
    print(f"Max episode steps: {max_steps}")
    print(f"Env start state:   bootstrap_phased_count = {lo}  "
          f"(segs 0..{lo-1} aligned, seg {lo} target, "
          f"segs {lo+1}..14 off-axis)" if lo > 0 else
          f"Env start state:   bootstrap_phased_count = 0  "
          f"(seg 0 target, segs 1..14 off-axis)")
    print(f"Env version:       {args.env_version}")
    print(f"Aperture:          {env_kwargs['aperture_type']}")
    print(f"Focal plane:       {env_kwargs['focal_plane_image_size_pixels']}px")
    print()

    # Rewrite the spec when slicing or step-trigger overrides apply.
    # CompositeAgent's phase list and triggers must match the
    # evaluation horizon.
    effective_spec = _maybe_rewrite_spec(
        args.policy_spec,
        steps_per_phase=args.steps_per_phase,
        start_at_phase=args.start_at_phase,
        run_through_phase=args.run_through_phase,
        phase_checkpoints=phase_checkpoints or None,
    )

    # Timestamped output directory.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"bootstrap_rollout_{timestamp}_{int(time.time()) % 10000}"
    test_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output directory: {test_dir}\n")

    episodes, metrics = run_rollouts(
        policy_spec_path=effective_spec,
        env_kwargs=env_kwargs,
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
                          lowres=args.lowres_gifs,
                          frame_stride=args.frame_stride)

    save_summary_figures(episodes, metrics, os.path.join(test_dir, "figures"))

    print(f"\nAll outputs in: {test_dir}")


if __name__ == "__main__":
    main()
