#!/usr/bin/env python
"""Action-scale sweep on top of the 15-phase bootstrap composite agent.

Default mode (``--scale-mode piston-only``): scales ONLY the segment
piston correction range across decades. Tip/tilt correction bounds
and the global ``env_action_scale`` stay at their training-time
defaults, so the TT effective per-step delta is identical at every
sweep point and the agent's TT commands behave the same as during
training. The default sweep grid ``[1, 10, 100, 1000]`` produces
``max_piston_correction_micron in [10, 100, 1000, 10000] µm``.

Alternate mode (``--scale-mode joint``): also multiplies
``max_tip_correction_arcsec``, ``max_tilt_correction_arcsec``, and
``env_action_scale`` by the same factor. The TT effective per-step
delta scales quadratically with ``s`` in this mode (max_tip stays
proportional and env_action_scale also grows). Useful when you want
"the whole action space" to scale uniformly.

Reuses the same ``ROLLOUT_ENV_KWARGS`` and spec-rewriting helpers as
``rollout_elf_bootstrap_ptt.py``; the only thing that changes between
sweep points is the four scale-related env_kwargs:

  - max_piston_correction_micron                  (always)
  - max_tip_correction_arcsec                     (joint mode only)
  - max_tilt_correction_arcsec                    (joint mode only)
  - env_action_scale                              (joint mode only)

Outputs per sweep point:
  test_output/<run_id>/scale_<sNN>/
    metrics.json
    gifs/                       (unless --no-gifs)
    figures/                    (per-scale summary)

Plus a combined comparison figure across scales:
  test_output/<run_id>/comparison.png

Usage:
    poetry run python train/ppo/sweep_action_scale_bootstrap_ptt.py \\
        --policy-spec agents/agent_20260419T211137Z_e3b7/composed.yaml

    # Two seeds per scale (default), custom grid
    poetry run python train/ppo/sweep_action_scale_bootstrap_ptt.py \\
        --policy-spec agents/.../composed.yaml \\
        --scale-grid 1 10 100

    # Five seeds per scale, lowres gifs
    poetry run python train/ppo/sweep_action_scale_bootstrap_ptt.py \\
        --policy-spec agents/.../composed.yaml \\
        --seeds-per-scale 5 --lowres-gifs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from train.ppo.rollout import (
    run_rollouts,
    save_episode_gifs,
    save_summary_figures,
    _DEFAULT_OUTPUT_DIR,
)
from train.ppo.rollout_elf_bootstrap_ptt import (
    ROLLOUT_ENV_KWARGS,
    DEFAULT_SPEC,
    _count_phases_and_steps,
    _maybe_rewrite_spec,
)


# Baselines for the four scale-coupled knobs at multiplier s = 1.
# Chosen so that s = 1 gives max_piston = 10 µm (one order of
# magnitude above the bootstrap-training default 1 µm), matching the
# sweep label "10, 100, 1000, 10000 µm" at the default grid.
_BASELINE_MAX_PISTON_UM = 10.0
_BASELINE_MAX_TIP_ARCSEC = 3.0
_BASELINE_MAX_TILT_ARCSEC = 3.0
_BASELINE_ENV_ACTION_SCALE = 1.0


_SCALE_MODES = ("piston-only", "joint")


def _scaled_env_kwargs(base_kwargs: dict, scale: float,
                       mode: str = "piston-only") -> dict:
    """Apply the scale multiplier to the action-range knobs in a copy
    of the base env_kwargs.

    Modes:
      "piston-only" (default): only ``max_piston_correction_micron`` is
        multiplied by ``scale``. Tip/tilt correction ranges and
        ``env_action_scale`` are left untouched so the agent's TT
        commands keep their training-time effective range.

      "joint": ``max_piston_correction_micron``, ``max_tip_correction
        _arcsec``, ``max_tilt_correction_arcsec``, AND ``env_action_scale``
        are all multiplied by ``scale``. The TT effective per-step delta
        scales quadratically with ``scale`` in this mode because the
        global ``env_action_scale`` also goes up.
    """
    if mode not in _SCALE_MODES:
        raise ValueError(f"mode must be one of {_SCALE_MODES}, got {mode!r}")
    kw = dict(base_kwargs)
    kw["max_piston_correction_micron"] = _BASELINE_MAX_PISTON_UM * scale
    if mode == "joint":
        kw["max_tip_correction_arcsec"] = _BASELINE_MAX_TIP_ARCSEC * scale
        kw["max_tilt_correction_arcsec"] = _BASELINE_MAX_TILT_ARCSEC * scale
        kw["env_action_scale"] = _BASELINE_ENV_ACTION_SCALE * scale
    # piston-only: leave TT bounds and env_action_scale alone -- they
    # take whatever the upstream ROLLOUT_ENV_KWARGS / training defaults
    # provided. TT errors (init_tip_arcsec_std etc.) are likewise not
    # touched, so the disturbance distribution the agent faces on TT is
    # identical across all sweep points.
    return kw


def _summarize(ep) -> dict:
    """Pull the small set of per-episode scalars we'll plot across scales."""
    strehls = np.asarray(ep["strehls"], dtype=np.float64)
    rewards = np.asarray(ep["rewards"], dtype=np.float64)
    actions = np.asarray(ep["actions"])                            # [T, D]
    return dict(
        strehls=strehls.tolist(),
        rewards=rewards.tolist(),
        final_strehl=float(strehls[-1]),
        mean_strehl=float(strehls.mean()),
        episodic_return=float(rewards.sum()),
        mean_abs_action=float(np.abs(actions).mean()),
        peak_abs_action=float(np.abs(actions).max()),
        action_saturated_frac=float((np.abs(actions) > 0.999).mean()),
    )


def _save_comparison_figure(per_scale: list[dict],
                            out_path: str,
                            mode: str = "piston-only",
                            ylim_strehl=(0.0, 1.0)) -> None:
    """Combined cross-scale figure: strehl trajectory mean+ribbon,
    final-strehl distribution, action saturation fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(per_scale)))

    # Panel 1: per-step Strehl trajectory, ribbon = min/max across seeds.
    ax = axes[0]
    for c, sc in zip(colors, per_scale):
        episode_strehls = np.asarray(
            [ep["strehls"] for ep in sc["summaries"]], dtype=np.float64)
        T = episode_strehls.shape[1] if episode_strehls.size else 0
        if T == 0:
            continue
        steps = np.arange(T)
        mean = episode_strehls.mean(axis=0)
        lo = episode_strehls.min(axis=0)
        hi = episode_strehls.max(axis=0)
        ax.plot(steps, mean, color=c, linewidth=1.6,
                label=f"piston={sc['max_piston_um']:.0f} µm")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel("step")
    ax.set_ylabel("Strehl")
    ax.set_ylim(*ylim_strehl)
    ax.set_title("Strehl trajectory per scale (mean + min/max ribbon)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: final-strehl per seed bar plot, grouped by scale.
    ax = axes[1]
    xs, heights, bar_colors, labels = [], [], [], []
    cursor = 0
    tick_positions, tick_labels = [], []
    for c, sc in zip(colors, per_scale):
        group_finals = [s["final_strehl"] for s in sc["summaries"]]
        for v in group_finals:
            xs.append(cursor); heights.append(v); bar_colors.append(c)
            cursor += 1
        tick_positions.append(cursor - len(group_finals) / 2 - 0.5)
        tick_labels.append(f"{sc['max_piston_um']:.0f}\nµm")
        cursor += 1  # gap between groups
    ax.bar(xs, heights, color=bar_colors, edgecolor="black",
           linewidth=0.4)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylim(*ylim_strehl)
    ax.set_ylabel("final Strehl (per seed)")
    ax.set_title("Final-step Strehl per seed, by scale")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel 3: action diagnostics per scale -- mean |a| and saturation.
    ax = axes[2]
    sc_idx = np.arange(len(per_scale))
    mean_abs = np.array([
        np.mean([s["mean_abs_action"] for s in sc["summaries"]])
        for sc in per_scale])
    peak_abs = np.array([
        np.mean([s["peak_abs_action"] for s in sc["summaries"]])
        for sc in per_scale])
    sat_frac = np.array([
        np.mean([s["action_saturated_frac"] for s in sc["summaries"]])
        for sc in per_scale])
    width = 0.32
    ax.bar(sc_idx - width, mean_abs, width=width, label="mean |a|",
           color="C0")
    ax.bar(sc_idx, peak_abs, width=width, label="peak |a|",
           color="C1")
    ax.bar(sc_idx + width, sat_frac, width=width,
           label="frac at ±1 (sat)", color="C3")
    ax.set_xticks(sc_idx)
    ax.set_xticklabels([f"{sc['max_piston_um']:.0f}\nµm" for sc in per_scale],
                       fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("action magnitude")
    ax.set_title("Action usage per scale")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    if mode == "piston-only":
        mode_desc = (f"piston-only scaling (s × piston only; "
                     f"tip/tilt fixed at "
                     f"{_BASELINE_MAX_TIP_ARCSEC:.0f} arcsec, "
                     f"env_action_scale={_BASELINE_ENV_ACTION_SCALE:.1f})")
    else:
        mode_desc = (f"joint scaling (s × piston, tip, tilt, AND "
                     f"env_action_scale)")
    fig.suptitle(
        f"Action-scale sweep over the 15-phase bootstrap composite agent\n"
        f"baseline (s=1): piston={_BASELINE_MAX_PISTON_UM:.0f} µm    "
        f"mode: {mode_desc}",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep max-action / env-action-scale over the "
                    "15-phase bootstrap composite agent.")
    parser.add_argument("--policy-spec", type=str, default=DEFAULT_SPEC,
                        help="Composite policy spec YAML (default: "
                             f"{DEFAULT_SPEC}).")
    parser.add_argument("--scale-grid", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0],
                        help="Multipliers on the baseline action scales "
                             "(default: 1 10 100 1000, which gives "
                             "piston max = 10/100/1000/10000 µm).")
    parser.add_argument("--scale-mode", choices=list(_SCALE_MODES),
                        default="piston-only",
                        help="Which knobs to multiply by --scale-grid. "
                             "'piston-only' (default) scales only "
                             "max_piston_correction_micron and leaves "
                             "tip/tilt + env_action_scale at their "
                             "training-time defaults. 'joint' scales "
                             "max_piston, max_tip, max_tilt AND "
                             "env_action_scale together (the original "
                             "behavior of this script).")
    parser.add_argument("--seeds-per-scale", type=int, default=2,
                        help="Episodes per scale (each a different env "
                             "reset seed). Default 2.")
    parser.add_argument("--steps-per-phase", type=int, default=None,
                        help="Override per-phase step trigger in the spec.")
    parser.add_argument("--start-at-phase", type=int, default=None)
    parser.add_argument("--run-through-phase", type=int, default=None)
    parser.add_argument("--env-version", type=str, default="v4",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--output-dir", type=str,
                        default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--lowres-gifs", action="store_true")
    args = parser.parse_args()

    if not os.path.isabs(args.policy_spec):
        args.policy_spec = os.path.join(_REPO_ROOT, args.policy_spec)
    if not os.path.isfile(args.policy_spec):
        parser.error(f"Policy spec not found: {args.policy_spec}")

    # Episode length comes from the spec (same logic the per-scale
    # rollout uses).
    num_phases, per_phase, _ = _count_phases_and_steps(
        args.policy_spec, args.steps_per_phase)
    lo = 0 if args.start_at_phase is None else args.start_at_phase
    hi = (num_phases - 1 if args.run_through_phase is None
          else args.run_through_phase)
    effective_phases = hi - lo + 1
    if effective_phases <= 0:
        parser.error(
            f"--start-at-phase ({lo}) must be <= --run-through-phase ({hi})")
    max_steps = per_phase * effective_phases

    effective_spec = _maybe_rewrite_spec(
        args.policy_spec,
        steps_per_phase=args.steps_per_phase,
        start_at_phase=args.start_at_phase,
        run_through_phase=args.run_through_phase)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = (f"bootstrap_action_scale_sweep_{timestamp}_"
              f"{int(time.time()) % 10000}")
    sweep_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"Sweep output: {sweep_dir}")
    print(f"Spec:         {args.policy_spec}")
    print(f"Phases:       {effective_phases} ({lo}..{hi})  "
          f"steps/phase={per_phase}  max_episode_steps={max_steps}")
    print(f"Grid:         {args.scale_grid}")
    print(f"Scale mode:   {args.scale_mode}")
    print(f"Seeds/scale:  {args.seeds_per_scale}")
    print()

    per_scale: list[dict] = []
    base_env_kwargs = dict(ROLLOUT_ENV_KWARGS)
    base_env_kwargs["bootstrap_phased_count"] = lo

    for sc in args.scale_grid:
        scale_label = f"s{sc:g}".replace(".", "p")
        scale_dir = os.path.join(sweep_dir, scale_label)
        os.makedirs(scale_dir, exist_ok=True)
        env_kwargs = _scaled_env_kwargs(base_env_kwargs, sc,
                                        mode=args.scale_mode)
        max_piston_um = env_kwargs["max_piston_correction_micron"]
        print(f"=== scale {sc:g}  ===  max_piston={max_piston_um:.0f} µm  "
              f"max_tip={env_kwargs['max_tip_correction_arcsec']:.1f} arcsec  "
              f"env_action_scale={env_kwargs['env_action_scale']:.3f}")

        episodes, metrics = run_rollouts(
            policy_spec_path=effective_spec,
            env_kwargs=env_kwargs,
            env_version=args.env_version,
            num_episodes=args.seeds_per_scale,
            max_episode_steps=max_steps,
        )

        summaries = [_summarize(ep) for ep in episodes]
        scale_record = dict(
            scale=float(sc),
            max_piston_um=float(max_piston_um),
            max_tip_arcsec=float(env_kwargs["max_tip_correction_arcsec"]),
            max_tilt_arcsec=float(env_kwargs["max_tilt_correction_arcsec"]),
            env_action_scale=float(env_kwargs["env_action_scale"]),
            metrics=metrics,
            summaries=summaries,
        )
        per_scale.append(scale_record)

        # Per-scale metrics + figures + gifs.
        with open(os.path.join(scale_dir, "metrics.json"), "w") as f:
            json.dump({k: v for k, v in scale_record.items()
                       if k != "summaries"}, f, indent=2)
        with open(os.path.join(scale_dir, "per_episode.json"), "w") as f:
            json.dump(summaries, f, indent=2)
        save_summary_figures(episodes, metrics,
                             os.path.join(scale_dir, "figures"))
        if not args.no_gifs:
            save_episode_gifs(episodes,
                              os.path.join(scale_dir, "gifs"),
                              lowres=args.lowres_gifs)

        print(f"  mean return    = {metrics['mean_return']:.4f} "
              f"+/- {metrics['std_return']:.4f}")
        print(f"  mean final S   = {metrics['mean_final_strehl']:.4f} "
              f"+/- {metrics['std_final_strehl']:.4f}")
        print(f"  mean |a|       = "
              f"{np.mean([s['mean_abs_action'] for s in summaries]):.4f}")
        print(f"  sat frac       = "
              f"{np.mean([s['action_saturated_frac'] for s in summaries]):.4f}")
        print()

    # Cross-scale comparison figure + roll-up json.
    rollup_path = os.path.join(sweep_dir, "sweep.json")
    with open(rollup_path, "w") as f:
        json.dump([{k: v for k, v in s.items() if k != "summaries"}
                   for s in per_scale], f, indent=2)
    print(f"Rollup: {rollup_path}")

    cmp_path = os.path.join(sweep_dir, "comparison.png")
    _save_comparison_figure(per_scale, cmp_path, mode=args.scale_mode)
    print(f"Comparison figure: {cmp_path}")

    print(f"\nAll outputs in: {sweep_dir}")


if __name__ == "__main__":
    main()
