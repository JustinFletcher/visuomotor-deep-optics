#!/usr/bin/env python
"""Per-phase rollout instrumentation for the segment_dm_v2 transfer pipeline.

Evaluates each of the 15 trained per-phase policies INDEPENDENTLY,
each starting from its own training-time bootstrap_phased_count
condition. Decouples per-phase performance from the composite
chain so a single failed phase no longer drags every downstream
phase out of distribution.

What it produces in --output-dir:

  per_phase/
    phase_00/
      metrics.json   (per-episode Strehl trajectory, action stats)
      best.gif       (best episode GIF for visual inspection)
    phase_01/...
    ...
    phase_14/...
  strehl_panel.png   (4x4 grid; one subplot per phase showing
                      Strehl vs step for every eval episode plus
                      an overlay of the per-step median across
                      episodes. FAIL/PASS marker per phase.)
  summary.json       (per-phase mean_final_strehl, std,
                      pass/fail verdict, source checkpoint path)

The summary.json's "failed_phases" list is the input for the
retry workflow: re-train just those phases by passing
``--phases <list>`` to launch_segment_zernike_dm_transfer (once
that flag exists -- see TODO at the bottom of this file).

Source modes:
  --source <run-dir>      use checkpoints from a training run dir
                          (e.g. segment_dm_finetuning/<run>/, with
                          phases/phase_NN/ppo_optomech_*/checkpoints/
                          {best,latest}.pt)
  --source <agent-dir>    use checkpoints from a packed agent
                          (e.g. agents/segment_dm_agent_v2/, with
                          checkpoints/phase_NN.pt)

Either is auto-detected from the directory structure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml  # noqa: F401  -- used inline in --env-kwarg parsing

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gymnasium as gym
import matplotlib.pyplot as plt

from train.ppo.train_ppo_optomech import register_optomech
from train.ppo.rollout_elf_bootstrap_ptt import ROLLOUT_ENV_KWARGS
from train.ppo.rollout import (
    load_agent, run_single_episode, save_episode_gifs,
    normalize_obs_fixed, _capture_pupil_opd,
)
from train.ppo.policy_spec import (
    _build_bootstrap_action_mask, read_env_kwarg_overrides,
)
from train.ppo.output_adapters import SegmentZernikePassthroughAdapter
from train.ppo.ppo_models import PPOActorWrapper
from train.ppo.agents import (
    CompositeAgent, NeverTrigger, Phase, SingleModelAgent,
)


_NUM_PHASES = 15
_DEFAULT_STEPS_PER_PHASE = 64       # matches training horizon
_DEFAULT_OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "test_output", "per_phase_eval")


def _discover_checkpoints(source: str) -> list[str]:
    """Return [ckpt_phase_00, ..., ckpt_phase_14] (15 paths).

    Auto-detects run-dir vs agent-dir layout."""
    src = Path(source).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--source not a directory: {source}")

    # Agent-dir layout: checkpoints/phase_NN.pt
    flat_layout = src / "checkpoints" / "phase_00.pt"
    if flat_layout.is_file():
        out = []
        for i in range(_NUM_PHASES):
            p = src / "checkpoints" / f"phase_{i:02d}.pt"
            if not p.is_file():
                raise FileNotFoundError(
                    f"Agent-dir layout: phase {i} checkpoint missing: {p}")
            out.append(str(p))
        return out

    # Run-dir layout: phases/phase_NN/ppo_optomech_*/checkpoints/best.pt
    deep_layout = list((src / "phases").glob("phase_00/ppo_optomech_*"))
    if deep_layout:
        out = []
        for i in range(_NUM_PHASES):
            phase_dir = src / "phases" / f"phase_{i:02d}"
            run_dirs = sorted(phase_dir.glob("ppo_optomech_*"),
                              key=lambda p: p.stat().st_mtime,
                              reverse=True)
            if not run_dirs:
                raise FileNotFoundError(
                    f"Run-dir layout: no ppo_optomech_* under {phase_dir}")
            ck = run_dirs[0] / "checkpoints" / "best.pt"
            if not ck.is_file():
                # Fall back to latest.pt -- sometimes best.pt doesn't
                # exist yet on an early-cancelled run.
                ck = run_dirs[0] / "checkpoints" / "latest.pt"
            if not ck.is_file():
                raise FileNotFoundError(
                    f"Run-dir layout: phase {i} has no best.pt or "
                    f"latest.pt under {run_dirs[0]}")
            out.append(str(ck))
        return out

    raise FileNotFoundError(
        f"--source {source}: neither agent-dir "
        f"(checkpoints/phase_00.pt) nor run-dir "
        f"(phases/phase_00/ppo_optomech_*/) layout detected.")


def _build_per_phase_agent(ckpt_path: str, env, device: str,
                           phase_idx: int, n_zernike: int,
                           n_seg_actions: int = 45,
                           dm_action_scale: float = 0.01):
    """Load the per-phase policy and wrap it as a 1-phase composite
    with the bootstrap action mask + SegmentZernikePassthroughAdapter
    that the rollout machinery expects. Same wiring as the composite
    rollout path; just truncated to a single phase."""
    native_action_dim = n_seg_actions + n_zernike
    model, ckpt_cfg, _orm = load_agent(
        ckpt_path, env, device=device,
        action_dim_override=native_action_dim)
    wrapper = PPOActorWrapper(model)
    single = SingleModelAgent(wrapper, device)

    adapter = SegmentZernikePassthroughAdapter.from_env(
        env, n_zernike=n_zernike,
        n_seg_actions=n_seg_actions,
        dm_action_scale=dm_action_scale,
        silence=True)

    # Override bootstrap_phased_count in the checkpoint's stored
    # config so the mask builder picks the right target segment.
    ck_env = dict((ckpt_cfg or {}).get("env_kwargs", {}) or {})
    ck_env["bootstrap_phased_count"] = phase_idx
    config_for_mask = dict(ckpt_cfg or {})
    config_for_mask["env_kwargs"] = ck_env

    action_mask = _build_bootstrap_action_mask(
        config_for_mask, native_action_dim, adapter=adapter)

    phase = Phase(
        agent=single,
        trigger=NeverTrigger(),
        name=f"phase-{phase_idx:02d}",
        action_mask=action_mask,
        bootstrap_phased_count=phase_idx,
        output_adapter=adapter)
    return CompositeAgent([phase], device=device), action_mask


def _build_env(env_kwargs: dict, max_episode_steps: int):
    register_optomech("optomech-v4", max_episode_steps=max_episode_steps)
    env = gym.make("optomech-v4", **env_kwargs)
    return env


def _strehl_panel(per_phase_episodes: dict, output_path: str,
                  strehl_threshold: float, steps_per_phase: int):
    """4x4 grid of Strehl-vs-step plots, one per phase + one empty.

    Each subplot:
      - faint lines: individual episode Strehl trajectories
      - bold line: per-step median across episodes
      - title: PHASE NN -- PASS / FAIL (based on median final Strehl
        vs threshold)
      - red background tint when FAIL
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True, sharey=True)
    for ax_idx, ax in enumerate(axes.flat):
        if ax_idx >= _NUM_PHASES:
            ax.axis("off")
            continue
        phase = ax_idx
        eps = per_phase_episodes.get(phase, [])
        if not eps:
            ax.text(0.5, 0.5, "no data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10)
            ax.set_title(f"phase-{phase:02d}  --  no data",
                         fontsize=10)
            continue

        # Per-episode Strehl curves + median overlay.
        all_curves = []
        for ep in eps:
            s = ep["strehls"]
            x = np.arange(1, len(s) + 1)
            ax.plot(x, s, color="0.65", linewidth=0.7, alpha=0.7)
            all_curves.append(np.asarray(s, dtype=np.float32))
        # Pad to common length for median (in case any episode is
        # short -- shouldn't happen but be safe).
        max_len = max(len(c) for c in all_curves)
        padded = np.full((len(all_curves), max_len), np.nan)
        for i, c in enumerate(all_curves):
            padded[i, :len(c)] = c
        median = np.nanmedian(padded, axis=0)
        ax.plot(np.arange(1, max_len + 1), median,
                color="C0", linewidth=1.8, label="median")

        # Final Strehl summary.
        finals = np.array([c[-1] for c in all_curves], dtype=np.float32)
        final_median = float(np.median(finals))
        passed = final_median >= strehl_threshold
        verdict = "PASS" if passed else "FAIL"
        color = "green" if passed else "red"
        ax.set_title(
            f"phase-{phase:02d}  --  {verdict}  "
            f"(med S_final={final_median:.3f})",
            fontsize=9, color=color)
        if not passed:
            ax.set_facecolor("#fee")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(strehl_threshold, color="k", linestyle="--",
                   linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

    # Common axis labels on the outer edges.
    for ax in axes[-1, :]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Strehl")
    fig.suptitle(
        f"Per-phase independent rollouts -- "
        f"threshold={strehl_threshold:.2f}, "
        f"steps/phase={steps_per_phase}",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Per-phase independent rollout for segment_dm_v2 "
                    "diagnostic / retry-list generation.")
    p.add_argument("--source", required=True,
                   help="Run dir (segment_dm_finetuning/<run>/) or "
                        "agent dir (agents/segment_dm_agent_v2/). "
                        "Layout auto-detected.")
    p.add_argument("--num-episodes", type=int, default=2,
                   help="Episodes per phase (default 2).")
    p.add_argument("--steps-per-phase", type=int,
                   default=_DEFAULT_STEPS_PER_PHASE,
                   help=f"Episode length per phase (default "
                        f"{_DEFAULT_STEPS_PER_PHASE}, matching the "
                        f"training horizon).")
    p.add_argument("--strehl-threshold", type=float, default=0.5,
                   help="Median final Strehl below this marks the "
                        "phase FAILED in the summary + panel (default "
                        "0.5).")
    p.add_argument("--n-zernike", type=int, default=32,
                   help="Zernike-DM head width baked into each phase "
                        "policy. Default 32 (matches segment_dm_v2 "
                        "training).")
    p.add_argument("--dm-action-scale", type=float, default=0.01,
                   help="DM-block action scale applied inside the "
                        "passthrough adapter. Default 0.01 "
                        "(matches training).")
    p.add_argument("--env-kwarg", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="env_kwarg override (repeatable). Same parser "
                        "as rollout_elf_bootstrap_ptt: yaml-scalar "
                        "values, including inline mappings like "
                        "'atmosphere={r0_total_m: 0.25, L0_m: 25.0, "
                        "static: true}'.")
    p.add_argument("--output-dir", type=str, default=None,
                   help=f"Output dir. Default: {_DEFAULT_OUTPUT_DIR}_<ts>/.")
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--lowres-gifs", action="store_true")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; each (phase, episode) gets seed = "
                        "base + phase*1000 + episode for "
                        "reproducibility.")
    args = p.parse_args()

    # ----------------------------------------------------------------
    # Output dir
    # ----------------------------------------------------------------
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{_DEFAULT_OUTPUT_DIR}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output:  {args.output_dir}")

    # ----------------------------------------------------------------
    # Discover checkpoints + base env_kwargs
    # ----------------------------------------------------------------
    checkpoints = _discover_checkpoints(args.source)
    print(f"Found 15 checkpoints under {args.source}:")
    for i, ck in enumerate(checkpoints):
        print(f"  phase_{i:02d}: {os.path.relpath(ck)}")

    # Base env_kwargs: ROLLOUT_ENV_KWARGS + the same env_kwarg_overrides
    # the segment_dm_agent_v2 composed spec declares (command_dm,
    # dm_incremental_control, ...) so the env is set up the same way
    # training was. If a composed.yaml is reachable from --source,
    # honour its env_kwarg_overrides too.
    base_env_kwargs = dict(ROLLOUT_ENV_KWARGS)
    base_env_kwargs.update({
        "command_secondaries": True,
        "command_dm": True,
        "dm_incremental_control": True,
    })
    # Try to load env_kwarg_overrides from <source>/composed.yaml if
    # present (agent-dir layout); harmless if absent.
    spec_path = os.path.join(args.source, "composed.yaml")
    if os.path.isfile(spec_path):
        spec_overrides = read_env_kwarg_overrides(spec_path)
        if spec_overrides:
            base_env_kwargs.update(spec_overrides)
            print(f"Applied env_kwarg_overrides from {spec_path}: "
                  f"{list(spec_overrides.keys())}")

    # --env-kwarg overrides (user CLI).
    for entry in args.env_kwarg:
        if "=" not in entry:
            p.error(f"--env-kwarg expects KEY=VALUE, got {entry!r}")
        k, raw = entry.split("=", 1)
        v = yaml.safe_load(raw)
        base_env_kwargs[k] = v
        print(f"  env_kwarg override: {k} = {v!r}")

    # ----------------------------------------------------------------
    # Per-phase rollouts
    # ----------------------------------------------------------------
    per_phase_episodes: dict[int, list] = {}
    per_phase_meta: dict[int, dict] = {}
    for phase_idx in range(_NUM_PHASES):
        ck = checkpoints[phase_idx]
        phase_dir = os.path.join(args.output_dir,
                                 "per_phase", f"phase_{phase_idx:02d}")
        os.makedirs(phase_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"Phase {phase_idx:02d}   ckpt: {os.path.relpath(ck)}")
        print(f"{'=' * 60}")

        # Env: bootstrap_phased_count = this phase's index.
        env_kwargs = dict(base_env_kwargs)
        env_kwargs["bootstrap_phased_count"] = phase_idx
        env = _build_env(env_kwargs, args.steps_per_phase)
        env.reset(seed=args.seed + phase_idx * 1000)

        # Agent: load the per-phase policy, wrap with adapter + mask.
        agent, mask = _build_per_phase_agent(
            ck, env, args.device, phase_idx,
            n_zernike=args.n_zernike,
            dm_action_scale=args.dm_action_scale)
        mask_nonzero = (int(mask.nonzero().numel())
                        if mask is not None else "no mask")
        print(f"  agent native action_dim: "
              f"{agent._phases[0].agent.action_dim}, "
              f"env action_dim: {env.action_space.shape[0]}, "
              f"mask non-zero DOFs: {mask_nonzero}")

        # obs_ref_max from env's reference flux (used for image norm).
        base = env.unwrapped
        obs_ref_max = float(getattr(
            getattr(base, "optical_system", None),
            "_reference_fpi_max", 1.0))

        # Run K episodes for this phase.
        eps = []
        for ep_i in range(args.num_episodes):
            seed = args.seed + phase_idx * 1000 + ep_i
            # Reset BEFORE the rollout so each episode has a fresh draw.
            env.reset(seed=seed)
            ep = run_single_episode(
                agent, env, seed=seed,
                obs_ref_max=obs_ref_max, device=args.device)
            eps.append(ep)
            print(f"  episode {ep_i + 1}/{args.num_episodes}: "
                  f"return={ep['return']:.3f}  "
                  f"final_strehl={ep['strehls'][-1]:.4f}")
        per_phase_episodes[phase_idx] = eps

        # Per-phase metrics + GIF.
        finals = [float(e["strehls"][-1]) for e in eps]
        per_phase_meta[phase_idx] = {
            "checkpoint": ck,
            "n_episodes": len(eps),
            "final_strehl_median": float(np.median(finals)),
            "final_strehl_mean": float(np.mean(finals)),
            "final_strehl_std": float(np.std(finals)),
            "final_strehl_min": float(np.min(finals)),
            "final_strehl_max": float(np.max(finals)),
            "passed": bool(np.median(finals) >= args.strehl_threshold),
        }

        # Save metrics to per-phase dir.
        with open(os.path.join(phase_dir, "metrics.json"), "w") as f:
            json.dump({
                "phase": phase_idx,
                "checkpoint": ck,
                "episodes": [
                    {"seed": e["seed"],
                     "return": float(e["return"]),
                     "strehls": [float(s) for s in e["strehls"]],
                     "length": e["length"]}
                    for e in eps],
                **per_phase_meta[phase_idx],
            }, f, indent=2)

        # GIFs (best/worst/median).
        if not args.no_gifs:
            save_episode_gifs(
                eps, os.path.join(phase_dir, "gifs"),
                lowres=args.lowres_gifs,
                frame_stride=args.frame_stride)

        env.close()

    # ----------------------------------------------------------------
    # Panel + summary
    # ----------------------------------------------------------------
    panel_path = os.path.join(args.output_dir, "strehl_panel.png")
    _strehl_panel(per_phase_episodes, panel_path,
                  strehl_threshold=args.strehl_threshold,
                  steps_per_phase=args.steps_per_phase)
    print(f"\nPanel: {panel_path}")

    summary = {
        "source": os.path.abspath(args.source),
        "num_episodes": args.num_episodes,
        "steps_per_phase": args.steps_per_phase,
        "strehl_threshold": args.strehl_threshold,
        "env_kwarg_overrides_applied": list(base_env_kwargs.keys()),
        "phases": {
            f"{idx:02d}": per_phase_meta[idx]
            for idx in range(_NUM_PHASES)},
        "passed_phases": [
            idx for idx in range(_NUM_PHASES)
            if per_phase_meta[idx]["passed"]],
        "failed_phases": [
            idx for idx in range(_NUM_PHASES)
            if not per_phase_meta[idx]["passed"]],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{'=' * 60}")
    print(f"  PASSED ({len(summary['passed_phases'])}/15): "
          f"{summary['passed_phases']}")
    print(f"  FAILED ({len(summary['failed_phases'])}/15): "
          f"{summary['failed_phases']}")
    print(f"{'=' * 60}")
    print()
    print(f"Summary: {os.path.join(args.output_dir, 'summary.json')}")
    if summary["failed_phases"]:
        print(f"\nTo retry just the failed phases, pass them to a "
              f"future --phases flag on launch_segment_zernike_dm_"
              f"transfer.py: --phases "
              f"{','.join(str(p) for p in summary['failed_phases'])}")


if __name__ == "__main__":
    main()
