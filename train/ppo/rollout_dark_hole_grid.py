#!/usr/bin/env python
"""Roll out a trained target-aware dark-hole policy on each grid target.

For each of the 16 dark-hole targets defined by ``build_grid()`` in
``launch_dark_hole_grid.py``, this script builds a v4 env with the
target's geometry baked in, runs one deterministic episode under the
provided checkpoint, and writes an animated GIF that overlays the
target hole as a dotted cyan circle on the focal-plane image.

Intended use case: a dynamic-dark-hole-trained policy (i.e. one
checkpoint that learned the [sin θ, cos θ, radius, size] target input)
evaluated against the static grid to characterise per-target
performance from a single agent.

Usage:
    python train/ppo/rollout_dark_hole_grid.py \\
        --checkpoint runs/<dynamic-run>/checkpoints/best.pt \\
        --output-dir test_output/dh_grid_eval/

    # Limit to one target id for quick iteration:
    python train/ppo/rollout_dark_hole_grid.py --checkpoint ... \\
        --output-dir test_output/dh_one/ --target-id 5
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.ppo.launch_dark_hole_grid import build_grid                # noqa: E402
from train.ppo.ppo_models import RecurrentActorCritic                  # noqa: E402
from train.ppo.train_ppo_elf_dark_hole import (                        # noqa: E402
    ELF_DARK_HOLE_ENV_KWARGS,
)
from train.ppo.train_ppo_optomech import (                             # noqa: E402
    normalize_obs_fixed,
    register_optomech,
)


class _EnvShim:
    """Minimal envs.* shim for RecurrentActorCritic construction."""
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


def _load_agent(ckpt_path, env, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    agent = RecurrentActorCritic(
        _EnvShim(env), torch.device(device),
        lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
        channel_scale=config.get("channel_scale", 32),
        fc_scale=config.get("fc_scale", 256),
        action_scale=config.get("action_scale", 1.0),
        init_log_std=config.get("init_log_std", -0.5),
        model_type=config.get("model_type", "small"),
        target_dim=int(config.get("target_dim", 0)),
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent, config


def _build_env(target, max_steps):
    angle, radius_frac, size_frac = target
    kw = dict(ELF_DARK_HOLE_ENV_KWARGS)
    kw["dark_hole"] = True
    kw["dark_hole_angular_location_degrees"] = float(angle)
    kw["dark_hole_location_radius_fraction"] = float(radius_frac)
    kw["dark_hole_size_radius"] = float(size_frac)
    kw["dark_hole_randomize_on_reset"] = False
    kw["max_episode_steps"] = int(max_steps)
    kw["silence"] = True
    kw["observation_window_size"] = 1
    register_optomech("optomech-v4", max_episode_steps=int(max_steps))
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make("optomech-v4", **kw)
    return env


def _capture_diagnostics(base):
    """Pull (OPD, raw_psf, contrast, peak_brightness) from the env.

    OPD: pupil-plane surface in metres, computed on-the-fly from the
    current actuator state and influence functions.
    raw_psf: polychromatic, pre-detector intensity frame stashed by v4
    in OpticalSystem._last_raw_psf.
    contrast: min(raw_psf[hole]) / max(raw_psf), evaluated at full
    intensity precision so the typical 1e-6..1e-10 dark-hole regime
    isn't truncated by detector quantization.
    peak_brightness: max(raw_psf), full intensity precision (units are
    whatever the v4 polychromatic accumulator emits — only the
    relative shape across steps matters).
    """
    os4 = base.optical_system
    actuators = os4._actuators_t              # [n_modes]
    infl = os4._influence_t                   # [n_modes, H, W]
    opd = torch.einsum("i,ihw->hw", actuators, infl).detach().cpu().numpy()
    # Pre-detector polychromatic frame is stashed on the env, not the
    # optical system (it accumulates across the per-step wavelength loop
    # in OptomechEnv.step before the detector model runs).
    raw_psf = getattr(base, "_last_raw_psf", None)
    if raw_psf is None:
        return opd, None, float("nan"), float("nan")
    raw_psf = np.asarray(raw_psf)
    mask = base._target_zero_mask
    psf_max = float(np.max(raw_psf))
    if mask is None or psf_max <= 0.0:
        contrast = float("nan")
    else:
        contrast = float(np.min(raw_psf[mask]) / psf_max)
    return opd, raw_psf, contrast, psf_max


def run_episode(agent, env, target, seed, device):
    """One deterministic rollout. Returns dict of per-step data."""
    angle, radius_frac, size_frac = target
    th = np.deg2rad(angle)
    tv_np = np.array(
        [np.sin(th), np.cos(th), radius_frac, size_frac], dtype=np.float32)
    tv = torch.from_numpy(tv_np).unsqueeze(0).to(device)

    base = env.unwrapped
    obs_ref_max = float(getattr(base, "reference_fpi_max", 1.0))

    obs_raw, _ = env.reset(seed=seed)
    obs_np = normalize_obs_fixed(obs_raw[np.newaxis], obs_ref_max)

    h = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(1, agent.action_dim, device=device)
    prior_reward = torch.zeros(1, device=device)

    # Capture initial-state diagnostics (the warm-up frame inside reset
    # already populated _last_raw_psf).
    opd0, psf0, ct0, pk0 = _capture_diagnostics(base)

    rewards, actions = [], []
    obs_list = [obs_raw.copy()]
    opd_list = [opd0]
    psf_list = [psf0]
    contrast_list = [ct0]
    peak_list = [pk0]
    strehls = []

    done = False
    while not done:
        obs_t = torch.from_numpy(obs_np).float().to(device)
        with torch.no_grad():
            action_t, (h, c) = agent.get_deterministic_action(
                obs_t, prior_action, prior_reward, (h, c), target_vec=tv)
        env_action = action_t.cpu().numpy()[0]
        next_obs, reward, term, trunc, info = env.step(env_action)
        done = bool(term or trunc)
        rewards.append(float(reward))
        actions.append(env_action.copy())
        obs_list.append(next_obs.copy())
        opd_t, psf_t, ct_t, pk_t = _capture_diagnostics(base)
        opd_list.append(opd_t)
        psf_list.append(psf_t)
        contrast_list.append(ct_t)
        peak_list.append(pk_t)
        if "strehl" in info:
            strehls.append(float(info["strehl"]))
        prior_action = action_t
        prior_reward = torch.tensor(
            [reward], dtype=torch.float32, device=device)
        obs_np = normalize_obs_fixed(next_obs[np.newaxis], obs_ref_max)

    return {
        "rewards": rewards,
        "actions": np.array(actions),
        "obs_raw": obs_list,
        "opd": opd_list,
        "raw_psf": psf_list,
        "contrast": contrast_list,
        "peak": peak_list,
        "strehls": strehls,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "seed": int(seed),
        "target": target,
    }


def _prep_obs(o):
    """Strip a single-frame channel dim if present, returning [H, W]."""
    a = np.asarray(o)
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1:
        return a[0, 0]
    return a


def render_gif(ep_data, save_path, dpi=110, frame_duration=0.1):
    """4-panel animated GIF: OPD | raw PSF | observation | contrast trace.

    All three image panels carry the target dark-hole circle overlaid as
    a dotted cyan outline. The contrast trace (raw_psf median in the
    hole / raw_psf max) is on a log scale and accumulates frame-by-frame
    so the viewer sees the depth evolve.
    """
    target = ep_data["target"]
    angle, radius_frac, size_frac = target
    target_id = ep_data.get("target_id", -1)

    obs_raw = ep_data["obs_raw"]
    opds = ep_data["opd"]
    psfs = ep_data["raw_psf"]
    contrasts = np.array(ep_data["contrast"], dtype=np.float64)
    peaks = np.array(ep_data.get("peak", []), dtype=np.float64)
    rewards = ep_data["rewards"]
    strehls = ep_data["strehls"]
    cumulative = np.cumsum(rewards)
    T = len(rewards)

    obs_imgs = [_prep_obs(obs_raw[t]) for t in range(T + 1)]
    H, W = obs_imgs[0].shape[-2:]
    # Image-panel scales are computed per-frame below so each frame
    # uses its own dynamic range (colors shift between frames, but
    # nothing saturates against an episode-wide ceiling that may live
    # many decades above what's actually present at this step).
    th = np.deg2rad(angle)
    cx_px = W / 2.0 + radius_frac * (W / 2.0) * np.cos(th)
    cy_px = H / 2.0 + radius_frac * (H / 2.0) * np.sin(th)
    r_px = size_frac * (W / 2.0)

    def _linear_bounds(arr):
        finite = arr[np.isfinite(arr)]
        if not finite.size:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + abs(lo) * 0.1 + 1e-12
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    ct_lo, ct_hi = _linear_bounds(contrasts)
    pk_lo, pk_hi = _linear_bounds(peaks)
    timesteps = np.arange(T + 1)

    def _circle(ax):
        ax.add_patch(Circle(
            (cx_px, cy_px), r_px, fill=False, edgecolor="cyan",
            linestyle=(0, (1.5, 1.5)), linewidth=1.2, alpha=0.95))

    # Layout sized so each top panel is ~square (good fit for 256x256
    # imshow), the bottom plot has reasonable headroom, and panel
    # margins are tight.
    frames = []
    TITLE_FS = 10
    TICK_FS = 8
    CB_FS = 7
    SUP_FS = 11

    for t in range(T + 1):
        # Top row: 3 near-square image panels (aspect="equal").
        # Bottom: stacked contrast trace + peak-brightness trace.
        # Right margin leaves ~6% for the rightmost colorbar's labels.
        fig = plt.figure(figsize=(9.0, 4.7), dpi=dpi)
        gs = fig.add_gridspec(
            3, 3,
            height_ratios=[3.8, 0.7, 0.7],
            hspace=0.55, wspace=0.04,
            left=0.025, right=0.94, top=0.88, bottom=0.10,
        )

        # Panel 1: OPD (per-frame symmetric scale).
        ax_opd = fig.add_subplot(gs[0, 0])
        opd = opds[t]
        opd_show = np.where(np.abs(opd) < 1e-14, np.nan, opd)
        opd_max_t = float(np.nanmax(np.abs(opd_show))) if np.any(
            np.isfinite(opd_show)) else 1e-12
        opd_max_t = max(opd_max_t, 1e-12)
        im_opd = ax_opd.imshow(
            opd_show, cmap="RdBu_r", origin="lower",
            vmin=-opd_max_t, vmax=opd_max_t, aspect="equal")
        _circle(ax_opd)
        ax_opd.set_xticks([]); ax_opd.set_yticks([])
        ax_opd.set_title("OPD (m)", fontsize=TITLE_FS, pad=3)
        cb = fig.colorbar(im_opd, ax=ax_opd, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=CB_FS)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()

        # Panel 2: raw PSF (per-frame log scale).
        ax_psf = fig.add_subplot(gs[0, 1])
        psf = psfs[t] if psfs[t] is not None else np.zeros((H, W))
        psf_max_t = max(float(np.max(psf)), 1e-30)
        psf_floor_t = max(psf_max_t * 1e-8, 1e-30)
        psf_norm_t = mcolors.LogNorm(vmin=psf_floor_t, vmax=psf_max_t)
        im_psf = ax_psf.imshow(
            np.maximum(psf, psf_floor_t), cmap="inferno",
            norm=psf_norm_t, origin="lower", aspect="equal")
        _circle(ax_psf)
        ax_psf.set_xticks([]); ax_psf.set_yticks([])
        ax_psf.set_title("raw PSF (pre-detector, log)",
                         fontsize=TITLE_FS, pad=3)
        cb = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=CB_FS)

        # Panel 3: observation (per-frame log scale).
        ax_obs = fig.add_subplot(gs[0, 2])
        obs_max_t = max(float(np.max(obs_imgs[t])), 2.0)
        obs_norm_t = mcolors.LogNorm(vmin=1.0, vmax=obs_max_t)
        im_obs = ax_obs.imshow(
            np.maximum(obs_imgs[t], 1.0), cmap="inferno",
            norm=obs_norm_t, origin="lower", aspect="equal")
        _circle(ax_obs)
        ax_obs.set_xticks([]); ax_obs.set_yticks([])
        ax_obs.set_title("detector obs (DN, log)",
                         fontsize=TITLE_FS, pad=3)
        cb = fig.colorbar(im_obs, ax=ax_obs, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=CB_FS)

        # Panel 4: contrast trace (linear).
        ax_ct = fig.add_subplot(gs[1, :])
        ax_ct.set_ylim(ct_lo, ct_hi)
        ax_ct.set_xlim(0, max(T, 1))
        ax_ct.plot(timesteps, contrasts,
                   color="#888888", lw=0.7, alpha=0.4)
        ax_ct.plot(timesteps[: t + 1], contrasts[: t + 1],
                   color="#1f3b5e", lw=1.6)
        if np.isfinite(contrasts[t]):
            ax_ct.plot([t], [contrasts[t]], "o",
                       color="#d94a4a", markersize=5)
        ax_ct.grid(True, alpha=0.3, lw=0.4)
        ax_ct.tick_params(labelsize=TICK_FS, labelbottom=False)
        ax_ct.set_title("Contrast (min(hole) / max(PSF))",
                        fontsize=TITLE_FS, pad=2)

        # Panel 5: peak brightness trace (linear).
        ax_pk = fig.add_subplot(gs[2, :], sharex=ax_ct)
        ax_pk.set_ylim(pk_lo, pk_hi)
        ax_pk.plot(timesteps, peaks,
                   color="#888888", lw=0.7, alpha=0.4)
        ax_pk.plot(timesteps[: t + 1], peaks[: t + 1],
                   color="#2a6f4d", lw=1.6)
        if t < len(peaks) and np.isfinite(peaks[t]):
            ax_pk.plot([t], [peaks[t]], "o",
                       color="#d94a4a", markersize=5)
        ax_pk.grid(True, alpha=0.3, lw=0.4)
        ax_pk.tick_params(labelsize=TICK_FS)
        ax_pk.set_xlabel("step", fontsize=TICK_FS + 1)
        ax_pk.set_title("Peak brightness (max(PSF))",
                        fontsize=TITLE_FS, pad=2)

        head = (f"target {target_id:02d}  "
                f"angle={angle:5.1f}°  r={radius_frac:.3f}  "
                f"size={size_frac:.3f}")
        if t == 0:
            sub = f"t=0  (initial)  contrast={contrasts[0]:.2e}"
        else:
            r = rewards[t - 1]
            c = cumulative[t - 1]
            s = f"  S={strehls[t-1]:.3f}" if strehls else ""
            sub = (f"t={t:>3d}  r={r:+.3f}  Σ={c:+.2f}{s}  "
                   f"contrast={contrasts[t]:.2e}")
        fig.suptitle(f"{head}\n{sub}", fontsize=SUP_FS, y=0.985)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(save_path, frames, duration=frame_duration)


def main():
    parser = argparse.ArgumentParser(
        description="Roll a target-aware dark-hole policy across the grid.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a target-aware checkpoint (config.target_dim > 0).")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write target_NN.gif files into.")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--target-id", type=int, default=None,
        help="Limit to one target id (0..15) for quick iteration.")
    parser.add_argument("--frame-duration", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    targets = build_grid()
    if args.target_id is not None:
        if not (0 <= args.target_id < len(targets)):
            print(f"Error: --target-id must be in [0, {len(targets)-1}]")
            sys.exit(1)
        target_indices = [args.target_id]
    else:
        target_indices = list(range(len(targets)))

    os.makedirs(args.output_dir, exist_ok=True)

    # Build a one-off env to instantiate the agent. The space shape is
    # geometry-independent, so any target works.
    env0 = _build_env(targets[target_indices[0]], args.max_steps)
    agent, config = _load_agent(args.checkpoint, env0, args.device)
    td = int(config.get("target_dim", 0))
    print(f"Loaded checkpoint: target_dim={td}")
    if td == 0:
        print("WARNING: target_dim=0 — checkpoint is target-blind. "
              "It will receive zero target_vec input and treat every "
              "geometry identically.")
    env0.close()

    summary = []
    for i in target_indices:
        target = targets[i]
        env = _build_env(target, args.max_steps)
        ep = run_episode(agent, env, target, args.seed, args.device)
        ep["target_id"] = i
        gif_path = os.path.join(args.output_dir, f"target_{i:02d}.gif")
        render_gif(ep, gif_path, dpi=args.dpi,
                   frame_duration=args.frame_duration)
        env.close()
        final_s = ep["strehls"][-1] if ep["strehls"] else float("nan")
        final_ct = ep["contrast"][-1] if ep["contrast"] else float("nan")
        best_ct = (min(c for c in ep["contrast"]
                       if np.isfinite(c) and c > 0)
                   if any(np.isfinite(c) and c > 0 for c in ep["contrast"])
                   else float("nan"))
        summary.append((i, target, ep["return"], final_s, final_ct, best_ct))
        print(f"  target {i:>2}: angle={target[0]:6.1f}  "
              f"r={target[1]:.3f}  size={target[2]:.3f}  "
              f"R={ep['return']:+.3f}  final_S={final_s:.4f}  "
              f"final_C={final_ct:.2e}  best_C={best_ct:.2e}  "
              f"-> {gif_path}")

    print(f"\nWrote {len(summary)} GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
