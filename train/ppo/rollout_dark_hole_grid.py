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

    rewards, actions = [], []
    obs_list = [obs_raw.copy()]
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


def render_gif(ep_data, save_path, dpi=72, frame_duration=0.1):
    """Animated GIF of the focal-plane evolution with target overlay."""
    target = ep_data["target"]
    angle, radius_frac, size_frac = target
    target_id = ep_data.get("target_id", -1)

    obs_raw = ep_data["obs_raw"]
    rewards = ep_data["rewards"]
    strehls = ep_data["strehls"]
    cumulative = np.cumsum(rewards)
    T = len(rewards)

    all_raw = [_prep_obs(obs_raw[t]) for t in range(T + 1)]
    global_max = max(float(np.max(img)) for img in all_raw)
    global_max = max(global_max, 1.0)
    norm = mcolors.LogNorm(vmin=1.0, vmax=global_max)

    H, W = all_raw[0].shape[-2:]
    th = np.deg2rad(angle)
    cx_px = W / 2.0 + radius_frac * (W / 2.0) * np.cos(th)
    cy_px = H / 2.0 + radius_frac * (H / 2.0) * np.sin(th)
    r_px = size_frac * (W / 2.0)

    frames = []
    for t in range(T + 1):
        fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=dpi)
        im = ax.imshow(np.maximum(all_raw[t], 1.0),
                       cmap="inferno", norm=norm, origin="lower")
        ax.add_patch(Circle(
            (cx_px, cy_px), r_px, fill=False,
            edgecolor="cyan", linestyle=(0, (1.5, 1.5)),
            linewidth=0.9, alpha=0.95))
        ax.set_xticks([]); ax.set_yticks([])
        head = (f"target {target_id:02d}  "
                f"angle={angle:5.1f}°  r={radius_frac:.3f}  "
                f"size={size_frac:.3f}")
        if t == 0:
            sub = "t=0  (initial)"
        else:
            r = rewards[t - 1]
            c = cumulative[t - 1]
            s = f"  S={strehls[t-1]:.3f}" if strehls else ""
            sub = f"t={t:>3d}  r={r:+.3f}  Σ={c:+.2f}{s}"
        ax.set_title(f"{head}\n{sub}", fontsize=7, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
        fig.tight_layout(pad=0.3)
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
    parser.add_argument("--dpi", type=int, default=72)
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
        summary.append((i, target, ep["return"], final_s))
        print(f"  target {i:>2}: angle={target[0]:6.1f}  "
              f"r={target[1]:.3f}  size={target[2]:.3f}  "
              f"R={ep['return']:+.3f}  final_S={final_s:.4f}  "
              f"-> {gif_path}")

    print(f"\nWrote {len(summary)} GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
