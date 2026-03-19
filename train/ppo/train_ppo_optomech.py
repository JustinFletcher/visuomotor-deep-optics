"""
Shared PPO training infrastructure for optomech environments.

This module contains all the machinery needed to train a recurrent PPO agent
on any optomech telescope-alignment environment. Individual experiment scripts
(e.g. ``train_ppo_nanoelf_piston.py``, ``train_ppo_nanoelf_ptt.py``) define
their own env kwargs and PPO hyperparameters, then call :func:`run_main` from
here.

Exported API
------------
- ``run_main(full_config, fast_config)`` — CLI entry point
- ``run_ppo_training(config, run_dir)`` — training loop
- ``evaluate_with_visualization(...)`` — eval with TensorBoard figures
- ``evaluate_zero_policy / evaluate_random_policy`` — baselines
- ``normalize_obs_fixed`` — fixed-reference observation normalization
- ``register_optomech`` / ``make_optomech_env`` — environment helpers
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import imageio
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper
from models.td3_models import conv_init
from train.ppo.ppo_recurrent import normalize_obs, compute_gae, recurrent_generator


# ============================================================================
# Environment helpers
# ============================================================================

# Module-level env ID, set at runtime by run_main.
_ENV_ID = "optomech-v4"

_ENTRY_POINTS = {
    "optomech-v1": "optomech.optomech.optomech:OptomechEnv",
    "optomech-v2": "optomech.optomech.optomech_v2:OptomechEnv",
    "optomech-v3": "optomech.optomech.optomech_v3:OptomechEnv",
    "optomech-v4": "optomech.optomech.optomech_v4:OptomechEnv",
}


def register_optomech(env_id: str, max_episode_steps=100):
    """Register the specified optomech version."""
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]

    entry_point = _ENTRY_POINTS[env_id]
    gym.envs.registration.register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
    )


def make_optomech_env(env_kwargs: dict, max_episode_steps: int = 100, idx: int = 0):
    """Create an optomech environment factory."""

    def thunk():
        kwargs = dict(env_kwargs)
        kwargs["max_episode_steps"] = max_episode_steps
        env = gym.make(_ENV_ID, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ============================================================================
# Fixed-reference observation normalization
# ============================================================================


def normalize_obs_fixed(obs: np.ndarray, ref_max: float) -> np.ndarray:
    """Normalize observations to float32 using a fixed reference maximum.

    Unlike per-sample max normalization, this preserves relative flux
    information: when the PSF moves off the detector, the observation
    becomes dimmer, giving the agent a gradient to follow.

    Parameters
    ----------
    obs : ndarray
        Raw observations (single or batched).
    ref_max : float
        Reference maximum pixel value (peak DN when PSF is perfectly aligned).
    """
    return obs.astype(np.float32) / max(ref_max, 1e-10)


# ============================================================================
# Observation display helpers
# ============================================================================


def _prepare_obs_for_display(obs):
    """
    Prepare an optomech focal-plane observation for matplotlib display.

    Shows exactly what the model receives: raw observation passed through
    normalize_obs (per-sample max normalization to [0, 1]), first channel.
    """
    img = normalize_obs(np.asarray(obs)[np.newaxis])[0]
    # Handle (C, H, W) or (H, W, C) or (H, W)
    if img.ndim == 3:
        if img.shape[0] <= img.shape[-1]:  # channels-first
            img = img[0]
        else:
            img = img[:, :, 0]
    return img


def _prepare_obs_raw(obs):
    """Extract raw 2-D focal-plane image in detector DN (no normalization).

    Use this instead of ``_prepare_obs_for_display`` when the absolute
    signal scale matters (e.g. for colorbars showing DN values).
    """
    img = np.asarray(obs, dtype=np.float32)
    if img.ndim == 3:
        if img.shape[0] <= img.shape[-1]:  # channels-first
            img = img[0]
        else:
            img = img[:, :, 0]
    return img


# ============================================================================
# Evaluation rollout with instrumentation
# ============================================================================


def _run_zero_action_episode(eval_env, seed):
    """Run a single zero-action episode and return (return, first_reward, rewards)."""
    obs_raw, _ = eval_env.reset(seed=seed)
    done = False
    rewards = []
    while not done:
        action = np.zeros(eval_env.action_space.shape, dtype=np.float32)
        obs_raw, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
    ep_return = sum(rewards)
    first_reward = rewards[0] if rewards else 0.0
    return ep_return, first_reward, rewards


def _run_random_action_episode(eval_env, seed):
    """Run a single random-action episode and return (return, rewards)."""
    obs_raw, _ = eval_env.reset(seed=seed)
    done = False
    rewards = []
    while not done:
        action = eval_env.action_space.sample()
        obs_raw, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
    return sum(rewards), rewards


def evaluate_with_visualization(
    agent: RecurrentActorCritic,
    config: dict,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    tag_prefix: str = "eval",
    num_episodes: int = 3,
    run_dir: str = "",
    obs_ref_max: float = None,
) -> dict:
    """
    Run deterministic evaluation episodes with fixed seeds and produce
    matplotlib summary figures for TensorBoard.

    Uses a vectorized SyncVectorEnv to run all episodes in parallel:
    3 per seed (zero-action, random-action, agent) x num_episodes seeds.
    """
    agent.eval()
    wrapper = PPOActorWrapper(agent)

    env_kwargs = dict(config["env_kwargs"])
    env_kwargs["max_episode_steps"] = config["max_episode_steps"]
    max_steps = config["max_episode_steps"]

    eval_seeds = config.get("eval_seeds")
    if eval_seeds is None:
        eval_seeds = list(range(num_episodes))
    seeds_to_use = eval_seeds[:num_episodes]
    N = len(seeds_to_use)
    total_envs = 3 * N  # zero + random + agent per seed

    # Index helpers: env[i*3+0]=zero, env[i*3+1]=random, env[i*3+2]=agent
    zero_idx = list(range(0, total_envs, 3))
    rand_idx = list(range(1, total_envs, 3))
    agnt_idx = list(range(2, total_envs, 3))

    # --- Create vectorized eval env --------------------------------
    eval_envs = gym.vector.SyncVectorEnv([
        make_optomech_env(env_kwargs, max_episode_steps=max_steps, idx=i)
        for i in range(total_envs)
    ])
    action_space = eval_envs.single_action_space
    action_dim = action_space.shape[0]

    # Extract reference max for fixed normalization if not provided
    if obs_ref_max is None:
        _tmp = gym.make(_ENV_ID, **env_kwargs)
        _b = _tmp.unwrapped
        if hasattr(_b, 'optical_system') and hasattr(_b.optical_system, '_reference_fpi_max'):
            obs_ref_max = _b.optical_system._reference_fpi_max
        else:
            obs_ref_max = 1.0
        _tmp.close()

    # --- Reset all envs with per-env seeds -------------------------
    reset_seeds = []
    for seed in seeds_to_use:
        reset_seeds.extend([seed, seed, seed])
    obs_all, _ = eval_envs.reset(seed=reset_seeds)

    # --- Initialize agent state (batched across N seeds) -----------
    h = torch.zeros(agent.lstm_num_layers, N, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(agent.lstm_num_layers, N, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(N, action_dim, device=device)
    prior_reward = torch.zeros(N, device=device)

    # --- Pre-allocate per-step storage -----------------------------
    all_step_rewards = np.zeros((max_steps, total_envs), dtype=np.float32)
    agnt_step_actions = np.zeros((max_steps, N, action_dim), dtype=np.float32)
    agnt_step_strehls = np.zeros((max_steps, N), dtype=np.float32)
    agnt_step_mses = np.zeros((max_steps, N), dtype=np.float32)
    agnt_step_oobs = np.zeros((max_steps, N), dtype=np.float32)
    agnt_obs_raw = [[obs_all[agnt_idx[i]].copy()] for i in range(N)]

    # --- Main stepping loop ----------------------------------------
    for step in range(max_steps):
        actions = np.zeros((total_envs, action_dim), dtype=np.float32)
        # Zero-action slots: already zeros
        # Random-action slots
        for idx in rand_idx:
            actions[idx] = action_space.sample()
        # Agent slots: batched inference
        agent_obs = obs_all[agnt_idx]
        obs_norm = normalize_obs_fixed(agent_obs, obs_ref_max)
        obs_t = torch.from_numpy(obs_norm).float().to(device)
        with torch.no_grad():
            action_t, (h, c) = wrapper(obs_t, prior_action, prior_reward, (h, c))
        act_np = action_t.cpu().numpy()
        for i, idx in enumerate(agnt_idx):
            actions[idx] = act_np[i]

        obs_all, rewards, terms, truncs, infos = eval_envs.step(actions)

        all_step_rewards[step] = rewards
        agnt_step_actions[step] = act_np
        for i in range(N):
            agnt_obs_raw[i].append(obs_all[agnt_idx[i]].copy())

        if isinstance(infos, dict):
            _s = infos.get("strehl", None)
            _m = infos.get("mse", None)
            _o = infos.get("oob_frac", None)
            for i, idx in enumerate(agnt_idx):
                if _s is not None:
                    agnt_step_strehls[step, i] = float(_s[idx])
                if _m is not None:
                    agnt_step_mses[step, i] = float(_m[idx])
                if _o is not None:
                    agnt_step_oobs[step, i] = float(_o[idx])

        prior_action = action_t.clone()
        prior_reward = torch.from_numpy(rewards[agnt_idx]).float().to(device)

    eval_envs.close()

    # --- Reassemble per-seed data ----------------------------------
    all_returns = []
    all_lengths = []
    all_zero_returns = []
    all_improvement_gaps = []
    all_relative_improvements = []
    all_first_rewards = []
    all_last_rewards = []
    all_final_strehls = []
    all_final_mses = []
    all_agent_reward_traces = []
    all_zero_reward_traces = []
    all_random_reward_traces = []
    all_agent_strehl_traces = []
    all_agent_mse_traces = []
    all_episode_data = []

    for i, seed in enumerate(seeds_to_use):
        zero_rewards = all_step_rewards[:, zero_idx[i]].tolist()
        random_rewards = all_step_rewards[:, rand_idx[i]].tolist()
        ep_rewards = all_step_rewards[:, agnt_idx[i]].tolist()
        ep_strehls = agnt_step_strehls[:, i].tolist()
        ep_mses = agnt_step_mses[:, i].tolist()
        ep_oob_fracs = agnt_step_oobs[:, i].tolist()
        ep_actions = [agnt_step_actions[t, i].copy() for t in range(max_steps)]

        zero_return = sum(zero_rewards)
        ep_return = sum(ep_rewards)
        ep_length = max_steps

        all_zero_returns.append(zero_return)
        all_zero_reward_traces.append(zero_rewards)
        all_random_reward_traces.append(random_rewards)
        all_returns.append(ep_return)
        all_lengths.append(ep_length)
        all_agent_reward_traces.append(ep_rewards)
        all_agent_strehl_traces.append(ep_strehls)
        all_agent_mse_traces.append(ep_mses)
        all_final_strehls.append(ep_strehls[-1] if ep_strehls else 0.0)
        all_final_mses.append(ep_mses[-1] if ep_mses else 0.0)

        improvement_gap = ep_return - zero_return
        all_improvement_gaps.append(improvement_gap)
        if abs(zero_return) > 1e-10:
            relative_improvement = ep_return / zero_return
        else:
            relative_improvement = 1.0
        all_relative_improvements.append(relative_improvement)

        first_reward = ep_rewards[0] if ep_rewards else 0.0
        last_reward = ep_rewards[-1] if ep_rewards else 0.0
        all_first_rewards.append(first_reward)
        all_last_rewards.append(last_reward)

        writer.add_scalar(f"{tag_prefix}/seed_{seed}/agent_return", ep_return, global_step)
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/zero_return", zero_return, global_step)
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/improvement_gap", improvement_gap, global_step)
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/first_reward", first_reward, global_step)
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/last_reward", last_reward, global_step)
        final_strehl = ep_strehls[-1] if ep_strehls else 0.0
        final_mse = ep_mses[-1] if ep_mses else 0.0
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/final_strehl", final_strehl, global_step)
        writer.add_scalar(f"{tag_prefix}/seed_{seed}/final_mse", final_mse, global_step)

        all_episode_data.append({
            "rewards": ep_rewards,
            "actions": ep_actions,
            "obs_raw": agnt_obs_raw[i],
            "strehls": ep_strehls,
            "mses": ep_mses,
            "oob_fracs": ep_oob_fracs,
            "return": ep_return,
            "length": ep_length,
            "seed": seed,
            "zero_return": zero_return,
            "improvement_gap": improvement_gap,
        })

    # Select best / worst / median episodes by return
    all_episode_data.sort(key=lambda d: d["return"])
    worst_episode_data = all_episode_data[0]
    median_episode_data = all_episode_data[len(all_episode_data) // 2]
    best_episode_data = all_episode_data[-1]
    best_return = best_episode_data["return"]

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    mean_return = float(np.mean(all_returns))
    mean_zero_return = float(np.mean(all_zero_returns))
    mean_gap = float(np.mean(all_improvement_gaps))
    mean_first = float(np.mean(all_first_rewards))
    mean_last = float(np.mean(all_last_rewards))

    # Relative improvement: last_reward / first_reward (item #2)
    if abs(mean_first) > 1e-10:
        mean_reward_ratio = mean_last / mean_first
    else:
        mean_reward_ratio = 1.0

    mean_final_strehl = float(np.mean(all_final_strehls)) if all_final_strehls else 0.0
    mean_final_mse = float(np.mean(all_final_mses)) if all_final_mses else 0.0

    metrics = {
        "mean_return": mean_return,
        "std_return": float(np.std(all_returns)),
        "mean_length": float(np.mean(all_lengths)),
        "best_return": best_return,
        "mean_zero_return": mean_zero_return,
        "mean_improvement_gap": mean_gap,
        "mean_first_reward": mean_first,
        "mean_last_reward": mean_last,
        "mean_reward_ratio": mean_reward_ratio,
        "mean_final_strehl": mean_final_strehl,
        "mean_final_mse": mean_final_mse,
    }

    # TensorBoard: aggregate scalars
    writer.add_scalar(f"{tag_prefix}/mean_return", metrics["mean_return"], global_step)
    writer.add_scalar(f"{tag_prefix}/std_return", metrics["std_return"], global_step)
    writer.add_scalar(f"{tag_prefix}/best_return", metrics["best_return"], global_step)
    writer.add_scalar(
        f"{tag_prefix}/mean_zero_return", mean_zero_return, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_improvement_gap", mean_gap, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_first_reward", mean_first, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_last_reward", mean_last, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_reward_ratio", mean_reward_ratio, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_final_strehl", mean_final_strehl, global_step
    )
    writer.add_scalar(
        f"{tag_prefix}/mean_final_mse", mean_final_mse, global_step
    )

    if best_episode_data is not None:
        _log_rollout_summary_figure(writer, best_episode_data, global_step, tag_prefix)
        _log_observation_filmstrip(writer, best_episode_data, global_step, tag_prefix)

    # Aggregate reward traces across all eval episodes
    _log_aggregate_reward_figure(
        writer,
        all_agent_reward_traces,
        all_zero_reward_traces,
        all_random_reward_traces,
        global_step,
        tag_prefix,
    )

    # Generate GIFs for best / worst / median episodes
    if run_dir:
        gif_dir = os.path.join(run_dir, "gifs", f"step_{global_step}")
        os.makedirs(gif_dir, exist_ok=True)
        for ep, label in [
            (best_episode_data, "best"),
            (worst_episode_data, "worst"),
            (median_episode_data, "median"),
        ]:
            _render_episode_gif(
                ep,
                os.path.join(gif_dir, f"{label}.gif"),
                label.upper(),
                global_step,
            )
        print(f"  GIFs saved to {gif_dir}")

    agent.train()
    return metrics


# ---------------------------------------------------------------------------
# Matplotlib figure builders
# ---------------------------------------------------------------------------


def _log_rollout_summary_figure(
    writer: SummaryWriter,
    ep_data: dict,
    global_step: int,
    tag_prefix: str,
):
    """
    2x2 summary:
      Top-left:     Step-wise reward
      Top-right:    Cumulative return
      Bottom-left:  Action trace (one line per DOF)
      Bottom-right: Strehl & signal diagnostics
    """
    rewards = np.array(ep_data["rewards"])
    actions = np.array(ep_data["actions"])  # [T, action_dim]
    obs_raw = ep_data["obs_raw"]

    cumulative = np.cumsum(rewards)
    timesteps = np.arange(len(rewards))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Top-left: step-wise reward ---
    ax_rew = fig.add_subplot(gs[0, 0])
    ax_rew.plot(timesteps, rewards, color="steelblue", linewidth=1.0, alpha=0.8)
    if len(rewards) > 10:
        kernel = np.ones(10) / 10
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax_rew.plot(
            timesteps[4 : 4 + len(smoothed)],
            smoothed,
            color="darkblue",
            linewidth=2.0,
            label="smoothed (10-step)",
        )
        ax_rew.legend(fontsize=8)
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Step-wise Reward")
    ax_rew.grid(True, alpha=0.3)

    # --- Top-right: cumulative return ---
    ax_cum = fig.add_subplot(gs[0, 1])
    ax_cum.plot(timesteps, cumulative, color="forestgreen", linewidth=2.0,
                label=f"agent = {cumulative[-1]:.1f}")
    # Zero-action baseline (dashed) for comparison
    if "zero_return" in ep_data:
        zero_per_step = ep_data["zero_return"] / len(rewards)
        zero_cumulative = np.cumsum(np.full_like(rewards, zero_per_step))
        ax_cum.plot(timesteps, zero_cumulative, color="gray", linewidth=1.5,
                    linestyle="--", alpha=0.7,
                    label=f"zero-action = {ep_data['zero_return']:.1f}")
    ax_cum.set_xlabel("Step")
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.set_title("Cumulative Return vs Zero-Action Baseline")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(True, alpha=0.3)

    # --- Bottom-left: action trace ---
    ax_act = fig.add_subplot(gs[1, 0])
    action_dim = actions.shape[1] if actions.ndim > 1 else 1
    if actions.ndim == 1:
        actions = actions[:, np.newaxis]
    labels = [f"a[{d}]" for d in range(action_dim)]
    for d in range(action_dim):
        ax_act.plot(
            timesteps, actions[:, d], linewidth=1.0, label=labels[d], alpha=0.8
        )
    ax_act.set_xlabel("Step")
    ax_act.set_ylabel("Action")
    ax_act.set_title("Commands")
    ax_act.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    if action_dim <= 6:
        ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    # --- Bottom-right: Strehl + signal diagnostics ---
    ax_br = fig.add_subplot(gs[1, 1])
    strehls_arr = np.array(ep_data.get("strehls", []))
    oob_fracs = np.array(ep_data.get("oob_fracs", []))

    if len(strehls_arr) > 0:
        # Strehl trace (primary y-axis)
        ax_br.plot(timesteps, strehls_arr, color="darkorange", linewidth=1.5,
                   label="Strehl")
        ax_br.set_ylabel("Strehl", color="darkorange")
        ax_br.set_ylim(0, max(1.05, float(np.max(strehls_arr)) * 1.05))
        ax_br.tick_params(axis="y", labelcolor="darkorange")
        ax_br.set_xlabel("Step")
        ax_br.set_title("Strehl & Signal Level")
        ax_br.grid(True, alpha=0.3)

        # Sum-DN trace on secondary axis to show flux level
        obs_raw = ep_data["obs_raw"]
        sum_dns = [float(np.sum(_prepare_obs_raw(obs_raw[t + 1])))
                   for t in range(len(timesteps))]
        ax_flux = ax_br.twinx()
        ax_flux.plot(timesteps, sum_dns, color="steelblue", linewidth=1.0,
                     alpha=0.7, linestyle="--", label="sum DN")
        ax_flux.set_ylabel("sum DN", color="steelblue", fontsize=8)
        ax_flux.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

        # Combined legend
        lines1, labels1 = ax_br.get_legend_handles_labels()
        lines2, labels2 = ax_flux.get_legend_handles_labels()
        ax_br.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
    elif len(oob_fracs) > 0:
        oob_colors = ["#d94a4a" if f > 0 else "#4a90d9" for f in oob_fracs]
        ax_br.bar(timesteps, oob_fracs, color=oob_colors, width=1.0, alpha=0.8)
        ax_br.set_xlabel("Step")
        ax_br.set_ylabel("OOB Fraction")
        total_oob_steps = int(np.sum(oob_fracs > 0))
        ax_br.set_title(
            f"OOB ({total_oob_steps}/{len(oob_fracs)} steps)", fontsize=9)
        ax_br.set_ylim(0, max(1.0, float(np.max(oob_fracs)) * 1.1))
        ax_br.grid(True, alpha=0.3)
    else:
        ax_br.text(0.5, 0.5, "No diagnostic data", ha="center", va="center",
                   fontsize=10, transform=ax_br.transAxes)
        ax_br.set_title("Diagnostics")

    # Build title with zero-action baseline comparison if available
    title_parts = [
        f"Return: {ep_data['return']:.3f}",
        f"Length: {ep_data['length']}",
    ]
    if "zero_return" in ep_data:
        title_parts.append(f"Zero-baseline: {ep_data['zero_return']:.3f}")
        title_parts.append(f"Gap: {ep_data['improvement_gap']:+.3f}")
    if "seed" in ep_data:
        title_parts.append(f"Seed: {ep_data['seed']}")
    title_parts.append(f"Step: {global_step}")

    fig.suptitle(
        "Rollout  |  " + "  |  ".join(title_parts),
        fontsize=11,
        fontweight="bold",
    )

    writer.add_figure(f"{tag_prefix}/rollout_summary", fig, global_step)
    plt.close(fig)


def _log_observation_filmstrip(
    writer: SummaryWriter,
    ep_data: dict,
    global_step: int,
    tag_prefix: str,
    num_frames: int = 8,
):
    """
    Filmstrip of focal-plane observations sampled uniformly from the episode.
    Shows raw DN values with log-scale colorbars for signal-level visibility.
    """
    obs_raw = ep_data["obs_raw"]
    rewards = np.array(ep_data["rewards"])
    strehls = ep_data.get("strehls", [])
    cumulative = np.cumsum(rewards)
    T = len(obs_raw)

    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        frame_indices = [
            int(i * (T - 1) / (num_frames - 1)) for i in range(num_frames)
        ]

    # Pre-extract raw images and find global max for shared colour scale.
    raw_imgs = [_prepare_obs_raw(obs_raw[idx]) for idx in frame_indices]
    global_max = max(float(np.max(img)) for img in raw_imgs)
    global_max = max(global_max, 1.0)  # guard against all-zero frames

    n = len(frame_indices)
    fig, axes = plt.subplots(
        2, n, figsize=(3.0 * n, 6.0), gridspec_kw={"height_ratios": [3, 1]}
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    norm = mcolors.LogNorm(vmin=1.0, vmax=global_max)

    for col, idx in enumerate(frame_indices):
        ax_img = axes[0, col]
        img_dn = raw_imgs[col]
        # Clip floor to 1 DN for log display (zeros -> 1)
        im = ax_img.imshow(np.maximum(img_dn, 1.0), cmap="inferno", norm=norm)
        ax_img.set_title(f"t={idx}", fontsize=9, fontweight="bold")
        ax_img.axis("off")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        ax_txt = axes[1, col]
        ax_txt.axis("off")

        dn_max = float(np.max(img_dn))
        dn_sum = float(np.sum(img_dn))
        if idx == 0:
            text = f"start\nmax={dn_max:.0f}\nsum={dn_sum:.0f}"
        else:
            r = rewards[idx - 1]
            c = cumulative[idx - 1]
            s_txt = f"\nS={strehls[idx-1]:.3f}" if strehls else ""
            text = (f"r={r:.4f}  \u03a3r={c:.3f}{s_txt}"
                    f"\nmax={dn_max:.0f}  sum={dn_sum:.0f}")
        ax_txt.text(
            0.5,
            0.5,
            text,
            transform=ax_txt.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            fontfamily="monospace",
        )

    fig.suptitle(
        f"Focal-Plane Filmstrip (DN, log scale)  |  "
        f"Return: {ep_data['return']:.3f}  |  Step: {global_step}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    writer.add_figure(f"{tag_prefix}/observation_filmstrip", fig, global_step)
    plt.close(fig)


def _render_episode_gif(ep_data, save_path, label, global_step,
                        dpi=72, frame_duration=0.2):
    """Render an evaluation episode as an animated GIF.

    Each frame shows the focal-plane observation in raw DN with a log-scale
    colorbar (top), action bar chart (bottom-left), and step metrics
    including signal-level diagnostics (bottom-right).
    """
    obs_raw = ep_data["obs_raw"]
    rewards = ep_data["rewards"]
    actions = np.array(ep_data["actions"])  # [T, action_dim]
    strehls = ep_data.get("strehls", [])
    cumulative = np.cumsum(rewards)
    T = len(rewards)
    action_dim = actions.shape[1] if len(actions) > 0 else 0

    # Global colour scale across the whole episode for consistency.
    all_raw = [_prepare_obs_raw(obs_raw[t]) for t in range(T + 1)]
    global_max = max(float(np.max(img)) for img in all_raw)
    global_max = max(global_max, 1.0)
    norm = mcolors.LogNorm(vmin=1.0, vmax=global_max)

    frames = []
    for t in range(T + 1):  # T+1 because obs_raw includes initial obs
        fig = plt.figure(figsize=(5, 5.5), dpi=dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[3, 1], hspace=0.35, wspace=0.4)

        # Top: observation image spanning both columns (raw DN, log scale)
        ax_obs = fig.add_subplot(gs[0, :])
        img_dn = all_raw[t]
        im = ax_obs.imshow(np.maximum(img_dn, 1.0), cmap="inferno", norm=norm)
        ax_obs.axis("off")
        dn_max = float(np.max(img_dn))
        dn_sum = float(np.sum(img_dn))
        ax_obs.set_title(
            f"{label} | seed={ep_data['seed']} | R={ep_data['return']:.3f}"
            f" | step {global_step}"
            f"\nmax={dn_max:.0f}  sum={dn_sum:.0f}",
            fontsize=7)
        fig.colorbar(im, ax=ax_obs, fraction=0.046, pad=0.04,
                     label="DN")

        # Bottom-left: action bars
        ax_act = fig.add_subplot(gs[1, 0])
        if t > 0 and action_dim > 0:
            act = actions[t - 1]
            colors = (["#4a90d9", "#d94a4a", "#4ad94a",
                       "#d9d94a", "#d94ad9", "#4ad9d9"] * 2)[:action_dim]
            ax_act.barh(range(action_dim), act, color=colors)
            ax_act.set_xlim(-1.1, 1.1)
            ax_act.set_yticks(range(action_dim))
            ax_act.set_yticklabels([f"a[{i}]" for i in range(action_dim)],
                                   fontsize=6)
            ax_act.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
            for i, v in enumerate(act):
                ax_act.text(v, i, f" {v:.2f}", va="center", fontsize=5)
        else:
            ax_act.text(0.5, 0.5, "t=0\n(initial)", ha="center",
                        va="center", fontsize=7, transform=ax_act.transAxes)
            ax_act.set_xticks([])
            ax_act.set_yticks([])
        ax_act.set_title("Action", fontsize=7)

        # Bottom-right: text metrics
        ax_txt = fig.add_subplot(gs[1, 1])
        ax_txt.axis("off")
        if t == 0:
            text = (f"t = 0 / {T}\n(initial obs)"
                    f"\nmax DN = {dn_max:.0f}"
                    f"\nsum DN = {dn_sum:.0f}")
        else:
            r = rewards[t - 1]
            c = cumulative[t - 1]
            s_txt = f"\nS = {strehls[t-1]:.4f}" if strehls else ""
            text = (f"t = {t} / {T}\nr = {r:.4f}  R = {c:.3f}{s_txt}"
                    f"\nmax DN = {dn_max:.0f}"
                    f"\nsum DN = {dn_sum:.0f}")
        ax_txt.text(0.5, 0.5, text, ha="center", va="center",
                    fontsize=7, fontfamily="monospace",
                    transform=ax_txt.transAxes)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=frame_duration)


def _log_aggregate_reward_figure(
    writer: SummaryWriter,
    agent_traces: list,
    zero_traces: list,
    random_traces: list,
    global_step: int,
    tag_prefix: str,
):
    """
    1x2 aggregate figure across all eval episodes:
      Left:  Mean +/- std step-wise reward (agent vs zero vs random)
      Right: Mean +/- std cumulative return (agent vs zero vs random)

    Each trace list is a list of lists (one per episode). Episodes may
    differ in length, so we pad shorter traces with their last value
    before stacking.
    """
    if not agent_traces:
        return

    def _pad_and_stack(traces):
        """Pad variable-length traces to uniform length and stack."""
        max_len = max(len(t) for t in traces)
        padded = np.zeros((len(traces), max_len), dtype=np.float32)
        for i, t in enumerate(traces):
            arr = np.array(t, dtype=np.float32)
            padded[i, : len(arr)] = arr
            if len(arr) < max_len:
                padded[i, len(arr) :] = arr[-1]  # hold last value
        return padded

    agent_mat = _pad_and_stack(agent_traces)   # [N, T]
    zero_mat = _pad_and_stack(zero_traces)     # [N, T]
    random_mat = _pad_and_stack(random_traces) # [N, T]

    T = agent_mat.shape[1]
    timesteps = np.arange(T)

    fig, (ax_rew, ax_cum) = plt.subplots(1, 2, figsize=(16, 5))

    # --- Left: step-wise reward mean +/- std ---
    def _plot_mean_std(ax, mat, color, label, alpha_fill=0.15):
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        ax.plot(timesteps, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(timesteps, mean - std, mean + std,
                        color=color, alpha=alpha_fill)

    _plot_mean_std(ax_rew, agent_mat, "forestgreen", "agent")
    _plot_mean_std(ax_rew, zero_mat, "gray", "zero-action")
    _plot_mean_std(ax_rew, random_mat, "coral", "random-action")

    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Step-wise Reward (mean \u00b1 std)")
    ax_rew.legend(fontsize=9)
    ax_rew.grid(True, alpha=0.3)

    # --- Right: cumulative return mean +/- std ---
    agent_cum = np.cumsum(agent_mat, axis=1)   # [N, T]
    zero_cum = np.cumsum(zero_mat, axis=1)     # [N, T]
    random_cum = np.cumsum(random_mat, axis=1) # [N, T]

    _plot_mean_std(ax_cum, agent_cum, "forestgreen", "agent")
    _plot_mean_std(ax_cum, zero_cum, "gray", "zero-action")
    _plot_mean_std(ax_cum, random_cum, "coral", "random-action")

    ax_cum.set_xlabel("Step")
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.set_title("Cumulative Return (mean \u00b1 std)")
    ax_cum.legend(fontsize=9)
    ax_cum.grid(True, alpha=0.3)

    n_ep = len(agent_traces)
    fig.suptitle(
        f"Aggregate Eval ({n_ep} episodes)  |  Step: {global_step}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    writer.add_figure(f"{tag_prefix}/aggregate_reward", fig, global_step)
    plt.close(fig)


# ============================================================================
# Baseline policies
# ============================================================================


def evaluate_zero_policy(config: dict, num_episodes: int = 5) -> float:
    """Roll out a zero-action policy and return mean episodic return."""
    env_kwargs = dict(config["env_kwargs"])
    env_kwargs["max_episode_steps"] = config["max_episode_steps"]
    env = gym.make(_ENV_ID, **env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def evaluate_random_policy(config: dict, num_episodes: int = 5) -> float:
    """Roll out a uniform-random policy and return mean episodic return."""
    env_kwargs = dict(config["env_kwargs"])
    env_kwargs["max_episode_steps"] = config["max_episode_steps"]
    env = gym.make(_ENV_ID, **env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


# ============================================================================
# Main training loop
# ============================================================================


def run_ppo_training(config: dict, run_dir: str):
    """
    Trains a recurrent PPO agent on an optomech environment and returns
    (best_eval_return, tensorboard_log_dir).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("cpu")  # MPS has LSTM issues; fall back to CPU
    else:
        device = torch.device("cpu")

    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Vectorized environments
    # Default: SyncVectorEnv (lower overhead for moderate env counts).
    # Use --async-envs to enable AsyncVectorEnv for true multiprocess parallelism.
    env_fns = [
        make_optomech_env(
            config["env_kwargs"],
            max_episode_steps=config["max_episode_steps"],
            idx=i,
        )
        for i in range(config["num_envs"])
    ]
    use_async = config.get("async_envs", False)
    VecEnvCls = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
    print(f"  Creating {config['num_envs']} envs with {VecEnvCls.__name__}")
    envs = VecEnvCls(env_fns)

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    # Extract reference max for fixed-reference normalization.
    # This preserves flux information when the PSF moves off the detector.
    # Create a temporary single env to read the reference value, since
    # AsyncVectorEnv does not expose .envs directly.
    _tmp_env = gym.make(
        _ENV_ID,
        **config["env_kwargs"],
    )
    _base_env = _tmp_env.unwrapped
    if hasattr(_base_env, 'optical_system') and hasattr(_base_env.optical_system, '_reference_fpi_max'):
        obs_ref_max = _base_env.optical_system._reference_fpi_max
    else:
        obs_ref_max = 1.0  # fallback
    _tmp_env.close()
    del _tmp_env, _base_env
    print(f"  Obs ref max (DN):  {obs_ref_max:.1f}")

    # --- Load pretrained encoder if provided ---
    pretrained_encoder = None
    freeze_encoder = config.get("freeze_encoder", False)
    pretrained_path = config.get("pretrained_encoder")
    if pretrained_path:
        print(f"  Loading pretrained encoder from: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        ae_cfg = ckpt["config"]
        cs = ae_cfg["channel_scale"]
        in_ch = ae_cfg["input_channels"]
        pretrained_encoder = nn.Sequential(
            conv_init(nn.Conv2d(in_ch, cs, kernel_size=8, stride=4)),
            nn.ReLU(),
            conv_init(nn.Conv2d(cs, cs * 2, kernel_size=4, stride=2)),
            nn.ReLU(),
            conv_init(nn.Conv2d(cs * 2, cs * 4, kernel_size=2, stride=2)),
            nn.ReLU(),
        )
        pretrained_encoder.load_state_dict(ckpt["encoder_state_dict"])
        mode = "frozen" if freeze_encoder else "trainable"
        print(f"  Pretrained encoder loaded ({mode}, cs={cs}, in_ch={in_ch})")

    agent = RecurrentActorCritic(
        envs,
        device,
        encoder=pretrained_encoder,
        lstm_hidden_dim=config["lstm_hidden_dim"],
        channel_scale=config["channel_scale"],
        fc_scale=config["fc_scale"],
        action_scale=config["action_scale"],
        init_log_std=config["init_log_std"],
        freeze_encoder=freeze_encoder,
    ).to(device)

    # Load bottleneck weights into the MLP layer if available
    if pretrained_path and "bottleneck_encode_state_dict" in ckpt:
        agent.mlp.load_state_dict(ckpt["bottleneck_encode_state_dict"])
        print("  Loaded bottleneck weights into agent MLP")

    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    # --- Resume from full PPO checkpoint if provided ---
    resume_step = 0
    resume_path = config.get("resume_from")
    init_path = config.get("init_from")
    if resume_path:
        print(f"  Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_step = ckpt.get("global_step", 0)
        resume_update = ckpt.get("update", 0)
        resume_eval = ckpt.get("best_eval_return", -float("inf"))
        print(f"  Resumed: global_step={resume_step:,}, update={resume_update}, "
              f"best_eval_return={resume_eval:.4f}")
    elif init_path:
        print(f"  Initialising weights from: {init_path}")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        src_step = ckpt.get("global_step", "?")
        src_eval = ckpt.get("best_eval_return", "?")
        print(f"  Loaded weights (source: step={src_step}, eval={src_eval}). "
              f"Training from scratch with fresh optimizer.")

    num_envs = config["num_envs"]
    num_steps = config["num_steps"]
    batch_size = num_envs * num_steps
    num_minibatches = config["num_minibatches"]
    # Guard: total chunks must be >= num_minibatches to avoid zero chunk_size
    seq_len = config["seq_len"]
    total_chunks = (num_steps // seq_len) * num_envs
    if num_minibatches > total_chunks:
        print(f"  WARNING: num_minibatches ({num_minibatches}) > total_chunks "
              f"({total_chunks} = {num_steps}//{ seq_len} * {num_envs}). "
              f"Clamping to {total_chunks}.")
        num_minibatches = max(1, total_chunks)
        config["num_minibatches"] = num_minibatches
    minibatch_size = batch_size // num_minibatches
    num_updates = config["total_timesteps"] // batch_size
    reward_scale = config.get("reward_scale", 1.0)
    save_interval = config.get("model_save_interval", 0)

    run_name = f"ppo_optomech_{seed}_{int(time.time())}"
    this_run_dir = os.path.join(run_dir, run_name)
    os.makedirs(this_run_dir, exist_ok=True)

    log_dir = os.path.join(this_run_dir, "tensorboard")
    writer = SummaryWriter(log_dir)
    writer.add_text("config", json.dumps(config, indent=2, default=str))

    ckpt_dir = os.path.join(this_run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PPO Optomech Training")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Obs shape:        {obs_shape}")
    print(f"  Action shape:     {action_shape}")
    print(f"  Total timesteps:  {config['total_timesteps']:,}")
    print(f"  Num updates:      {num_updates}")
    print(f"  Batch size:       {batch_size} ({num_envs} envs x {num_steps} steps)")
    print(f"  Minibatch size:   {minibatch_size}")
    print(f"  Reward scale:     {reward_scale}")
    print(f"  Save interval:    {save_interval} updates" if save_interval > 0 else "  Save interval:    disabled")
    print(f"  Parameters:       {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  Run directory:    {this_run_dir}")
    print(f"  TensorBoard:      {log_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Pre-allocate rollout buffers
    # ------------------------------------------------------------------
    obs_buf = torch.zeros(num_steps, num_envs, *obs_shape)
    action_buf = torch.zeros(num_steps, num_envs, *action_shape)
    logprob_buf = torch.zeros(num_steps, num_envs)
    reward_buf = torch.zeros(num_steps, num_envs)
    done_buf = torch.zeros(num_steps, num_envs)
    value_buf = torch.zeros(num_steps, num_envs)
    prior_action_buf = torch.zeros(num_steps, num_envs, *action_shape)
    prior_reward_buf = torch.zeros(num_steps, num_envs)
    hidden_h_buf = torch.zeros(
        num_steps, num_envs, agent.lstm_num_layers, config["lstm_hidden_dim"]
    )
    hidden_c_buf = torch.zeros(
        num_steps, num_envs, agent.lstm_num_layers, config["lstm_hidden_dim"]
    )
    episode_start_buf = torch.zeros(num_steps, num_envs)

    # ------------------------------------------------------------------
    # Initialise environment state
    # ------------------------------------------------------------------
    obs_np, _ = envs.reset(seed=seed)
    obs_np = normalize_obs_fixed(obs_np, obs_ref_max)
    next_obs = torch.from_numpy(obs_np).float()
    next_done = torch.zeros(num_envs)

    lstm_h = torch.zeros(
        agent.lstm_num_layers, num_envs, config["lstm_hidden_dim"], device=device
    )
    lstm_c = torch.zeros(
        agent.lstm_num_layers, num_envs, config["lstm_hidden_dim"], device=device
    )

    running_prior_action = torch.zeros(num_envs, *action_shape)
    running_prior_reward = torch.zeros(num_envs)
    running_episode_start = torch.ones(num_envs)

    best_eval_return = -np.inf
    global_step = 0
    start_update = 1
    start_time = time.time()
    recent_train_returns = []

    # Apply resume offsets
    if resume_step > 0:
        global_step = resume_step
        start_update = resume_step // batch_size + 1
        best_eval_return = resume_eval
        print(f"  Resuming training from update {start_update}, "
              f"global_step {global_step:,}")

    # --- Curriculum: anneal tip/tilt init error from easy to hard ---
    curriculum_cfg = config.get("curriculum")
    if curriculum_cfg:
        _cur_start = curriculum_cfg["tip_tilt_start"]
        _cur_end = curriculum_cfg["tip_tilt_end"]
        _cur_steps = curriculum_cfg["anneal_timesteps"]
        print(f"  Curriculum: tip/tilt {_cur_start:.2f} → {_cur_end:.2f} "
              f"over {_cur_steps:,} steps")

    for update in range(start_update, num_updates + 1):
        # LR annealing
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = config["learning_rate"] * frac

        # Curriculum annealing: linearly interpolate tip/tilt init error
        if curriculum_cfg:
            progress = min(global_step / _cur_steps, 1.0)
            cur_std = _cur_start + progress * (_cur_end - _cur_start)
            if hasattr(envs, 'envs'):
                # SyncVectorEnv: direct access
                for env_wrapper in envs.envs:
                    base_env = env_wrapper.unwrapped
                    base_env.cfg["init_wind_tip_arcsec_std_tt"] = cur_std
                    base_env.cfg["init_wind_tilt_arcsec_std_tt"] = cur_std
            else:
                # AsyncVectorEnv: not yet supported for curriculum.
                # Curriculum requires num_envs=1 or SyncVectorEnv.
                if update == 1:
                    print("  WARNING: Curriculum annealing is not supported "
                          "with AsyncVectorEnv. Ignoring curriculum config.")
            if update % 100 == 1:
                writer.add_scalar("curriculum/tip_tilt_std", cur_std, global_step)

        # ==============================================================
        # ROLLOUT PHASE
        # ==============================================================
        agent.eval()
        update_start_time = time.time()
        rollout_start_time = time.time()
        for step in range(num_steps):
            step_start_time = time.time()
            global_step += num_envs

            obs_buf[step] = next_obs
            done_buf[step] = next_done
            prior_action_buf[step] = running_prior_action
            prior_reward_buf[step] = running_prior_reward
            episode_start_buf[step] = running_episode_start
            hidden_h_buf[step] = lstm_h.permute(1, 0, 2).cpu()
            hidden_c_buf[step] = lstm_c.permute(1, 0, 2).cpu()

            with torch.no_grad():
                raw_action, logprob, _, value, (lstm_h, lstm_c) = (
                    agent.get_action_and_value(
                        next_obs.to(device),
                        running_prior_action.to(device),
                        running_prior_reward.to(device),
                        (lstm_h, lstm_c),
                    )
                )
                env_action = agent.scale_and_clamp_action(raw_action)

            action_buf[step] = raw_action.cpu()
            logprob_buf[step] = logprob.cpu()
            value_buf[step] = value.flatten().cpu()

            next_obs_np, rewards_np, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            next_obs_np = normalize_obs_fixed(next_obs_np, obs_ref_max)
            rewards_np = reward_scale * rewards_np

            next_obs = torch.from_numpy(next_obs_np).float()
            reward_buf[step] = torch.from_numpy(rewards_np.astype(np.float32))

            dones_step = np.logical_or(terminations, truncations)
            next_done = torch.from_numpy(dones_step.astype(np.float32))

            running_prior_action = env_action.cpu().clone()
            running_prior_reward = torch.from_numpy(rewards_np.astype(np.float32))
            running_episode_start = next_done.clone()

            for env_idx in range(num_envs):
                if dones_step[env_idx]:
                    lstm_h[:, env_idx, :] = 0.0
                    lstm_c[:, env_idx, :] = 0.0
                    running_prior_action[env_idx] = 0.0
                    running_prior_reward[env_idx] = 0.0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None:
                        continue
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    recent_train_returns.append(ep_return)
                    writer.add_scalar(
                        "train/episodic_return", ep_return, global_step
                    )
                    writer.add_scalar(
                        "train/episodic_length", ep_length, global_step
                    )

            # Per-step instantaneous SPS
            step_dt = time.time() - step_start_time
            step_sps = num_envs / step_dt if step_dt > 0 else 0
            writer.add_scalar("performance/step_SPS", step_sps, global_step)

            # Per-step diagnostic metrics from env info dict
            if "strehl" in infos:
                for env_idx in range(num_envs):
                    writer.add_scalar(
                        "train/step_strehl",
                        float(infos["strehl"][env_idx]), global_step,
                    )
                    writer.add_scalar(
                        "train/step_mse",
                        float(infos["mse"][env_idx]), global_step,
                    )
                    writer.add_scalar(
                        "train/step_oob_frac",
                        float(infos["oob_frac"][env_idx]), global_step,
                    )
                    writer.add_scalar(
                        "train/step_reward",
                        float(rewards_np[env_idx]), global_step,
                    )
                    writer.add_scalar(
                        "train/step_reward_raw",
                        float(infos["reward_raw"][env_idx]), global_step,
                    )

        rollout_end_time = time.time()
        rollout_dt = rollout_end_time - rollout_start_time
        rollout_sps = (num_steps * num_envs) / rollout_dt if rollout_dt > 0 else 0

        # ==============================================================
        # COMPUTE GAE
        # ==============================================================
        with torch.no_grad():
            _, _, _, next_value, _ = agent.get_action_and_value(
                next_obs.to(device),
                running_prior_action.to(device),
                running_prior_reward.to(device),
                (lstm_h, lstm_c),
            )
            next_value = next_value.flatten().cpu()

        advantages, returns = compute_gae(
            next_value,
            reward_buf,
            done_buf,
            value_buf,
            config["gamma"],
            config["gae_lambda"],
        )

        # ==============================================================
        # PPO UPDATE
        # ==============================================================
        agent.train()

        all_pg_losses = []
        all_v_losses = []
        all_entropy = []
        all_approx_kl = []

        for epoch in range(config["update_epochs"]):
            generator = recurrent_generator(
                obs_buf,
                action_buf,
                logprob_buf,
                value_buf,
                advantages,
                returns,
                prior_action_buf,
                prior_reward_buf,
                hidden_h_buf,
                hidden_c_buf,
                episode_start_buf,
                done_buf,
                num_minibatches,
                config["seq_len"],
            )

            for batch in generator:
                mb_obs = batch["obs"].to(device)
                mb_actions = batch["actions"].to(device)
                mb_logprobs = batch["log_probs"].to(device)
                mb_advantages = batch["advantages"].to(device)
                mb_returns = batch["returns"].to(device)
                mb_values = batch["values"].to(device)
                mb_prior_actions = batch["prior_actions"].to(device)
                mb_prior_rewards = batch["prior_rewards"].to(device)
                mb_hidden_h = batch["hidden_h"].to(device)
                mb_hidden_c = batch["hidden_c"].to(device)
                mb_episode_starts = batch["episode_starts"].to(device)

                _, new_logprob, entropy, new_value, _ = (
                    agent.get_action_and_value(
                        mb_obs,
                        mb_prior_actions,
                        mb_prior_rewards,
                        (
                            mb_hidden_h.permute(1, 0, 2).contiguous(),
                            mb_hidden_c.permute(1, 0, 2).contiguous(),
                        ),
                        action=mb_actions,
                        episode_starts=mb_episode_starts,
                    )
                )

                new_value = new_value.squeeze(-1)
                logratio = new_logprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    all_approx_kl.append(approx_kl.item())

                mb_adv = mb_advantages
                if config["norm_adv"]:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - config["clip_coef"], 1 + config["clip_coef"]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if config["clip_vloss"]:
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values,
                        -config["clip_coef"],
                        config["clip_coef"],
                    )
                    v_loss1 = (new_value - mb_returns) ** 2
                    v_loss2 = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - config["ent_coef"] * entropy_loss
                    + config["vf_coef"] * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), config["max_grad_norm"]
                )
                optimizer.step()

                all_pg_losses.append(pg_loss.item())
                all_v_losses.append(v_loss.item())
                all_entropy.append(entropy_loss.item())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        writer.add_scalar(
            "losses/policy_loss", np.mean(all_pg_losses), global_step
        )
        writer.add_scalar(
            "losses/value_loss", np.mean(all_v_losses), global_step
        )
        writer.add_scalar("losses/entropy", np.mean(all_entropy), global_step)
        writer.add_scalar(
            "losses/approx_kl", np.mean(all_approx_kl), global_step
        )
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar(
            "performance/avg_SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

        # Per-update phase timing
        update_dt = time.time() - update_start_time
        optimize_dt = update_dt - rollout_dt
        current_sps = (num_steps * num_envs) / update_dt if update_dt > 0 else 0
        batch_size = num_steps * num_envs
        rollout_pct = (rollout_dt / update_dt * 100) if update_dt > 0 else 0
        optimize_pct = (optimize_dt / update_dt * 100) if update_dt > 0 else 0
        avg_step_ms = (rollout_dt / num_steps * 1000) if num_steps > 0 else 0
        elapsed_total = time.time() - start_time

        writer.add_scalar("performance/current_SPS", current_sps, global_step)
        writer.add_scalar("performance/rollout_SPS", rollout_sps, global_step)
        writer.add_scalar("performance/rollout_s", rollout_dt, global_step)
        writer.add_scalar("performance/optimize_s", optimize_dt, global_step)
        writer.add_scalar("performance/update_total_s", update_dt, global_step)
        writer.add_scalar("performance/rollout_pct", rollout_pct, global_step)
        writer.add_scalar("performance/optimize_pct", optimize_pct, global_step)
        writer.add_scalar("performance/avg_step_ms", avg_step_ms, global_step)

        # Text summary for easy copy-paste to diagnose bottlenecks
        if update % 10 == 0 or update == 1:
            perf_text = (
                f"**Performance Summary — Update {update}, Step {global_step:,}**\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| num_envs | {num_envs} |\n"
                f"| num_steps (per update) | {num_steps} |\n"
                f"| batch_size (steps×envs) | {batch_size} |\n"
                f"| device | {device} |\n"
                f"| env_version | {_ENV_ID} |\n"
                f"| elapsed_total | {elapsed_total:.1f}s |\n"
                f"| **current_SPS** | **{current_sps:.0f}** |\n"
                f"| rollout_SPS | {rollout_sps:.0f} |\n"
                f"| rollout_s | {rollout_dt:.2f}s ({rollout_pct:.0f}%) |\n"
                f"| optimize_s | {optimize_dt:.2f}s ({optimize_pct:.0f}%) |\n"
                f"| update_total_s | {update_dt:.2f}s |\n"
                f"| avg_step_ms | {avg_step_ms:.1f}ms |\n"
                f"| avg_SPS (lifetime) | {global_step / elapsed_total:.0f} |\n"
            )
            writer.add_text("performance/summary", perf_text, global_step)

        if len(recent_train_returns) > 0:
            writer.add_scalar(
                "train/mean_recent_return",
                np.mean(recent_train_returns[-20:]),
                global_step,
            )

        # Progress print
        if update % 10 == 0 or update == 1:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            mean_ret = (
                np.mean(recent_train_returns[-20:])
                if recent_train_returns
                else float("nan")
            )
            print(
                f"  Update {update:>4}/{num_updates} | Step {global_step:>7,} | "
                f"{sps:.0f} avg SPS | {current_sps:.0f} cur SPS | "
                f"rollout {rollout_dt:.1f}s ({rollout_sps:.0f} SPS) | "
                f"optim {optimize_dt:.1f}s | "
                f"PG: {np.mean(all_pg_losses):.4f} | "
                f"V: {np.mean(all_v_losses):.4f} | Ent: {np.mean(all_entropy):.4f} | "
                f"Mean Ret (20): {mean_ret:.4f}"
            )

        # ==============================================================
        # PERIODIC CHECKPOINT SAVING
        # ==============================================================
        if save_interval > 0 and update % save_interval == 0:
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "best_eval_return": best_eval_return,
                "config": config,
            }, os.path.join(ckpt_dir, f"update_{update}.pt"))
            print(f"  Saved checkpoint at update {update}")

        # ==============================================================
        # PERIODIC EVALUATION WITH VISUALISATION
        # ==============================================================
        if not config.get("no_eval", False) and (update == 1 or update % config["eval_interval"] == 0 or update == num_updates):
            print(
                f"\n  --- Evaluation at update {update} (step {global_step:,}) ---"
            )
            eval_metrics = evaluate_with_visualization(
                agent,
                config,
                device,
                writer,
                global_step,
                tag_prefix="eval",
                num_episodes=config["eval_episodes"],
                run_dir=this_run_dir,
                obs_ref_max=obs_ref_max,
            )
            print(
                f"  Eval: mean_return={eval_metrics['mean_return']:.4f} "
                f"\u00b1 {eval_metrics['std_return']:.4f} | "
                f"best={eval_metrics['best_return']:.4f} | "
                f"zero_baseline={eval_metrics['mean_zero_return']:.4f} | "
                f"gap={eval_metrics['mean_improvement_gap']:+.4f} | "
                f"r_last/r_first={eval_metrics['mean_reward_ratio']:.3f}"
            )

            if eval_metrics["mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["mean_return"]

                # Save best checkpoint (full state + deployable wrapper)
                best_path = os.path.join(ckpt_dir, "best.pt")
                wrapper = PPOActorWrapper(agent)
                torch.save({
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "update": update,
                    "best_eval_return": best_eval_return,
                    "config": config,
                    "policy_wrapper": wrapper,
                }, best_path)

                print(f"  New best eval model saved: {best_eval_return:.4f}")

            writer.add_scalar(
                "eval/best_mean_return", best_eval_return, global_step
            )
            print()

    envs.close()
    writer.close()

    print(f"\nTraining complete. Best eval return: {best_eval_return:.4f}")
    print(f"Run directory:    {this_run_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints:      {ckpt_dir}")

    return best_eval_return, this_run_dir


# ============================================================================
# Generic CLI entry point
# ============================================================================


def run_main(full_config: dict, fast_config: dict):
    """Parse CLI arguments and run PPO training.

    Parameters
    ----------
    full_config : dict
        Full-length training configuration (used by default).
    fast_config : dict
        Smoke-test configuration (used with ``--fast``).
    """
    global _ENV_ID

    parser = argparse.ArgumentParser(description="PPO Optomech Training")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke test with fewer timesteps",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for outputs (default: temp dir)",
    )
    parser.add_argument(
        "--env-version",
        type=str,
        default="v4",
        choices=["v1", "v2", "v3", "v4"],
        help="Optomech environment version (default: v4)",
    )
    parser.add_argument(
        "--action-penalty-weight",
        type=float,
        default=None,
        help="L1 action penalty weight (overrides env config)",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=None,
        help="Checkpoint every N updates (0 = disabled, default: from config)",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default=None,
        help="Path to autoencoder checkpoint for pretrained encoder weights",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help="Freeze pretrained encoder weights during PPO training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (default: from config)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a full PPO checkpoint to resume training from "
             "(loads model weights, optimizer state, and global step)",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to a full PPO checkpoint to initialise weights from "
             "(loads model weights only, starts training from scratch)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel environments (default: from config)",
    )
    parser.add_argument(
        "--async-envs",
        action="store_true",
        default=False,
        help="Use AsyncVectorEnv (multiprocess) instead of SyncVectorEnv",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Disable all evaluation (baselines + periodic eval) for max throughput",
    )
    cli = parser.parse_args()

    _ENV_ID = f"optomech-{cli.env_version}"
    config = dict(fast_config if cli.fast else full_config)

    # Override action penalty weight if specified on command line
    if cli.action_penalty_weight is not None:
        config["env_kwargs"] = dict(config["env_kwargs"])
        config["env_kwargs"]["action_penalty_weight"] = cli.action_penalty_weight

    # Override model save interval if specified on command line
    if cli.model_save_interval is not None:
        config["model_save_interval"] = cli.model_save_interval

    # Override learning rate if specified on command line
    if cli.learning_rate is not None:
        config["learning_rate"] = cli.learning_rate

    # Pretrained encoder
    config["pretrained_encoder"] = cli.pretrained_encoder
    config["freeze_encoder"] = cli.freeze_encoder

    # Override num_envs if specified on command line
    if cli.num_envs is not None:
        config["num_envs"] = cli.num_envs

    # Async env parallelism
    config["async_envs"] = cli.async_envs

    # No-eval mode
    config["no_eval"] = cli.no_eval

    # Full model resume / init
    config["resume_from"] = cli.resume_from
    config["init_from"] = cli.init_from

    # Generate fixed eval seeds deterministically from the main seed.
    # These same seeds are reused every eval cycle so that learning
    # progress is measured on identical initial conditions.
    import numpy as np
    rng = np.random.RandomState(config["seed"])
    config["eval_seeds"] = rng.randint(0, 2**31, size=config["eval_episodes"]).tolist()

    print(f"Using environment: {_ENV_ID}")
    print(f"Fixed eval seeds:  {config['eval_seeds']}")

    # Register environment
    register_optomech(_ENV_ID, max_episode_steps=config["max_episode_steps"])

    # Output directory
    if cli.run_dir:
        run_dir = cli.run_dir
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = os.path.join(_REPO_ROOT, "runs")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")

    # Baselines
    if config.get("no_eval", False):
        print("\n  --no-eval: skipping baselines and periodic evaluation")
        zero_return = float("nan")
        random_return = float("nan")
    else:
        print("\nEvaluating zero-action baseline...")
        zero_return = evaluate_zero_policy(config, num_episodes=3)
        print(f"Zero-action policy mean return: {zero_return:.4f}")

        print("Evaluating random baseline...")
        random_return = evaluate_random_policy(config, num_episodes=3)
        print(f"Random policy mean return: {random_return:.4f}")

    # Train
    best_eval_return, this_run_dir = run_ppo_training(config, run_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Zero-action baseline:  {zero_return:.4f}")
    print(f"  Random baseline:       {random_return:.4f}")
    print(f"  Best trained policy:   {best_eval_return:.4f}")
    print(f"{'='*60}")
    print(f"\nTensorBoard: tensorboard --logdir {os.path.join(this_run_dir, 'tensorboard')}")
