"""
End-to-end PPO training test on optomech with nanoelf aperture.

Trains a recurrent PPO agent on the nanoelf distributed-aperture telescope
environment and verifies that the trained policy outperforms a random
baseline (zero-action policy). During training, periodic evaluation
rollouts are performed with matplotlib visualizations written to
TensorBoard so that training progress can be visually verified.

The nanoelf environment has:
  - Observation: (1, 128, 128) float32 focal-plane image in [0, 1]
  - Action: (2,) float32 piston commands in [-1, 1]
  - Reward: alignment-based (higher = better)

Usage:
    poetry run python tests/test_ppo_nanoelf.py                          # full run (v4)
    poetry run python tests/test_ppo_nanoelf.py --fast                   # quick smoke test
    poetry run python tests/test_ppo_nanoelf.py --env-version v3         # use optomech-v3
    poetry run python tests/test_ppo_nanoelf.py --env-version v3 --fast  # v3 smoke test
    poetry run python tests/test_ppo_nanoelf.py --run-dir ./output       # specify output dir
"""

import os
import sys
import time
import json
import argparse
import tempfile
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
import imageio
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optomech"))

from models.ppo_models import RecurrentActorCritic, PPOActorWrapper
from optomech.rl.ppo_recurrent import normalize_obs, compute_gae, recurrent_generator


# ============================================================================
# Configuration
# ============================================================================

# Load the full nanoelf config — the optomech env requires all kwargs to be
# present (no defaults), so we start from the JSON and override as needed.
_NANOELF_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "optomech", "rl", "nanoelf_optomech_config.json"
)
with open(_NANOELF_CONFIG_PATH, "r") as _f:
    _nanoelf_base = json.load(_f)

# Add any missing required kwargs with safe defaults
_nanoelf_base.setdefault("report_time", False)
_nanoelf_base.setdefault("render_dpi", 100.0)
_nanoelf_base.setdefault("record_env_state_info", False)
_nanoelf_base.setdefault("write_env_state_info", False)
_nanoelf_base.setdefault("render_frequency", 1)
_nanoelf_base.setdefault("randomize_dm", False)
_nanoelf_base.setdefault("command_tip_tilt", False)
_nanoelf_base.setdefault("command_tensioners", False)
_nanoelf_base.setdefault("discrete_control", False)
_nanoelf_base.setdefault("discrete_control_steps", 128)
_nanoelf_base.setdefault("simulate_differential_motion", False)
_nanoelf_base.setdefault("incremental_control", False)
_nanoelf_base.setdefault("dark_hole", False)
_nanoelf_base.setdefault("dark_hole_angular_location_degrees", 60)
_nanoelf_base.setdefault("dark_hole_location_radius_fraction", 0.4)
_nanoelf_base.setdefault("dark_hole_size_radius", 0.1)
_nanoelf_base.setdefault("model_gravity_diff_motion", False)
_nanoelf_base.setdefault("model_temp_diff_motion", False)
_nanoelf_base.setdefault("model_ao", False)
_nanoelf_base.setdefault("num_tensioners", 0)

# Strip out keys that are RL-training-specific (not env kwargs)
_rl_only_keys = {
    "env_id", "seed", "cuda", "gpu_list", "total_timesteps",
    "actor_learning_rate", "critic_learning_rate", "buffer_size", "gamma",
    "tau", "batch_size", "exploration_noise", "policy_noise",
    "learning_starts", "experience_sampling_delay", "policy_frequency",
    "noise_clip", "action_scale", "reward_scale", "max_grad_norm",
    "actor_type", "critic_type", "actor_channel_scale", "actor_fc_scale",
    "qnetwork_channel_scale", "qnetwork_fc_scale", "lstm_hidden_dim",
    "tbptt_seq_len", "target_smoothing", "model_save_interval",
    "writer_interval", "num_eval_rollouts", "use_td3_replay_buffer",
    "pretrained_encoder_path", "freeze_encoder", "pretrained_actor_path",
    "freeze_actor_encoder", "replay_buffer_cutoff_step", "sa_dataset_path",
    "sa_action_type", "sa_max_sequences", "sa_max_episodes",
    "timing_enabled", "timing_interval", "num_envs",
}
NANOELF_ENV_KWARGS = {k: v for k, v in _nanoelf_base.items() if k not in _rl_only_keys}
# RL-only overrides from the JSON (e.g. reward_scale) — merged into config at runtime.
_rl_overrides_from_json = {k: v for k, v in _nanoelf_base.items() if k in _rl_only_keys}

FULL_CONFIG = dict(
    # PPO hyperparameters
    total_timesteps=20_000_000,
    num_envs=4,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    anneal_lr=True,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    # Model architecture
    lstm_hidden_dim=128,
    channel_scale=16,
    fc_scale=128,
    init_log_std=-0.5,
    action_scale=0.1,
    # Environment
    max_episode_steps=256,
    # Evaluation — fixed seeds for deterministic eval scenarios
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,  # populated at startup from seed
    pass_threshold_ratio=1.1,
    seed=42,
    # Nanoelf env kwargs
    env_kwargs=NANOELF_ENV_KWARGS,
)

FAST_CONFIG = dict(
    total_timesteps=2_000,
    num_envs=1,
    num_steps=100,
    num_minibatches=2,
    update_epochs=2,
    seq_len=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    anneal_lr=True,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    lstm_hidden_dim=64,
    channel_scale=8,
    fc_scale=32,
    init_log_std=-0.5,
    action_scale=1.0,
    max_episode_steps=100,
    eval_interval=3,
    eval_episodes=4,
    eval_seeds=None,  # populated at startup from seed
    pass_threshold_ratio=None,  # skip pass/fail in fast mode
    seed=42,
    env_kwargs=NANOELF_ENV_KWARGS,
)


# ============================================================================
# Environment helpers
# ============================================================================

# Module-level env ID, set by --env-version flag
_ENV_ID = "optomech-v4"

_ENTRY_POINTS = {
    "optomech-v1": "optomech.optomech:OptomechEnv",
    "optomech-v2": "optomech.optomech_v2:OptomechEnv",
    "optomech-v3": "optomech.optomech_v3:OptomechEnv",
    "optomech-v4": "optomech.optomech_v4:OptomechEnv",
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


def make_nanoelf_env(env_kwargs: dict, max_episode_steps: int = 100, idx: int = 0):
    """Create an optomech nanoelf environment factory."""

    def thunk():
        kwargs = dict(env_kwargs)
        kwargs["max_episode_steps"] = max_episode_steps
        env = gym.make(_ENV_ID, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


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
    run_name: str = "",
) -> dict:
    """
    Run deterministic evaluation episodes with fixed seeds and produce
    matplotlib summary figures for TensorBoard.

    For each fixed-seed episode, also runs a zero-action baseline on the
    same seed and logs:
      - Per-seed agent return vs zero-action baseline
      - Relative improvement (agent_return / zero_baseline)
      - Reward at step 0 vs step T
      - Mean improvement gap across all seeds
    """
    agent.eval()
    wrapper = PPOActorWrapper(agent)

    env_kwargs = dict(config["env_kwargs"])
    env_kwargs["max_episode_steps"] = config["max_episode_steps"]
    eval_env = gym.make(_ENV_ID, **env_kwargs)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

    eval_seeds = config.get("eval_seeds")
    if eval_seeds is None:
        eval_seeds = list(range(num_episodes))

    # Use the requested number of episodes (may be fewer than seeds available)
    seeds_to_use = eval_seeds[:num_episodes]

    all_returns = []
    all_lengths = []
    all_zero_returns = []
    all_improvement_gaps = []
    all_relative_improvements = []
    all_first_rewards = []
    all_last_rewards = []
    all_final_strehls = []
    all_final_mses = []

    # Per-step traces for aggregate plots
    all_agent_reward_traces = []
    all_zero_reward_traces = []
    all_random_reward_traces = []
    all_agent_strehl_traces = []
    all_agent_mse_traces = []

    all_episode_data = []

    for ep, seed in enumerate(seeds_to_use):
        # --- Zero-action baseline for this seed ----------------------
        zero_return, zero_first_reward, zero_rewards = _run_zero_action_episode(
            eval_env, seed)
        all_zero_returns.append(zero_return)
        all_zero_reward_traces.append(zero_rewards)

        # --- Random-action baseline for this seed --------------------
        random_return, random_rewards = _run_random_action_episode(eval_env, seed)
        all_random_reward_traces.append(random_rewards)

        # --- Agent rollout on the SAME seed --------------------------
        obs_raw, _ = eval_env.reset(seed=seed)
        obs_np = normalize_obs(obs_raw[np.newaxis])  # add batch dim

        hidden = wrapper.get_zero_hidden()
        hidden = (hidden[0].to(device), hidden[1].to(device))

        prior_action = torch.zeros(1, agent.action_dim, device=device)
        prior_reward = torch.zeros(1, device=device)

        ep_rewards = []
        ep_actions = []
        ep_obs_raw = [obs_raw.copy()]
        ep_strehls = []
        ep_mses = []
        ep_oob_fracs = []

        done = False
        while not done:
            obs_t = torch.from_numpy(obs_np).float().to(device)
            with torch.no_grad():
                action_t, hidden = wrapper(obs_t, prior_action, prior_reward, hidden)
            action = action_t.cpu().numpy()[0]

            next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            ep_rewards.append(float(reward))
            ep_actions.append(action.copy())
            ep_obs_raw.append(next_obs_raw.copy())
            ep_strehls.append(float(info.get("strehl", 0.0)))
            ep_mses.append(float(info.get("mse", 0.0)))
            ep_oob_fracs.append(float(info.get("oob_frac", 0.0)))

            obs_np = normalize_obs(next_obs_raw[np.newaxis])
            prior_action = action_t.clone()
            prior_reward = torch.tensor([reward], dtype=torch.float32, device=device)

        ep_return = sum(ep_rewards)
        ep_length = len(ep_rewards)
        all_returns.append(ep_return)
        all_lengths.append(ep_length)
        all_agent_reward_traces.append(ep_rewards)
        all_agent_strehl_traces.append(ep_strehls)
        all_agent_mse_traces.append(ep_mses)
        all_final_strehls.append(ep_strehls[-1] if ep_strehls else 0.0)
        all_final_mses.append(ep_mses[-1] if ep_mses else 0.0)

        # Relative improvement: how much better than zero-action?
        improvement_gap = ep_return - zero_return
        all_improvement_gaps.append(improvement_gap)

        # Ratio of agent vs zero-action (avoid div-by-zero)
        if abs(zero_return) > 1e-10:
            relative_improvement = ep_return / zero_return
        else:
            relative_improvement = 1.0
        all_relative_improvements.append(relative_improvement)

        # First vs last step reward (item #2)
        first_reward = ep_rewards[0] if ep_rewards else 0.0
        last_reward = ep_rewards[-1] if ep_rewards else 0.0
        all_first_rewards.append(first_reward)
        all_last_rewards.append(last_reward)

        # Per-seed TensorBoard logging
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/agent_return", ep_return, global_step
        )
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/zero_return", zero_return, global_step
        )
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/improvement_gap", improvement_gap, global_step
        )
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/first_reward", first_reward, global_step
        )
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/last_reward", last_reward, global_step
        )
        final_strehl = ep_strehls[-1] if ep_strehls else 0.0
        final_mse = ep_mses[-1] if ep_mses else 0.0
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/final_strehl", final_strehl, global_step
        )
        writer.add_scalar(
            f"{tag_prefix}/seed_{seed}/final_mse", final_mse, global_step
        )

        all_episode_data.append({
            "rewards": ep_rewards,
            "actions": ep_actions,
            "obs_raw": ep_obs_raw,
            "strehls": ep_strehls,
            "mses": ep_mses,
            "oob_fracs": ep_oob_fracs,
            "return": ep_return,
            "length": ep_length,
            "seed": seed,
            "zero_return": zero_return,
            "improvement_gap": improvement_gap,
        })

    eval_env.close()

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
    if run_name:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gif_dir = os.path.join(project_root, "eval_gifs", run_name, f"step_{global_step}")
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
    2×2 summary:
      Top-left:     Step-wise reward
      Top-right:    Cumulative return
      Bottom-left:  Action trace (one line per piston DOF)
      Bottom-right: OOB (out-of-bounds) fraction per step
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
    labels = [f"piston[{d}]" for d in range(action_dim)]
    for d in range(action_dim):
        ax_act.plot(
            timesteps, actions[:, d], linewidth=1.0, label=labels[d], alpha=0.8
        )
    ax_act.set_xlabel("Step")
    ax_act.set_ylabel("Action")
    ax_act.set_title("Piston Commands")
    ax_act.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    if action_dim <= 6:
        ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    # --- Bottom-right: OOB fraction per step ---
    ax_oob = fig.add_subplot(gs[1, 1])
    oob_fracs = np.array(ep_data.get("oob_fracs", []))
    if len(oob_fracs) > 0:
        oob_colors = ["#d94a4a" if f > 0 else "#4a90d9" for f in oob_fracs]
        ax_oob.bar(timesteps, oob_fracs, color=oob_colors, width=1.0, alpha=0.8)
        ax_oob.set_xlabel("Step")
        ax_oob.set_ylabel("OOB Fraction")
        total_oob_steps = int(np.sum(oob_fracs > 0))
        ax_oob.set_title(
            f"OOB Occurrences ({total_oob_steps}/{len(oob_fracs)} steps)", fontsize=9)
        ax_oob.set_ylim(0, max(1.0, float(np.max(oob_fracs)) * 1.1))
        ax_oob.grid(True, alpha=0.3)
    else:
        ax_oob.text(0.5, 0.5, "No OOB data", ha="center", va="center",
                     fontsize=10, transform=ax_oob.transAxes)
        ax_oob.set_title("OOB Occurrences")

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
        "Nanoelf Rollout  |  " + "  |  ".join(title_parts),
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
    Uses log-stretch and inferno colourmap for visibility.
    """
    obs_raw = ep_data["obs_raw"]
    rewards = np.array(ep_data["rewards"])
    cumulative = np.cumsum(rewards)
    T = len(obs_raw)

    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        frame_indices = [
            int(i * (T - 1) / (num_frames - 1)) for i in range(num_frames)
        ]

    n = len(frame_indices)
    fig, axes = plt.subplots(
        2, n, figsize=(2.5 * n, 5.5), gridspec_kw={"height_ratios": [3, 1]}
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(frame_indices):
        ax_img = axes[0, col]
        img_disp = _prepare_obs_for_display(obs_raw[idx])
        ax_img.imshow(img_disp, cmap="inferno", vmin=0, vmax=1)
        ax_img.set_title(f"t={idx}", fontsize=9, fontweight="bold")
        ax_img.axis("off")

        ax_txt = axes[1, col]
        ax_txt.axis("off")

        if idx == 0:
            text = "start"
        else:
            r = rewards[idx - 1]
            c = cumulative[idx - 1]
            text = f"r = {r:.4f}\n\u03a3r = {c:.3f}"
        ax_txt.text(
            0.5,
            0.5,
            text,
            transform=ax_txt.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontfamily="monospace",
        )

    fig.suptitle(
        f"Focal-Plane Filmstrip  |  Return: {ep_data['return']:.3f}  |  "
        f"Step: {global_step}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    writer.add_figure(f"{tag_prefix}/observation_filmstrip", fig, global_step)
    plt.close(fig)


def _render_episode_gif(ep_data, save_path, label, global_step,
                        dpi=72, frame_duration=0.2):
    """Render an evaluation episode as an animated GIF.

    Each frame shows the focal-plane observation (top), action bar chart
    (bottom-left), and step metrics (bottom-right).
    """
    obs_raw = ep_data["obs_raw"]
    rewards = ep_data["rewards"]
    actions = np.array(ep_data["actions"])  # [T, action_dim]
    strehls = ep_data.get("strehls", [])
    cumulative = np.cumsum(rewards)
    T = len(rewards)
    action_dim = actions.shape[1] if len(actions) > 0 else 0

    frames = []
    for t in range(T + 1):  # T+1 because obs_raw includes initial obs
        fig = plt.figure(figsize=(4, 5), dpi=dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[3, 1], hspace=0.35, wspace=0.4)

        # Top: observation image spanning both columns
        ax_obs = fig.add_subplot(gs[0, :])
        img_disp = _prepare_obs_for_display(obs_raw[t])
        ax_obs.imshow(img_disp, cmap="inferno", vmin=0, vmax=1)
        ax_obs.axis("off")
        ax_obs.set_title(
            f"{label} | seed={ep_data['seed']} | R={ep_data['return']:.3f}"
            f" | train step {global_step}",
            fontsize=7)

        # Bottom-left: action bars
        ax_act = fig.add_subplot(gs[1, 0])
        if t > 0 and action_dim > 0:
            act = actions[t - 1]
            colors = ["#4a90d9", "#d94a4a", "#4ad94a", "#d9d94a"][:action_dim]
            ax_act.barh(range(action_dim), act, color=colors)
            ax_act.set_xlim(-1.1, 1.1)
            ax_act.set_yticks(range(action_dim))
            ax_act.set_yticklabels([f"p[{i}]" for i in range(action_dim)],
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
            text = f"t = 0 / {T}\n(initial obs)"
        else:
            r = rewards[t - 1]
            c = cumulative[t - 1]
            s_txt = f"\nS = {strehls[t-1]:.4f}" if strehls else ""
            text = f"t = {t} / {T}\nr = {r:.4f}\nR = {c:.3f}{s_txt}"
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
    1×2 aggregate figure across all eval episodes:
      Left:  Mean ± std step-wise reward (agent vs zero vs random)
      Right: Mean ± std cumulative return (agent vs zero vs random)

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

    # --- Left: step-wise reward mean ± std ---
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

    # --- Right: cumulative return mean ± std ---
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
# Main training + test loop
# ============================================================================


def run_ppo_training(config: dict, run_dir: str):
    """
    Trains a recurrent PPO agent on optomech (nanoelf) and returns
    (best_eval_return, tensorboard_log_dir).
    """
    device = torch.device("cpu")  # CPU to avoid MPS LSTM issues

    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [
            make_nanoelf_env(
                config["env_kwargs"],
                max_episode_steps=config["max_episode_steps"],
                idx=i,
            )
            for i in range(config["num_envs"])
        ]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    agent = RecurrentActorCritic(
        envs,
        device,
        lstm_hidden_dim=config["lstm_hidden_dim"],
        channel_scale=config["channel_scale"],
        fc_scale=config["fc_scale"],
        action_scale=config["action_scale"],
        init_log_std=config["init_log_std"],
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

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

    run_name = f"ppo_nanoelf_test_{seed}_{int(time.time())}"
    log_dir = os.path.join(run_dir, "runs", run_name)
    writer = SummaryWriter(log_dir)
    writer.add_text("test_config", json.dumps(config, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"PPO Nanoelf Training Test")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Obs shape:        {obs_shape}")
    print(f"  Action shape:     {action_shape}")
    print(f"  Total timesteps:  {config['total_timesteps']:,}")
    print(f"  Num updates:      {num_updates}")
    print(f"  Batch size:       {batch_size} ({num_envs} envs x {num_steps} steps)")
    print(f"  Minibatch size:   {minibatch_size}")
    print(f"  Reward scale:     {reward_scale}")
    print(f"  Parameters:       {sum(p.numel() for p in agent.parameters()):,}")
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
    obs_np = normalize_obs(obs_np)
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
    start_time = time.time()
    recent_train_returns = []

    for update in range(1, num_updates + 1):
        # LR annealing
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = config["learning_rate"] * frac

        # ==============================================================
        # ROLLOUT PHASE
        # ==============================================================
        agent.eval()
        for step in range(num_steps):
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
            next_obs_np = normalize_obs(next_obs_np)
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
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

        if len(recent_train_returns) > 0:
            writer.add_scalar(
                "train/mean_recent_return",
                np.mean(recent_train_returns[-20:]),
                global_step,
            )

        # Progress print
        if update % max(1, num_updates // 20) == 0 or update == 1:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            mean_ret = (
                np.mean(recent_train_returns[-20:])
                if recent_train_returns
                else float("nan")
            )
            print(
                f"  Update {update:>4}/{num_updates} | Step {global_step:>7,} | "
                f"{sps:.0f} SPS | PG: {np.mean(all_pg_losses):.4f} | "
                f"V: {np.mean(all_v_losses):.4f} | Ent: {np.mean(all_entropy):.4f} | "
                f"Mean Ret (20): {mean_ret:.4f}"
            )

        # ==============================================================
        # PERIODIC EVALUATION WITH VISUALISATION
        # ==============================================================
        if update % config["eval_interval"] == 0 or update == num_updates:
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
                run_name=run_name,
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
                model_path = os.path.join(run_dir, "best_policy.pt")
                wrapper = PPOActorWrapper(agent)
                torch.save(wrapper, model_path)
                print(f"  New best eval model saved: {best_eval_return:.4f}")

            writer.add_scalar(
                "eval/best_mean_return", best_eval_return, global_step
            )
            print()

    envs.close()
    writer.close()

    print(f"\nTraining complete. Best eval return: {best_eval_return:.4f}")
    print(f"TensorBoard logs: {log_dir}")

    return best_eval_return, log_dir


# ============================================================================
# Entry point
# ============================================================================


def main():
    global _ENV_ID

    parser = argparse.ArgumentParser(description="PPO Nanoelf Optomech Test")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke test with fewer timesteps (no pass/fail threshold)",
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
        help="L2 action penalty weight (default: env default, typically 0.03)",
    )
    cli = parser.parse_args()

    _ENV_ID = f"optomech-{cli.env_version}"
    config = dict(FAST_CONFIG if cli.fast else FULL_CONFIG)
    config.update(_rl_overrides_from_json)  # JSON RL keys override hardcoded defaults

    # Override action penalty weight if specified on command line
    if cli.action_penalty_weight is not None:
        config["env_kwargs"] = dict(config["env_kwargs"])
        config["env_kwargs"]["action_penalty_weight"] = cli.action_penalty_weight

    # Generate fixed eval seeds deterministically from the main seed.
    # These same seeds are reused every eval cycle so that learning
    # progress is measured on identical initial conditions.
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
        run_dir = tempfile.mkdtemp(prefix="ppo_nanoelf_test_")

    print(f"Output directory: {run_dir}")

    # Baselines
    print("\nEvaluating zero-action baseline...")
    zero_return = evaluate_zero_policy(config, num_episodes=3)
    print(f"Zero-action policy mean return: {zero_return:.4f}")

    print("Evaluating random baseline...")
    random_return = evaluate_random_policy(config, num_episodes=3)
    print(f"Random policy mean return: {random_return:.4f}")

    # Train
    best_eval_return, log_dir = run_ppo_training(config, run_dir)

    # Pass / Fail
    threshold = config.get("pass_threshold_ratio")
    if threshold is not None:
        # For nanoelf, higher return = better alignment.
        # The trained policy should beat the better of (zero, random) baselines.
        baseline = max(zero_return, random_return)
        passed = best_eval_return > baseline * threshold

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Zero-action baseline:  {zero_return:.4f}")
        print(f"  Random baseline:       {random_return:.4f}")
        print(f"  Best trained policy:   {best_eval_return:.4f}")
        print(f"  Threshold (ratio):     {threshold}")
        print(f"  Required (baseline x {threshold}): {baseline * threshold:.4f}")

        if passed:
            print(
                f"\n  \u2713 TEST PASSED \u2014 trained policy outperforms baselines"
            )
        else:
            print(
                f"\n  \u2717 TEST FAILED \u2014 trained policy did not sufficiently outperform baselines"
            )
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {log_dir}")

        sys.exit(0 if passed else 1)
    else:
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST COMPLETE (no pass/fail threshold in fast mode)")
        print(f"  Best eval return:      {best_eval_return:.4f}")
        print(f"  Zero-action baseline:  {zero_return:.4f}")
        print(f"  Random baseline:       {random_return:.4f}")
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
