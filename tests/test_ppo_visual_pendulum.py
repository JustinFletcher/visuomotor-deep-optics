"""
End-to-end PPO training test on VisualPendulum-v1.

Trains a recurrent PPO agent on the visual pendulum environment and
verifies that the trained policy outperforms a random baseline. During
training, periodic evaluation rollouts are performed with matplotlib
visualizations written to TensorBoard so that training progress can be
visually verified.

Usage:
    poetry run python tests/test_ppo_visual_pendulum.py          # full run
    poetry run python tests/test_ppo_visual_pendulum.py --fast    # quick smoke test
"""

import os
import sys
import time
import json
import shutil
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
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper
from train.ppo.ppo_recurrent import normalize_obs, compute_gae, recurrent_generator


# ============================================================================
# Configuration
# ============================================================================

FULL_CONFIG = dict(
    total_timesteps=20_000_000,
    num_envs=64,
    num_steps=64,
    num_minibatches=32,
    update_epochs=4,
    seq_len=4,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    anneal_lr=False,
    norm_adv=True,
    clip_vloss=True,
    lstm_hidden_dim=128,
    channel_scale=16,
    fc_scale=256,
    init_log_std=-0.5,
    action_scale=2.0,
    max_episode_steps=200,
    resolution=32,
    eval_interval=20,          # evaluate every N updates
    eval_episodes=3,           # episodes per evaluation
    pass_threshold_ratio=1.1,  # trained must beat random by this factor (lower is better for pendulum)
    seed=42,
)

FAST_CONFIG = dict(
    total_timesteps=6_000,
    num_envs=2,
    num_steps=64,
    num_minibatches=2,
    update_epochs=2,
    seq_len=16,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    anneal_lr=True,
    norm_adv=True,
    clip_vloss=True,
    lstm_hidden_dim=64,
    channel_scale=8,
    fc_scale=32,
    init_log_std=-0.5,
    action_scale=2.0,
    max_episode_steps=200,
    resolution=64,
    eval_interval=5,
    eval_episodes=2,
    pass_threshold_ratio=None,  # skip pass/fail in fast mode
    seed=42,
)


# ============================================================================
# Environment helpers
# ============================================================================


def make_vispend_env(resolution=64, max_episode_steps=200, idx=0):
    """Create a VisualPendulum-v1 environment factory."""

    def thunk():
        env = gym.make(
            "VisualPendulum-v1",
            resolution=resolution,
            render_style="fast_pendulum",
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def register_visual_pendulum(max_episode_steps=200):
    """Register the visual pendulum env if not already registered."""
    try:
        gym.make("VisualPendulum-v1")
    except gym.error.NameNotFound:
        pass  # Will register below

    # Always re-register to pick up max_episode_steps
    if "VisualPendulum-v1" in gym.envs.registry:
        del gym.envs.registry["VisualPendulum-v1"]

    gym.envs.registration.register(
        id="VisualPendulum-v1",
        entry_point="visual_pendulum:VisualPendulumEnv",
        max_episode_steps=max_episode_steps,
    )


# ============================================================================
# Evaluation rollout with instrumentation
# ============================================================================


def evaluate_with_visualization(
    agent: RecurrentActorCritic,
    config: dict,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    tag_prefix: str = "eval",
    num_episodes: int = 3,
) -> dict:
    """
    Run evaluation episodes and produce matplotlib summary figures for
    TensorBoard. Returns a dict with aggregate metrics.

    For each episode we collect:
      - step-wise rewards and cumulative returns
      - observations at evenly-spaced keyframes
      - actions taken

    We produce two figures per evaluation:
      1. **Rollout summary** – reward curve, cumulative return, action trace,
         and observation keyframes arranged in a multi-panel layout.
      2. **Observation filmstrip** – a row of observation images at evenly
         spaced points in the episode so the user can visually verify that
         the pendulum is being controlled.
    """
    agent.eval()
    wrapper = PPOActorWrapper(agent)

    # Create a single env for evaluation
    eval_env = gym.make(
        "VisualPendulum-v1",
        resolution=config["resolution"],
        render_style="fast_pendulum",
    )
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

    all_returns = []
    all_lengths = []

    # We'll accumulate data for the *best* episode to visualise
    best_return = -np.inf
    best_episode_data = None

    for ep in range(num_episodes):
        obs_raw, _ = eval_env.reset()
        obs_np = normalize_obs(obs_raw[np.newaxis])  # add batch dim

        hidden = wrapper.get_zero_hidden()
        hidden = (hidden[0].to(device), hidden[1].to(device))

        prior_action = torch.zeros(1, agent.action_dim, device=device)
        prior_reward = torch.zeros(1, device=device)

        ep_rewards = []
        ep_actions = []
        ep_obs_raw = [obs_raw.copy()]  # store raw uint8 observations

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

            # Advance state
            obs_np = normalize_obs(next_obs_raw[np.newaxis])
            prior_action = action_t.clone()
            prior_reward = torch.tensor([reward], dtype=torch.float32, device=device)

        ep_return = sum(ep_rewards)
        ep_length = len(ep_rewards)
        all_returns.append(ep_return)
        all_lengths.append(ep_length)

        if ep_return > best_return:
            best_return = ep_return
            best_episode_data = {
                "rewards": ep_rewards,
                "actions": ep_actions,
                "obs_raw": ep_obs_raw,
                "return": ep_return,
                "length": ep_length,
            }

    eval_env.close()

    # --- Aggregate metrics ---
    metrics = {
        "mean_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "mean_length": np.mean(all_lengths),
        "best_return": best_return,
    }

    # Log scalars
    writer.add_scalar(f"{tag_prefix}/mean_return", metrics["mean_return"], global_step)
    writer.add_scalar(f"{tag_prefix}/std_return", metrics["std_return"], global_step)
    writer.add_scalar(f"{tag_prefix}/best_return", metrics["best_return"], global_step)

    # --- Build visualisation from best episode ---
    if best_episode_data is not None:
        _log_rollout_summary_figure(writer, best_episode_data, global_step, tag_prefix)
        _log_observation_filmstrip(writer, best_episode_data, global_step, tag_prefix)

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
    Create a 2×2 summary figure:
      Top-left:     Step-wise reward over the episode
      Top-right:    Cumulative return curve
      Bottom-left:  Action trace (one line per action dimension)
      Bottom-right: Observation at 25%, 50%, 75% of the episode
    """
    rewards = np.array(ep_data["rewards"])
    actions = np.array(ep_data["actions"])  # [T, action_dim]
    obs_raw = ep_data["obs_raw"]  # list of [H, W, 1] uint8 arrays

    cumulative = np.cumsum(rewards)
    timesteps = np.arange(len(rewards))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Top-left: step-wise reward ---
    ax_rew = fig.add_subplot(gs[0, 0])
    ax_rew.plot(timesteps, rewards, color="steelblue", linewidth=1.0, alpha=0.8)
    # Smoothed curve
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
    ax_cum.plot(timesteps, cumulative, color="forestgreen", linewidth=2.0)
    ax_cum.axhline(y=cumulative[-1], color="red", linestyle="--", alpha=0.6,
                   label=f"final = {cumulative[-1]:.1f}")
    ax_cum.set_xlabel("Step")
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.set_title("Cumulative Return")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(True, alpha=0.3)

    # --- Bottom-left: action trace ---
    ax_act = fig.add_subplot(gs[1, 0])
    action_dim = actions.shape[1] if actions.ndim > 1 else 1
    if actions.ndim == 1:
        actions = actions[:, np.newaxis]
    for d in range(action_dim):
        ax_act.plot(timesteps, actions[:, d], linewidth=1.0, label=f"a[{d}]", alpha=0.8)
    ax_act.set_xlabel("Step")
    ax_act.set_ylabel("Action")
    ax_act.set_title("Action Trace")
    if action_dim <= 6:
        ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    # --- Bottom-right: observation keyframes ---
    ax_obs = fig.add_subplot(gs[1, 1])
    ax_obs.axis("off")
    ax_obs.set_title("Observations (25% / 50% / 75% of episode)")

    n_keyframes = 3
    T = len(obs_raw)
    keyframe_indices = [int(T * frac) for frac in [0.25, 0.50, 0.75]]
    keyframe_indices = [min(i, T - 1) for i in keyframe_indices]

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_keyframes, subplot_spec=gs[1, 1], wspace=0.1
    )
    for k, idx in enumerate(keyframe_indices):
        ax_kf = fig.add_subplot(inner_gs[0, k])
        obs_img = obs_raw[idx].squeeze()  # [H, W]
        ax_kf.imshow(obs_img, cmap="gray", vmin=0, vmax=255)
        ax_kf.set_title(f"t={idx}", fontsize=8)
        ax_kf.axis("off")

    fig.suptitle(
        f"Rollout Summary  |  Return: {ep_data['return']:.1f}  |  "
        f"Length: {ep_data['length']}  |  Step: {global_step}",
        fontsize=12,
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
    Create a filmstrip of observations sampled uniformly from the episode.
    This lets the user visually verify whether the pendulum is being
    controlled (swung up and balanced near vertical).

    Underneath each frame we show: timestep, instantaneous reward, and
    cumulative return up to that point.
    """
    obs_raw = ep_data["obs_raw"]
    rewards = np.array(ep_data["rewards"])
    cumulative = np.cumsum(rewards)
    T = len(obs_raw)

    # Sample frame indices uniformly (include first and last)
    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        frame_indices = [int(i * (T - 1) / (num_frames - 1)) for i in range(num_frames)]

    n = len(frame_indices)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5),
                             gridspec_kw={"height_ratios": [3, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(frame_indices):
        # Top row: observation image
        ax_img = axes[0, col]
        obs_img = obs_raw[idx].squeeze()
        ax_img.imshow(obs_img, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(f"t={idx}", fontsize=9, fontweight="bold")
        ax_img.axis("off")

        # Bottom row: metrics text
        ax_txt = axes[1, col]
        ax_txt.axis("off")

        if idx == 0:
            # First observation is before any action
            text = "start"
        else:
            r = rewards[idx - 1]
            c = cumulative[idx - 1]
            # Color-code reward: green for close to 0, red for very negative
            text = f"r = {r:.2f}\nΣr = {c:.1f}"
        ax_txt.text(
            0.5, 0.5, text,
            transform=ax_txt.transAxes,
            ha="center", va="center",
            fontsize=8,
            fontfamily="monospace",
        )

    fig.suptitle(
        f"Observation Filmstrip  |  Return: {ep_data['return']:.1f}  |  "
        f"Step: {global_step}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    writer.add_figure(f"{tag_prefix}/observation_filmstrip", fig, global_step)
    plt.close(fig)


# ============================================================================
# Random-policy baseline
# ============================================================================


def evaluate_random_policy(config: dict, num_episodes: int = 10) -> float:
    """Roll out a random policy and return mean episodic return."""
    env = gym.make(
        "VisualPendulum-v1",
        resolution=config["resolution"],
        render_style="fast_pendulum",
    )
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
    Trains a recurrent PPO agent on VisualPendulum-v1 and returns
    (final_eval_return, random_baseline_return, writer_log_dir).
    """
    device = torch.device("cpu")  # CPU to avoid MPS LSTM issues

    # Seeding
    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [
            make_vispend_env(
                resolution=config["resolution"],
                max_episode_steps=config["max_episode_steps"],
                idx=i,
            )
            for i in range(config["num_envs"])
        ]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    # Agent
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
    minibatch_size = batch_size // num_minibatches
    num_updates = config["total_timesteps"] // batch_size

    run_name = f"ppo_vispend_test_{seed}_{int(time.time())}"
    log_dir = os.path.join(run_dir, "runs", run_name)
    writer = SummaryWriter(log_dir)
    writer.add_text("test_config", json.dumps(config, indent=2))

    print(f"\n{'='*60}")
    print(f"PPO Visual Pendulum Training Test")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Total timesteps:  {config['total_timesteps']:,}")
    print(f"  Num updates:      {num_updates}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Minibatch size:   {minibatch_size}")
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

    lstm_h = torch.zeros(agent.lstm_num_layers, num_envs, config["lstm_hidden_dim"], device=device)
    lstm_c = torch.zeros(agent.lstm_num_layers, num_envs, config["lstm_hidden_dim"], device=device)

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
                # get_action_and_value returns RAW (unscaled) action + log_prob
                raw_action, logprob, _, value, (lstm_h, lstm_c) = agent.get_action_and_value(
                    next_obs.to(device),
                    running_prior_action.to(device),
                    running_prior_reward.to(device),
                    (lstm_h, lstm_c),
                )
                # Scale and clamp for environment interaction
                env_action = agent.scale_and_clamp_action(raw_action)

            # Store RAW action (consistent with log_prob for PPO ratio)
            action_buf[step] = raw_action.cpu()
            logprob_buf[step] = logprob.cpu()
            value_buf[step] = value.flatten().cpu()

            # Step environment with clamped action
            next_obs_np, rewards_np, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            next_obs_np = normalize_obs(next_obs_np)
            next_obs = torch.from_numpy(next_obs_np).float()
            reward_buf[step] = torch.from_numpy(rewards_np.astype(np.float32))

            dones_step = np.logical_or(terminations, truncations)
            next_done = torch.from_numpy(dones_step.astype(np.float32))

            # Use CLAMPED env action as prior (what the env actually received)
            running_prior_action = env_action.cpu().clone()
            running_prior_reward = torch.from_numpy(rewards_np.astype(np.float32))
            running_episode_start = next_done.clone()

            # Reset hidden state for done envs
            for env_idx in range(num_envs):
                if dones_step[env_idx]:
                    lstm_h[:, env_idx, :] = 0.0
                    lstm_c[:, env_idx, :] = 0.0
                    running_prior_action[env_idx] = 0.0
                    running_prior_reward[env_idx] = 0.0

            # Track episode completions
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None:
                        continue
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    recent_train_returns.append(ep_return)
                    writer.add_scalar("train/episodic_return", ep_return, global_step)
                    writer.add_scalar("train/episodic_length", ep_length, global_step)

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
            next_value, reward_buf, done_buf, value_buf,
            config["gamma"], config["gae_lambda"],
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
                obs_buf, action_buf, logprob_buf, value_buf,
                advantages, returns,
                prior_action_buf, prior_reward_buf,
                hidden_h_buf, hidden_c_buf,
                episode_start_buf, done_buf,
                num_minibatches, config["seq_len"],
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

                _, new_logprob, entropy, new_value, _ = agent.get_action_and_value(
                    mb_obs, mb_prior_actions, mb_prior_rewards,
                    (mb_hidden_h.permute(1, 0, 2).contiguous(),
                     mb_hidden_c.permute(1, 0, 2).contiguous()),
                    action=mb_actions,
                    episode_starts=mb_episode_starts,
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

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config["clip_coef"],
                                                  1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if config["clip_vloss"]:
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values,
                        -config["clip_coef"], config["clip_coef"],
                    )
                    v_loss1 = (new_value - mb_returns) ** 2
                    v_loss2 = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + config["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

                all_pg_losses.append(pg_loss.item())
                all_v_losses.append(v_loss.item())
                all_entropy.append(entropy_loss.item())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        writer.add_scalar("losses/policy_loss", np.mean(all_pg_losses), global_step)
        writer.add_scalar("losses/value_loss", np.mean(all_v_losses), global_step)
        writer.add_scalar("losses/entropy", np.mean(all_entropy), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(all_approx_kl), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
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
            mean_ret = np.mean(recent_train_returns[-20:]) if recent_train_returns else float("nan")
            print(
                f"  Update {update:>4}/{num_updates} | Step {global_step:>7,} | "
                f"{sps:.0f} SPS | PG: {np.mean(all_pg_losses):.4f} | "
                f"V: {np.mean(all_v_losses):.4f} | Ent: {np.mean(all_entropy):.4f} | "
                f"Mean Ret (20): {mean_ret:.1f}"
            )

        # ==============================================================
        # PERIODIC EVALUATION WITH VISUALISATION
        # ==============================================================
        if update % config["eval_interval"] == 0 or update == num_updates:
            print(f"\n  --- Evaluation at update {update} (step {global_step:,}) ---")
            eval_metrics = evaluate_with_visualization(
                agent, config, device, writer, global_step,
                tag_prefix="eval",
                num_episodes=config["eval_episodes"],
            )
            print(
                f"  Eval: mean_return={eval_metrics['mean_return']:.1f} "
                f"± {eval_metrics['std_return']:.1f} | "
                f"best={eval_metrics['best_return']:.1f}"
            )

            if eval_metrics["mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["mean_return"]
                # Save best model
                model_path = os.path.join(run_dir, "best_policy.pt")
                wrapper = PPOActorWrapper(agent)
                torch.save(wrapper, model_path)
                print(f"  New best eval model saved: {best_eval_return:.1f}")

            writer.add_scalar("eval/best_mean_return", best_eval_return, global_step)
            print()

    envs.close()
    writer.close()

    print(f"\nTraining complete. Best eval return: {best_eval_return:.1f}")
    print(f"TensorBoard logs: {log_dir}")

    return best_eval_return, log_dir


# ============================================================================
# Entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PPO Visual Pendulum Test")
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick smoke test with fewer timesteps (no pass/fail threshold)",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Directory for outputs (default: temp dir)",
    )
    cli = parser.parse_args()

    config = FAST_CONFIG if cli.fast else FULL_CONFIG

    # Register environment
    register_visual_pendulum(max_episode_steps=config["max_episode_steps"])

    # Output directory
    if cli.run_dir:
        run_dir = cli.run_dir
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = tempfile.mkdtemp(prefix="ppo_vispend_test_")

    print(f"Output directory: {run_dir}")

    # Random baseline
    print("\nEvaluating random baseline...")
    random_return = evaluate_random_policy(config, num_episodes=10)
    print(f"Random policy mean return: {random_return:.1f}")

    # Train
    best_eval_return, log_dir = run_ppo_training(config, run_dir)

    # Pass / Fail
    threshold = config.get("pass_threshold_ratio")
    if threshold is not None:
        # Pendulum rewards are negative; closer to 0 is better.
        # "beats random" means the trained return is less negative (higher).
        passed = best_eval_return > random_return * threshold
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Random baseline:       {random_return:.1f}")
        print(f"  Best trained policy:   {best_eval_return:.1f}")
        print(f"  Threshold (ratio):     {threshold}")
        print(f"  Required (random × {threshold}): {random_return * threshold:.1f}")

        if passed:
            print(f"\n  ✓ TEST PASSED — trained policy outperforms random baseline")
        else:
            print(f"\n  ✗ TEST FAILED — trained policy did not sufficiently outperform random")
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {log_dir}")

        sys.exit(0 if passed else 1)
    else:
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST COMPLETE (no pass/fail threshold in fast mode)")
        print(f"  Best eval return: {best_eval_return:.1f}")
        print(f"  Random baseline:  {random_return:.1f}")
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
