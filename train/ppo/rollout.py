"""
Reusable rollout evaluation for PPO agents on optomech environments.

This module provides the core `run_rollouts()` function that loads a
checkpoint, performs N independent episodes, and returns structured
episode data.  It also provides helpers for GIF rendering and summary
figures suitable for papers / dissertations.

Designed to be called both as a standalone CLI and from analysis scripts
(e.g. tip/tilt sweep).

Standalone usage:
    poetry run python train/ppo/rollout.py \\
        --checkpoint runs/<run>/checkpoints/best.pt \\
        --env-version v3 --num-episodes 8

From another script:
    from train.ppo.rollout import run_rollouts, save_episode_gifs, save_summary_figures
    episodes, metrics = run_rollouts(checkpoint_path, env_kwargs, ...)
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import imageio

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper
from train.ppo.train_ppo_optomech import (
    register_optomech,
    normalize_obs_fixed,
    _prepare_obs_raw,
    _run_zero_action_episode,
    _run_random_action_episode,
    _ENV_ID,
)

# Default env kwargs matching train_ppo_nanoelf_tt.py
from train.ppo.train_ppo_nanoelf_tt import NANOELF_TT_ENV_KWARGS

_DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "test_output")


# ============================================================================
# Core rollout engine
# ============================================================================


def load_agent(checkpoint_path, env, device="cpu"):
    """Load a PPO agent from a checkpoint file.

    Returns (agent, config, obs_ref_max).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    agent = RecurrentActorCritic(
        env,
        device,
        lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
        channel_scale=config.get("channel_scale", 32),
        fc_scale=config.get("fc_scale", 256),
        action_scale=config.get("action_scale", 1.0),
        init_log_std=config.get("init_log_std", -0.5),
    ).to(device)

    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    # Get obs_ref_max from environment
    base_env = env.unwrapped
    if hasattr(base_env, 'optical_system') and hasattr(base_env.optical_system, '_reference_fpi_max'):
        obs_ref_max = base_env.optical_system._reference_fpi_max
    else:
        obs_ref_max = 1.0

    return agent, config, obs_ref_max


def run_single_episode(agent, env, seed, obs_ref_max, device="cpu"):
    """Run a single deterministic episode and return structured data.

    Returns a dict with keys: rewards, actions, obs_raw, strehls, mses,
    oob_fracs, return, length, seed, zero_return, random_return,
    improvement_gap.
    """
    wrapper = PPOActorWrapper(agent)

    # Zero-action baseline
    zero_return, _, zero_rewards = _run_zero_action_episode(env, seed)

    # Random-action baseline
    random_return, random_rewards = _run_random_action_episode(env, seed)

    # Agent rollout
    obs_raw, _ = env.reset(seed=seed)
    obs_np = normalize_obs_fixed(obs_raw[np.newaxis], obs_ref_max)

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

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_rewards.append(float(reward))
        ep_actions.append(action.copy())
        ep_obs_raw.append(next_obs_raw.copy())
        ep_strehls.append(float(info.get("strehl", 0.0)))
        ep_mses.append(float(info.get("mse", 0.0)))
        ep_oob_fracs.append(float(info.get("oob_frac", 0.0)))

        obs_np = normalize_obs_fixed(next_obs_raw[np.newaxis], obs_ref_max)
        prior_action = action_t.clone()
        prior_reward = torch.tensor([reward], dtype=torch.float32, device=device)

    ep_return = sum(ep_rewards)
    return {
        "rewards": ep_rewards,
        "actions": ep_actions,
        "obs_raw": ep_obs_raw,
        "strehls": ep_strehls,
        "mses": ep_mses,
        "oob_fracs": ep_oob_fracs,
        "return": ep_return,
        "length": len(ep_rewards),
        "seed": seed,
        "zero_return": zero_return,
        "zero_rewards": zero_rewards,
        "random_return": random_return,
        "random_rewards": random_rewards,
        "improvement_gap": ep_return - zero_return,
    }


def run_rollouts(
    checkpoint_path,
    env_kwargs=None,
    env_version="v3",
    num_episodes=8,
    max_episode_steps=256,
    seeds=None,
    device="cpu",
):
    """Run multiple independent evaluation rollouts.

    Parameters
    ----------
    checkpoint_path : str
        Path to a PPO checkpoint (.pt).
    env_kwargs : dict, optional
        Environment kwargs. Defaults to NANOELF_TT_ENV_KWARGS.
    env_version : str
        Optomech env version (default: "v3").
    num_episodes : int
        Number of episodes to run (default: 8).
    max_episode_steps : int
        Max steps per episode (default: 256).
    seeds : list[int], optional
        Seeds for each episode. If None, uses 0..num_episodes-1.
    device : str
        Torch device (default: "cpu").

    Returns
    -------
    episodes : list[dict]
        Per-episode data dicts sorted by return (worst to best).
    metrics : dict
        Aggregate metrics across all episodes.
    """
    import train.ppo.train_ppo_optomech as _mod
    _mod._ENV_ID = f"optomech-{env_version}"

    if env_kwargs is None:
        env_kwargs = dict(NANOELF_TT_ENV_KWARGS)
    else:
        env_kwargs = dict(env_kwargs)
    env_kwargs["max_episode_steps"] = max_episode_steps

    register_optomech(f"optomech-{env_version}", max_episode_steps=max_episode_steps)
    env = gym.make(f"optomech-{env_version}", **env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent, config, obs_ref_max = load_agent(checkpoint_path, env, device)

    if seeds is None:
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 2**31, size=num_episodes).tolist()

    print(f"Running {num_episodes} rollouts (device={device})...")
    episodes = []
    for i, seed in enumerate(seeds[:num_episodes]):
        t0 = time.time()
        ep = run_single_episode(agent, env, seed, obs_ref_max, device)
        elapsed = time.time() - t0
        print(f"  Episode {i+1}/{num_episodes}: seed={seed}, "
              f"return={ep['return']:.4f}, strehl_final={ep['strehls'][-1]:.4f}, "
              f"gap={ep['improvement_gap']:+.4f} ({elapsed:.1f}s)")
        episodes.append(ep)

    env.close()

    # Sort by return (worst first)
    episodes.sort(key=lambda e: e["return"])

    # Aggregate metrics
    returns = [e["return"] for e in episodes]
    zero_returns = [e["zero_return"] for e in episodes]
    final_strehls = [e["strehls"][-1] for e in episodes]
    gaps = [e["improvement_gap"] for e in episodes]

    metrics = {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "mean_zero_return": float(np.mean(zero_returns)),
        "mean_improvement_gap": float(np.mean(gaps)),
        "mean_final_strehl": float(np.mean(final_strehls)),
        "std_final_strehl": float(np.std(final_strehls)),
        "median_final_strehl": float(np.median(final_strehls)),
    }

    return episodes, metrics


# ============================================================================
# GIF rendering
# ============================================================================


def render_episode_gif(ep_data, save_path, label="", dpi=72, frame_duration=0.2):
    """Render a single episode as an animated GIF."""
    obs_raw = ep_data["obs_raw"]
    rewards = ep_data["rewards"]
    actions = np.array(ep_data["actions"])
    strehls = ep_data.get("strehls", [])
    cumulative = np.cumsum(rewards)
    T = len(rewards)
    action_dim = actions.shape[1] if len(actions) > 0 else 0

    all_raw = [_prepare_obs_raw(obs_raw[t]) for t in range(T + 1)]
    global_max = max(float(np.max(img)) for img in all_raw)
    global_max = max(global_max, 1.0)
    norm = mcolors.LogNorm(vmin=1.0, vmax=global_max)

    frames = []
    for t in range(T + 1):
        fig = plt.figure(figsize=(5, 5.5), dpi=dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[3, 1], hspace=0.35, wspace=0.4)

        ax_obs = fig.add_subplot(gs[0, :])
        img_dn = all_raw[t]
        im = ax_obs.imshow(np.maximum(img_dn, 1.0), cmap="inferno", norm=norm)
        ax_obs.axis("off")
        dn_max = float(np.max(img_dn))
        dn_sum = float(np.sum(img_dn))
        ax_obs.set_title(
            f"{label} | seed={ep_data['seed']} | R={ep_data['return']:.3f}"
            f"\nmax={dn_max:.0f}  sum={dn_sum:.0f}",
            fontsize=7)
        fig.colorbar(im, ax=ax_obs, fraction=0.046, pad=0.04, label="DN")

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
        else:
            ax_act.text(0.5, 0.5, "t=0\n(initial)", ha="center",
                        va="center", fontsize=7, transform=ax_act.transAxes)
            ax_act.set_xticks([])
            ax_act.set_yticks([])
        ax_act.set_title("Action", fontsize=7)

        ax_txt = fig.add_subplot(gs[1, 1])
        ax_txt.axis("off")
        if t == 0:
            text = f"t = 0 / {T}\n(initial obs)\nmax DN = {dn_max:.0f}\nsum DN = {dn_sum:.0f}"
        else:
            r = rewards[t - 1]
            c = cumulative[t - 1]
            s_txt = f"\nS = {strehls[t-1]:.4f}" if strehls else ""
            text = (f"t = {t} / {T}\nr = {r:.4f}  R = {c:.3f}{s_txt}"
                    f"\nmax DN = {dn_max:.0f}\nsum DN = {dn_sum:.0f}")
        ax_txt.text(0.5, 0.5, text, ha="center", va="center",
                    fontsize=7, fontfamily="monospace",
                    transform=ax_txt.transAxes)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=frame_duration)


def save_episode_gifs(episodes, output_dir):
    """Save best, worst, and median episode GIFs."""
    os.makedirs(output_dir, exist_ok=True)
    worst = episodes[0]
    median = episodes[len(episodes) // 2]
    best = episodes[-1]

    for ep, label in [(best, "best"), (worst, "worst"), (median, "median")]:
        path = os.path.join(output_dir, f"{label}.gif")
        render_episode_gif(ep, path, label=label.upper())
        print(f"  GIF: {path}")


# ============================================================================
# Summary figures (individual publication-quality plots)
# ============================================================================


def save_summary_figures(episodes, metrics, output_dir):
    """Save individual summary figures suitable for papers.

    Each figure is saved as both PNG and PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    _fig_reward_traces(episodes, output_dir)
    _fig_cumulative_returns(episodes, output_dir)
    _fig_strehl_traces(episodes, output_dir)
    _fig_action_magnitudes(episodes, output_dir)
    _fig_return_distribution(episodes, metrics, output_dir)
    _fig_final_strehl_distribution(episodes, metrics, output_dir)
    print(f"  Figures saved to {output_dir}")


def _savefig(fig, output_dir, name):
    """Save figure as PNG and PDF."""
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _pad_and_stack(traces):
    """Pad variable-length traces to uniform length and stack."""
    max_len = max(len(t) for t in traces)
    padded = np.zeros((len(traces), max_len), dtype=np.float32)
    for i, t in enumerate(traces):
        arr = np.array(t, dtype=np.float32)
        padded[i, :len(arr)] = arr
        if len(arr) < max_len:
            padded[i, len(arr):] = arr[-1]
    return padded


def _fig_reward_traces(episodes, output_dir):
    """Step-wise reward: agent vs zero-action vs random, mean +/- std."""
    agent_mat = _pad_and_stack([e["rewards"] for e in episodes])
    zero_mat = _pad_and_stack([e["zero_rewards"] for e in episodes])
    random_mat = _pad_and_stack([e["random_rewards"] for e in episodes])
    T = agent_mat.shape[1]
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    for mat, color, label in [
        (agent_mat, "forestgreen", "Agent"),
        (zero_mat, "gray", "Zero-action"),
        (random_mat, "coral", "Random"),
    ]:
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        ax.plot(ts, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Step-wise Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "reward_traces")


def _fig_cumulative_returns(episodes, output_dir):
    """Cumulative return: agent vs zero vs random."""
    agent_mat = _pad_and_stack([e["rewards"] for e in episodes])
    zero_mat = _pad_and_stack([e["zero_rewards"] for e in episodes])
    random_mat = _pad_and_stack([e["random_rewards"] for e in episodes])
    T = agent_mat.shape[1]
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    for mat, color, label in [
        (np.cumsum(agent_mat, axis=1), "forestgreen", "Agent"),
        (np.cumsum(zero_mat, axis=1), "gray", "Zero-action"),
        (np.cumsum(random_mat, axis=1), "coral", "Random"),
    ]:
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        ax.plot(ts, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "cumulative_returns")


def _fig_strehl_traces(episodes, output_dir):
    """Strehl ratio over time: individual + mean."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for ep in episodes:
        ax.plot(ep["strehls"], color="steelblue", alpha=0.2, linewidth=0.8)

    mat = _pad_and_stack([e["strehls"] for e in episodes])
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ts = np.arange(mat.shape[1])
    ax.plot(ts, mean, color="darkblue", linewidth=2.0, label="Mean")
    ax.fill_between(ts, mean - std, mean + std, color="steelblue", alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Strehl Ratio")
    ax.set_title("Strehl Ratio Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "strehl_traces")


def _fig_action_magnitudes(episodes, output_dir):
    """Mean action magnitude per DOF over time."""
    all_actions = [np.array(e["actions"]) for e in episodes]
    mat = _pad_and_stack([np.mean(np.abs(a), axis=1) for a in all_actions])
    T = mat.shape[1]
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(ts, mean, color="purple", linewidth=1.5, label="Mean |action|")
    ax.fill_between(ts, mean - std, mean + std, color="purple", alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean |Action|")
    ax.set_title("Action Magnitude Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "action_magnitudes")


def _fig_return_distribution(episodes, metrics, output_dir):
    """Histogram of episode returns."""
    returns = [e["return"] for e in episodes]
    zero_returns = [e["zero_return"] for e in episodes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(returns, bins=max(len(returns) // 2, 5), color="forestgreen",
            alpha=0.7, edgecolor="black", label="Agent")
    ax.axvline(metrics["mean_return"], color="darkgreen", linestyle="--",
               linewidth=2, label=f"Mean = {metrics['mean_return']:.3f}")
    ax.axvline(np.mean(zero_returns), color="gray", linestyle=":",
               linewidth=2, label=f"Zero = {np.mean(zero_returns):.3f}")
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.set_title("Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "return_distribution")


def _fig_final_strehl_distribution(episodes, metrics, output_dir):
    """Histogram of final Strehl ratios."""
    strehls = [e["strehls"][-1] for e in episodes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(strehls, bins=max(len(strehls) // 2, 5), color="steelblue",
            alpha=0.7, edgecolor="black")
    ax.axvline(metrics["mean_final_strehl"], color="darkblue", linestyle="--",
               linewidth=2, label=f"Mean = {metrics['mean_final_strehl']:.4f}")
    ax.set_xlabel("Final Strehl Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Final Strehl Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "final_strehl_distribution")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO agent with rollout instrumentation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--env-version", type=str, default="v3",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip GIF generation")
    args = parser.parse_args()

    # Create timestamped test directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp}_{int(time.time()) % 10000}"
    test_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output directory: {test_dir}")

    # Run rollouts
    episodes, metrics = run_rollouts(
        checkpoint_path=args.checkpoint,
        env_version=args.env_version,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Episodes: {metrics['num_episodes']}")
    print(f"  Mean return:  {metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"  Median return: {metrics['median_return']:.4f}")
    print(f"  Best / Worst:  {metrics['best_return']:.4f} / {metrics['worst_return']:.4f}")
    print(f"  Zero baseline: {metrics['mean_zero_return']:.4f}")
    print(f"  Mean gap:      {metrics['mean_improvement_gap']:+.4f}")
    print(f"  Mean final Strehl: {metrics['mean_final_strehl']:.4f} "
          f"+/- {metrics['std_final_strehl']:.4f}")
    print(f"{'='*60}")

    # Save metrics
    metrics_path = os.path.join(test_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")

    # GIFs
    if not args.no_gifs:
        save_episode_gifs(episodes, os.path.join(test_dir, "gifs"))

    # Summary figures
    save_summary_figures(episodes, metrics, os.path.join(test_dir, "figures"))

    print(f"\nAll outputs in: {test_dir}")


if __name__ == "__main__":
    main()
