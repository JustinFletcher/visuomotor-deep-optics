"""Eval rollout for full-DM strehl-only models.

Companion to ``train_ppo_elf_dm_strehl_only_full.py`` (trained via
``launch_dm_strehl_only_full.py``). Runs N independent episodes from
random resets, reports Strehl trajectory + summary stats, and writes
a JSON + a summary figure + a median-episode GIF.

Reuses the same env-from-checkpoint convention as
``rollout_bilateral_dm_grid.py``: env_kwargs come from
``ckpt["config"]["env_kwargs"]`` so the eval env exactly matches the
training env. No bilateral wrapper (cfg["bilateral_dm"] is False for
these runs).

Usage:
    # Single run, deterministic, 8 episodes:
    poetry run python train/ppo/rollout_dm_strehl_only_full.py \
        --checkpoint dark_hole_runs/<run_id>/seed_<S>/checkpoints/latest.pt

    # 32 episodes with both deterministic and stochastic traces:
    poetry run python train/ppo/rollout_dm_strehl_only_full.py \
        --checkpoint .../latest.pt --num-episodes 32 --stochastic both

    # All seeds in a run directory (one summary per seed):
    poetry run python train/ppo/rollout_dm_strehl_only_full.py \
        --run-dir dark_hole_runs/dm_strehl_only_full_<timestamp>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.ppo.ppo_models import RecurrentActorCritic  # noqa: E402
from train.ppo.train_ppo_optomech import normalize_obs_fixed  # noqa: E402


# ---------------------------------------------------------------------------
# Env-shim used to satisfy RecurrentActorCritic's constructor (it pulls
# single_observation_space / single_action_space from the env).
# ---------------------------------------------------------------------------
class _EnvShim:
    def __init__(self, env):
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _build_env(base_env_kwargs, max_steps, device):
    """v5 single-env, no bilateral wrapper, no dark hole.

    Mirrors the training-time env: same env_kwargs (pulled from the
    checkpoint's stored config), max_episode_steps overridden to
    max_steps + 1 so the loop can stop one step short of the env's
    auto-reset and never see a reset-transient terminal frame.
    """
    kw = dict(base_env_kwargs)
    kw["max_episode_steps"] = int(max_steps) + 1
    kw["silence"] = True
    kw["observation_window_size"] = 1
    kw["reward_vector_enabled"] = False
    # Defensive: these runs train with the wrapper OFF; if a stale
    # env_kwarg slipped in, force them off here too.
    kw["dark_hole"] = False

    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    with contextlib.redirect_stdout(io.StringIO()):
        env = BatchedOptomechEnv(num_envs=1, device=device, **kw)
    return env


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
    return agent, config, ckpt


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(agent, env, seed, device, max_steps, stochastic=False):
    """Single deterministic (or stochastic) episode.

    Returns dict with strehl trajectory, reward trajectory, action
    trajectory and pre-aggregated summary scalars. No image arrays
    are retained when --no-gifs is set on the CLI.
    """
    if stochastic:
        torch.manual_seed(int(seed))

    obs_ref_max = float(env._reference_fpi_max)
    obs_np, _ = env.reset(seed=seed)
    obs_norm = normalize_obs_fixed(obs_np, obs_ref_max)

    h = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(1, agent.action_dim, device=device)
    prior_reward = torch.zeros(1, device=device)

    strehls = []
    rewards = []
    actions = []
    obs_frames = []  # kept for --gifs; one [H, W] frame per step

    obs_frames.append(obs_np[0, 0].copy())

    steps_taken = 0
    done = False
    while not done and steps_taken < max_steps:
        obs_t = torch.from_numpy(obs_norm).float().to(device)
        with torch.no_grad():
            if stochastic:
                a_t, _, _, _, (h, c) = agent.get_action_and_value(
                    obs_t, prior_action, prior_reward, (h, c))
                a_t = agent.scale_and_clamp_action(a_t)
            else:
                a_t, (h, c) = agent.get_deterministic_action(
                    obs_t, prior_action, prior_reward, (h, c))
        a_np = a_t.detach().cpu().numpy()
        nxt, rew, term, trunc, info = env.step(a_np)
        steps_taken += 1
        done = bool(term[0] or trunc[0])
        obs_frames.append(nxt[0, 0].copy())
        rewards.append(float(rew[0]))
        actions.append(a_np[0].copy())
        if "strehl" in info:
            strehls.append(float(info["strehl"][0]))
        obs_norm = normalize_obs_fixed(nxt, obs_ref_max)
        prior_action = a_t
        prior_reward = torch.tensor(
            [rew[0]], dtype=torch.float32, device=device)

    return {
        "strehls": strehls,
        "rewards": rewards,
        "actions": np.array(actions),
        "obs_frames": obs_frames,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _save_summary_figure(episodes, out_path, stochastic_label):
    """Strehl trajectory across episodes + per-episode endpoint scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: per-episode Strehl trajectory.
    ax = axes[0]
    all_strehls = np.array([ep["strehls"] for ep in episodes])
    steps = np.arange(all_strehls.shape[1])
    for ep in episodes:
        ax.plot(steps, ep["strehls"], color="#888", alpha=0.35, linewidth=0.8)
    mean = all_strehls.mean(axis=0)
    median = np.median(all_strehls, axis=0)
    ax.plot(steps, mean, color="C0", linewidth=2.0, label="mean")
    ax.plot(steps, median, color="C1", linewidth=2.0,
            linestyle="--", label="median")
    ax.set_xlabel("step")
    ax.set_ylabel("Strehl")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Strehl trajectory ({stochastic_label})")
    ax.grid(True, alpha=0.3)

    # Right: histogram of final Strehl.
    ax = axes[1]
    finals = all_strehls[:, -1]
    ax.hist(finals, bins=20, color="C2", alpha=0.7, edgecolor="k")
    ax.axvline(finals.mean(), color="k", linestyle="--",
               label=f"mean={finals.mean():.3f}")
    ax.axvline(np.median(finals), color="C1", linestyle=":",
               label=f"median={np.median(finals):.3f}")
    ax.set_xlabel("final Strehl")
    ax.set_ylabel("episodes")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Final-step distribution")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_median_gif(episodes, out_path, frame_duration=80):
    """Save the median-final-Strehl episode as an animated GIF."""
    import imageio
    finals = np.array([ep["strehls"][-1] for ep in episodes])
    median_idx = int(np.argsort(finals)[len(finals) // 2])
    ep = episodes[median_idx]
    # Normalise frames to [0, 255] uint8 for GIF encoding.
    frames = []
    fmax = max(float(np.max(f)) for f in ep["obs_frames"])
    if fmax <= 0.0:
        fmax = 1.0
    for f in ep["obs_frames"]:
        ff = (np.clip(f / fmax, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(ff)
    imageio.mimsave(out_path, frames, duration=frame_duration / 1000.0,
                    loop=0)


# ---------------------------------------------------------------------------
# Per-checkpoint driver
# ---------------------------------------------------------------------------

def run_one_checkpoint(ckpt_path, args):
    print(f"\n=== {ckpt_path} ===")
    env = _build_env(
        base_env_kwargs=_pull_env_kwargs(ckpt_path),
        max_steps=args.max_steps, device=args.device)
    agent, config, _ckpt = _load_agent(ckpt_path, env, args.device)
    print(f"  global_step={config.get('global_step', '?')}  "
          f"action_dim={agent.action_dim}  "
          f"target_dim={int(config.get('target_dim', 0))}")

    modes = []
    if args.stochastic in ("det", "both"):
        modes.append(("det", False))
    if args.stochastic in ("stoch", "both"):
        modes.append(("stoch", True))

    output_dir = os.path.dirname(ckpt_path)
    out_root = os.path.join(
        output_dir, "..",
        f"eval_{os.path.basename(ckpt_path).rsplit('.', 1)[0]}")
    out_root = os.path.normpath(out_root)
    os.makedirs(out_root, exist_ok=True)

    all_summaries = {}
    for label, stoch in modes:
        episodes = []
        # Per-episode seeds: linear from --seed so reruns are
        # reproducible. With --seed 0 + 8 episodes: seeds 0..7.
        for k in range(args.num_episodes):
            ep = run_episode(
                agent, env, seed=args.seed + k, device=args.device,
                max_steps=args.max_steps, stochastic=stoch)
            episodes.append(ep)
        env.close()
        # Re-build env between modes so each mode starts clean (the v5
        # env carries state across episodes via _dm_actuators_t at
        # reset; safer to discard).
        if (label, stoch) != modes[-1]:
            env = _build_env(
                base_env_kwargs=_pull_env_kwargs(ckpt_path),
                max_steps=args.max_steps, device=args.device)

        # Aggregate scalars.
        all_strehls = np.array([ep["strehls"] for ep in episodes])
        per_ep_mean = all_strehls.mean(axis=1)
        per_ep_final = all_strehls[:, -1]
        per_step_mean = all_strehls.mean(axis=0)
        summary = {
            "mode": label,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "mean_strehl_over_episodes": float(per_ep_mean.mean()),
            "mean_final_strehl": float(per_ep_final.mean()),
            "median_final_strehl": float(np.median(per_ep_final)),
            "min_final_strehl": float(per_ep_final.min()),
            "max_final_strehl": float(per_ep_final.max()),
            "per_step_mean_strehl": per_step_mean.tolist(),
            "per_episode_final_strehl": per_ep_final.tolist(),
        }
        all_summaries[label] = summary

        print(f"  [{label:>5}]  mean(strehl)  per-episode={per_ep_mean.mean():.4f}"
              f"  final-mean={per_ep_final.mean():.4f}"
              f"  final-median={np.median(per_ep_final):.4f}"
              f"  range=[{per_ep_final.min():.4f}, {per_ep_final.max():.4f}]")

        # Figures + GIFs.
        if not args.no_figures:
            fig_path = os.path.join(out_root, f"strehl_{label}.png")
            _save_summary_figure(episodes, fig_path, label)
            print(f"  -> {fig_path}")
        if not args.no_gifs:
            gif_path = os.path.join(out_root, f"median_{label}.gif")
            _save_median_gif(episodes, gif_path,
                             frame_duration=args.frame_duration)
            print(f"  -> {gif_path}")

    # JSON summary.
    json_path = os.path.join(out_root, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "global_step": int(config.get("global_step", 0)),
            "modes": all_summaries,
        }, f, indent=2)
    print(f"  -> {json_path}")
    env.close()


def _pull_env_kwargs(ckpt_path):
    """Open the checkpoint just for its env_kwargs (light, no model load)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ek = ck["config"].get("env_kwargs", None)
    if ek is None:
        raise RuntimeError(
            f"{ckpt_path} has no env_kwargs in its config; cannot "
            f"reconstruct the training env.")
    return ek


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _resolve_checkpoints(args):
    """Either --checkpoint, or every seed_*/checkpoints/latest.pt under --run-dir."""
    if args.checkpoint:
        return [args.checkpoint]
    if not args.run_dir:
        print("Error: provide --checkpoint or --run-dir.")
        sys.exit(1)
    found = []
    for seed_dir in sorted(Path(args.run_dir).glob("seed_*")):
        ckpts_dir = seed_dir / "checkpoints"
        if not ckpts_dir.exists():
            continue
        latest = ckpts_dir / "latest.pt"
        if latest.exists():
            found.append(str(latest))
        else:
            # Fall back to the highest-numbered checkpoint.
            cands = sorted(ckpts_dir.glob("update_*.pt"))
            if cands:
                found.append(str(cands[-1]))
    if not found:
        print(f"No checkpoints found under {args.run_dir}")
        sys.exit(1)
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _device_default():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    p = argparse.ArgumentParser(
        description="Eval rollout for full-DM strehl-only models.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single checkpoint (.pt).")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Run dir containing seed_*/checkpoints/latest.pt.")
    p.add_argument("--num-episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0,
                   help="Base env seed; episode k uses seed+k.")
    p.add_argument("--stochastic", choices=["det", "stoch", "both"],
                   default="det",
                   help="Action mode: deterministic mean, stochastic "
                        "sample, or both (default: det).")
    p.add_argument("--device", type=str, default=_device_default())
    p.add_argument("--no-figures", action="store_true",
                   help="Skip matplotlib summary figures.")
    p.add_argument("--no-gifs", action="store_true",
                   help="Skip per-mode median-episode GIF.")
    p.add_argument("--frame-duration", type=int, default=80,
                   help="GIF frame duration (ms).")
    args = p.parse_args()

    ckpts = _resolve_checkpoints(args)
    print(f"Evaluating {len(ckpts)} checkpoint(s) on {args.device}; "
          f"{args.num_episodes} episodes x {args.max_steps} steps each.")
    t0 = time.time()
    for ckpt_path in ckpts:
        run_one_checkpoint(ckpt_path, args)
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
