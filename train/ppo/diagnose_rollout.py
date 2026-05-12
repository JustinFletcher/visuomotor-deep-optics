"""Per-step structural diagnostic for a bilateral-DM rollout.

Loads a checkpoint, builds the env from its stored env_kwargs
(exact training-time config), and runs a single deterministic
rollout printing every quantity that could plausibly differ
between training and eval:

    step  |action|_L1 |action|_max  S       reward  dm_max  obs_mean

Then runs the SAME rollout under explicit re-application of any
curriculum-driven reward-weight attributes at the checkpoint's
step, so the env's reward weights match what training was using
at save time. (Reward weights do not affect Strehl directly, but
they do enter prior_reward fed back into the LSTM next step.)

Run on HPC where the checkpoints live:

    poetry run python train/ppo/diagnose_rollout.py \\
        <path/to/checkpoints/history_step_NNN.pt> --target-id N
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.ppo.bilateral_dm import BilateralDMVectorEnv                 # noqa: E402
from train.ppo.launch_static_dark_hole import build_grid                # noqa: E402
from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper  # noqa: E402
from train.ppo.train_ppo_optomech import normalize_obs_fixed            # noqa: E402


def _build_env_for_checkpoint(ckpt_cfg, target, max_steps, device):
    """Build env matching the checkpoint's stored env_kwargs exactly."""
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv

    env_kwargs = dict(ckpt_cfg.get("env_kwargs", {}))
    angle, r_frac, s_frac = target
    env_kwargs["dark_hole"] = True
    env_kwargs["dark_hole_angular_location_degrees"] = float(angle)
    env_kwargs["dark_hole_location_radius_fraction"] = float(r_frac)
    env_kwargs["dark_hole_size_radius"] = float(s_frac)
    env_kwargs["dark_hole_randomize_on_reset"] = False
    env_kwargs["max_episode_steps"] = int(max_steps) + 1
    env_kwargs["silence"] = True
    env_kwargs["observation_window_size"] = 1
    env_kwargs["reward_vector_enabled"] = False

    bilateral = bool(ckpt_cfg.get("bilateral_dm", True))
    bilateral_mode = str(ckpt_cfg.get("bilateral_dm_mode", "fixed_vertical"))
    freeze_segments = bool(ckpt_cfg.get("bilateral_freeze_segments", True))

    with contextlib.redirect_stdout(io.StringIO()):
        base = BatchedOptomechEnv(num_envs=1, device=device, **env_kwargs)
        env = (BilateralDMVectorEnv(base, freeze_segments=freeze_segments,
                                    mode=bilateral_mode)
               if bilateral else base)
    return env, base, bilateral


def _apply_curriculum_at_step(env, base, ckpt_cfg, global_step):
    """If the checkpoint's config has a reward_weight_curriculum,
    compute the curriculum-driven weight at global_step and apply it
    to the v5 env. Walks through wrappers via vars()."""
    cur = ckpt_cfg.get("reward_weight_curriculum")
    if cur is None:
        cur = ckpt_cfg.get("dark_hole_curriculum")
    if cur is None:
        return None, None
    attr = str(cur.get("attr", "_rw_dark_hole"))
    warmup = int(cur.get("warmup_timesteps", 0))
    anneal = int(cur.get("anneal_timesteps", 1))
    start = float(cur.get("start_value", 0.0))
    end = float(cur.get("end_value", 1.0))
    if global_step < warmup:
        v = start
    else:
        progress = min((global_step - warmup) / max(anneal, 1), 1.0)
        v = start + progress * (end - start)
    b = env
    while hasattr(b, "_env") and attr not in vars(b):
        b = b._env
    if attr in vars(b):
        setattr(b, attr, v)
        return attr, v
    return attr, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--target-id", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions instead of using deterministic mean.")
    args = parser.parse_args()

    targets = build_grid()
    target = targets[args.target_id]

    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    global_step = int(ckpt.get("global_step", 0))

    print(f"checkpoint:   {args.checkpoint}")
    print(f"global_step:  {global_step:,}")
    print(f"target:       id={args.target_id}  angle={target[0]}  "
          f"r={target[1]}  size={target[2]}")
    print(f"bilateral_dm: {ckpt_cfg.get('bilateral_dm', True)}  "
          f"mode={ckpt_cfg.get('bilateral_dm_mode', 'fixed_vertical')}  "
          f"freeze_seg={ckpt_cfg.get('bilateral_freeze_segments', True)}")

    # log_std / sigma envelope
    sd = ckpt["model_state_dict"]
    if "log_std" in sd:
        ls = sd["log_std"]
        print(f"log_std:      mean={ls.mean().item():.3f}  "
              f"min={ls.min().item():.3f}  max={ls.max().item():.3f}  "
              f"sigma_mean={ls.exp().mean().item():.4f}")

    env, base, bilateral = _build_env_for_checkpoint(
        ckpt_cfg, target, args.max_steps, args.device)

    attr, val = _apply_curriculum_at_step(env, base, ckpt_cfg, global_step)
    if attr:
        print(f"curriculum:   {attr} -> {val} at step {global_step:,}")

    # Action / obs space sanity
    print(f"action space: {env.single_action_space.shape}  "
          f"obs space: {env.single_observation_space.shape}")

    class _Shim:
        single_observation_space = env.single_observation_space
        single_action_space = env.single_action_space

    agent = RecurrentActorCritic(
        _Shim(), torch.device(args.device),
        lstm_hidden_dim=ckpt_cfg.get("lstm_hidden_dim", 128),
        channel_scale=ckpt_cfg.get("channel_scale", 32),
        fc_scale=ckpt_cfg.get("fc_scale", 256),
        action_scale=ckpt_cfg.get("action_scale", 1.0),
        init_log_std=ckpt_cfg.get("init_log_std", -0.5),
        model_type=ckpt_cfg.get("model_type", "small"),
        target_dim=int(ckpt_cfg.get("target_dim", 0)),
        log_std_max=ckpt_cfg.get("log_std_max", None),
        log_std_min=ckpt_cfg.get("log_std_min", None),
    ).to(args.device)
    agent.load_state_dict(sd)
    agent.eval()
    wrapper = PPOActorWrapper(agent)

    obs_ref_max = float(getattr(base, "_reference_fpi_max", 1.0))
    print(f"obs_ref_max:  {obs_ref_max:.3f}")

    obs, _ = env.reset(seed=args.seed)
    if hasattr(env, "mask_obs"):
        obs = env.mask_obs(obs)
    obs_norm = normalize_obs_fixed(obs, obs_ref_max)

    angle, r_frac, s_frac = target
    th = np.deg2rad(angle)
    tv = torch.tensor(
        [[np.sin(th), np.cos(th), r_frac, s_frac]],
        dtype=torch.float32, device=args.device)
    h = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim,
                    device=args.device)
    c = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim,
                    device=args.device)
    prior_action = torch.zeros(1, agent.action_dim, device=args.device)
    prior_reward = torch.zeros(1, device=args.device)

    if args.stochastic:
        torch.manual_seed(args.seed)
    print(f"\nmode: {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"{'step':>4}  {'|a|_L1':>10} {'|a|_max':>10}  "
          f"{'dm_max':>10}  {'obs_mean':>10}  {'env_S':>8}  "
          f"{'reward':>10}")

    for step in range(args.max_steps):
        obs_t = torch.from_numpy(obs_norm).float().to(args.device)
        with torch.no_grad():
            if args.stochastic:
                action_t, _, _, _, (h, c) = agent.get_action_and_value(
                    obs_t, prior_action, prior_reward, (h, c),
                    target_vec=tv)
                action_t = agent.scale_and_clamp_action(action_t)
            else:
                action_t, (h, c) = wrapper(
                    obs_t, prior_action, prior_reward, (h, c), target_vec=tv)
        a_np = action_t.detach().cpu().numpy()
        obs_unmasked, rew, term, trunc, info = env.step(a_np)
        if hasattr(env, "mask_obs"):
            obs_for_policy = env.mask_obs(obs_unmasked)
        else:
            obs_for_policy = obs_unmasked
        obs_norm = normalize_obs_fixed(obs_for_policy, obs_ref_max)

        dm_max = float(base._dm_actuators_t.abs().max().item()) \
            if base._dm_actuators_t is not None else 0.0
        obs_mean = float(obs_unmasked.mean())
        env_S = float(info["strehl"][0])
        r = float(rew[0])
        if step < 8 or step % 8 == 7 or step == args.max_steps - 1:
            print(f"{step:>4d}  {float(np.abs(a_np).mean()):>10.5f} "
                  f"{float(np.abs(a_np).max()):>10.5f}  "
                  f"{dm_max:>10.3e}  {obs_mean:>10.3f}  "
                  f"{env_S:>8.4f}  {r:>+10.4f}")
        prior_action = action_t
        prior_reward = torch.tensor([r], dtype=torch.float32,
                                    device=args.device)


if __name__ == "__main__":
    main()
