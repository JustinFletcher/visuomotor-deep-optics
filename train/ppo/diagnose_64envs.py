"""Replicate the training-time rollout shape: num_envs=64 with the
same env_kwargs and the same policy weights, and report per-env Strehl
across one 64-step episode. If a substantial fraction of envs reach
Strehl ~0.85 (matching TB), the batched forward path is doing
something the singleton path isn't.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.ppo.bilateral_dm import BilateralDMVectorEnv                 # noqa: E402
from train.ppo.launch_static_dark_hole import build_grid                # noqa: E402
from train.ppo.ppo_models import RecurrentActorCritic                   # noqa: E402
from train.ppo.train_ppo_optomech import normalize_obs_fixed            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    targets = build_grid()
    target = targets[args.target_id]
    angle, r_frac, s_frac = target

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    sd = ckpt["model_state_dict"]
    global_step = int(ckpt.get("global_step", 0))

    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    env_kwargs = dict(ckpt_cfg.get("env_kwargs", {}))
    env_kwargs["dark_hole"] = True
    env_kwargs["dark_hole_angular_location_degrees"] = float(angle)
    env_kwargs["dark_hole_location_radius_fraction"] = float(r_frac)
    env_kwargs["dark_hole_size_radius"] = float(s_frac)
    env_kwargs["dark_hole_randomize_on_reset"] = False
    # Honour the checkpoint's max_episode_steps so auto-reset fires at
    # the same cadence as training (every 64 steps).
    env_kwargs["silence"] = True
    env_kwargs["observation_window_size"] = 1
    env_kwargs["reward_vector_enabled"] = False

    bilateral = bool(ckpt_cfg.get("bilateral_dm", True))
    bilateral_mode = str(ckpt_cfg.get("bilateral_dm_mode", "fixed_vertical"))
    freeze_segments = bool(ckpt_cfg.get("bilateral_freeze_segments", True))

    with contextlib.redirect_stdout(io.StringIO()):
        base = BatchedOptomechEnv(num_envs=args.num_envs, device=args.device,
                                  **env_kwargs)
        env = (BilateralDMVectorEnv(base, freeze_segments=freeze_segments,
                                    mode=bilateral_mode)
               if bilateral else base)

    # Apply curriculum to the v5 env
    cur = ckpt_cfg.get("reward_weight_curriculum")
    if cur:
        attr = str(cur.get("attr", "_rw_log_contrast_strehl"))
        warmup = int(cur.get("warmup_timesteps", 0))
        anneal = int(cur.get("anneal_timesteps", 1))
        start = float(cur.get("start_value", 0.0))
        end = float(cur.get("end_value", 1.0))
        progress = min(max(global_step - warmup, 0) / max(anneal, 1), 1.0)
        v = start + progress * (end - start)
        if attr in vars(base):
            setattr(base, attr, v)

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

    obs_ref_max = float(getattr(base, "_reference_fpi_max", 1.0))
    obs, _ = env.reset(seed=args.seed)
    if hasattr(env, "mask_obs"):
        obs = env.mask_obs(obs)
    obs_norm = normalize_obs_fixed(obs, obs_ref_max)

    N = args.num_envs
    h = torch.zeros(agent.lstm_num_layers, N, agent.lstm_hidden_dim, device=args.device)
    c = torch.zeros(agent.lstm_num_layers, N, agent.lstm_hidden_dim, device=args.device)
    prior_action = torch.zeros(N, agent.action_dim, device=args.device)
    prior_reward = torch.zeros(N, device=args.device)

    th = np.deg2rad(angle)
    tv = torch.tensor([[np.sin(th), np.cos(th), r_frac, s_frac]],
                      dtype=torch.float32, device=args.device).expand(N, -1)

    print(f"num_envs={N}  max_steps={args.max_steps}  "
          f"stochastic={args.stochastic}  global_step={global_step:,}")
    print(f"{'step':>4}  {'S_mean':>8}  {'S_min':>8}  {'S_max':>8}  "
          f"{'S_median':>10}  {'frac_S>0.5':>11}  {'frac_S>0.8':>11}")

    if args.stochastic:
        torch.manual_seed(args.seed)

    S_trace = []
    for step in range(args.max_steps):
        obs_t = torch.from_numpy(obs_norm).float().to(args.device)
        with torch.no_grad():
            lstm_out, (h, c) = agent._forward_shared(
                obs_t, prior_action, prior_reward, (h, c), target_vec=tv)
            raw_mu = agent.policy_head(lstm_out)
            if args.stochastic:
                from torch.distributions import Normal
                std = agent.log_std.exp().expand_as(raw_mu)
                action_raw = Normal(raw_mu, std).sample()
            else:
                action_raw = raw_mu
            action_t = agent.scale_and_clamp_action(action_raw)
        a_np = action_t.detach().cpu().numpy()
        obs_unmasked, rew, term, trunc, info = env.step(a_np)
        if hasattr(env, "mask_obs"):
            obs_for_policy = env.mask_obs(obs_unmasked)
        else:
            obs_for_policy = obs_unmasked
        obs_norm = normalize_obs_fixed(obs_for_policy, obs_ref_max)

        S = np.asarray(info["strehl"], dtype=np.float32)
        S_trace.append(S)
        prior_action = action_t.detach()
        prior_reward = torch.from_numpy(np.asarray(rew, dtype=np.float32)).to(args.device)

        # Mirror training's done-mask handling: zero lstm h/c, prior_action,
        # prior_reward for envs that just reset. The env emitted post-reset
        # frames into `obs_unmasked` already (v5 auto-reset behaviour).
        dones = np.logical_or(np.asarray(term), np.asarray(trunc))
        if dones.any():
            done_t = torch.from_numpy(dones.astype(bool))
            h[:, done_t, :] = 0.0
            c[:, done_t, :] = 0.0
            prior_action[done_t] = 0.0
            prior_reward[done_t] = 0.0

        if step < 8 or step % 8 == 7 or step == args.max_steps - 1:
            print(f"{step:>4d}  {S.mean():>8.4f}  {S.min():>8.4f}  "
                  f"{S.max():>8.4f}  {np.median(S):>10.4f}  "
                  f"{(S > 0.5).mean():>11.3f}  {(S > 0.8).mean():>11.3f}")

    S_all = np.stack(S_trace)  # [max_steps, N]
    print(f"\nOverall mean over all (env, step): {S_all.mean():.4f}")
    print(f"Per-env final-step Strehl: min={S_all[-1].min():.3f}  "
          f"max={S_all[-1].max():.3f}  mean={S_all[-1].mean():.3f}")
    print(f"Best 3 envs (final Strehl): "
          f"{sorted(S_all[-1].tolist(), reverse=True)[:3]}")


if __name__ == "__main__":
    main()
