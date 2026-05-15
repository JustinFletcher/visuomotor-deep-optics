"""Ablation rollouts to localize the train-eval mismatch.

For the same checkpoint, run several rollouts with one factor swapped
to its "step-0-like" value and see which one keeps Strehl high:

  baseline       : run as the policy was trained
  zero_hidden    : force LSTM (h, c) = 0 each step (kills recurrence)
  zero_prior_act : force prior_action = 0 each step
  zero_prior_rew : force prior_reward = 0 each step
  zero_target    : force target_vec = 0 each step (mimic step-0 of training startup)
  raw_mean       : print raw policy_head mean magnitude (pre-clamp) each step

Per-step columns:
  step  |raw_mu|_mean  |raw_mu|_max  |a|_mean  |a|_max  dm_max  env_S  reward  h_norm  c_norm
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


def build_env(ckpt_cfg, target, max_steps, device):
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
    return env, base


def apply_curriculum(env, ckpt_cfg, global_step):
    cur = ckpt_cfg.get("reward_weight_curriculum")
    if cur is None:
        return
    attr = str(cur.get("attr", "_rw_log_contrast_strehl"))
    warmup = int(cur.get("warmup_timesteps", 0))
    anneal = int(cur.get("anneal_timesteps", 1))
    start = float(cur.get("start_value", 0.0))
    end = float(cur.get("end_value", 1.0))
    progress = min(max(global_step - warmup, 0) / max(anneal, 1), 1.0)
    v = start + progress * (end - start)
    b = env
    while hasattr(b, "_env") and attr not in vars(b):
        b = b._env
    if attr in vars(b):
        setattr(b, attr, v)


def build_agent(env, ckpt_cfg, sd, device):
    class _Shim:
        single_observation_space = env.single_observation_space
        single_action_space = env.single_action_space
    agent = RecurrentActorCritic(
        _Shim(), torch.device(device),
        lstm_hidden_dim=ckpt_cfg.get("lstm_hidden_dim", 128),
        channel_scale=ckpt_cfg.get("channel_scale", 32),
        fc_scale=ckpt_cfg.get("fc_scale", 256),
        action_scale=ckpt_cfg.get("action_scale", 1.0),
        init_log_std=ckpt_cfg.get("init_log_std", -0.5),
        model_type=ckpt_cfg.get("model_type", "small"),
        target_dim=int(ckpt_cfg.get("target_dim", 0)),
        log_std_max=ckpt_cfg.get("log_std_max", None),
        log_std_min=ckpt_cfg.get("log_std_min", None),
    ).to(device)
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def run_rollout(agent, env, base, obs_ref_max, target, max_steps,
                device, seed, ablation):
    """Run one rollout under the given ablation and return per-step trace."""
    angle, r_frac, s_frac = target
    th = np.deg2rad(angle)
    tv_real = torch.tensor(
        [[np.sin(th), np.cos(th), r_frac, s_frac]],
        dtype=torch.float32, device=device)
    tv_zero = torch.zeros(1, 4, dtype=torch.float32, device=device)

    obs, _ = env.reset(seed=seed)
    if hasattr(env, "mask_obs"):
        obs = env.mask_obs(obs)
    obs_norm = normalize_obs_fixed(obs, obs_ref_max)

    h = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(1, agent.action_dim, device=device)
    prior_reward = torch.zeros(1, device=device)

    print(f"\n=== ablation: {ablation} ===")
    print(f"{'step':>4}  {'|mu|_mu':>9} {'|mu|_max':>9}  "
          f"{'|a|_mu':>9} {'|a|_max':>9}  {'dm_max':>10}  "
          f"{'env_S':>7}  {'reward':>9}  {'h_norm':>8}  {'c_norm':>8}")

    last_S = None
    for step in range(max_steps):
        obs_t = torch.from_numpy(obs_norm).float().to(device)

        # Apply ablations to LSTM-input tensors
        pa = torch.zeros_like(prior_action) if ablation == "zero_prior_act" else prior_action
        pr = torch.zeros_like(prior_reward) if ablation == "zero_prior_rew" else prior_reward
        tv_used = tv_zero if ablation == "zero_target" else tv_real
        h_in = torch.zeros_like(h) if ablation == "zero_hidden" else h
        c_in = torch.zeros_like(c) if ablation == "zero_hidden" else c

        with torch.no_grad():
            lstm_out, (h_new, c_new) = agent._forward_shared(
                obs_t, pa, pr, (h_in, c_in), target_vec=tv_used)
            raw_mu = agent.policy_head(lstm_out)
            action_t = agent.scale_and_clamp_action(raw_mu)

        h, c = h_new, c_new

        mu_mean = float(raw_mu.abs().mean())
        mu_max = float(raw_mu.abs().max())
        a_np = action_t.detach().cpu().numpy()

        obs_unmasked, rew, term, trunc, info = env.step(a_np)
        if hasattr(env, "mask_obs"):
            obs_for_policy = env.mask_obs(obs_unmasked)
        else:
            obs_for_policy = obs_unmasked
        obs_norm = normalize_obs_fixed(obs_for_policy, obs_ref_max)

        dm_max = (float(base._dm_actuators_t.abs().max().item())
                  if base._dm_actuators_t is not None else 0.0)
        env_S = float(info["strehl"][0])
        r = float(rew[0])
        h_norm = float(h.norm().item())
        c_norm = float(c.norm().item())
        last_S = env_S

        if step < 8 or step % 8 == 7 or step == max_steps - 1:
            print(f"{step:>4d}  {mu_mean:>9.4f} {mu_max:>9.4f}  "
                  f"{float(np.abs(a_np).mean()):>9.5f} "
                  f"{float(np.abs(a_np).max()):>9.5f}  "
                  f"{dm_max:>10.3e}  "
                  f"{env_S:>7.4f}  {r:>+9.4f}  "
                  f"{h_norm:>8.3f}  {c_norm:>8.3f}")

        prior_action = action_t.detach()
        prior_reward = torch.tensor([r], dtype=torch.float32, device=device)

    print(f"  -> final Strehl = {last_S:.4f}")
    return last_S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ablations", type=str, default=(
        "baseline,zero_hidden,zero_prior_act,zero_prior_rew,zero_target"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    global_step = int(ckpt.get("global_step", 0))
    sd = ckpt["model_state_dict"]

    targets = build_grid()
    target = targets[args.target_id]
    print(f"checkpoint:   {args.checkpoint}")
    print(f"global_step:  {global_step:,}")
    print(f"target:       id={args.target_id} angle={target[0]} r={target[1]} size={target[2]}")
    if "log_std" in sd:
        ls = sd["log_std"]
        print(f"log_std:      mean={ls.mean().item():.3f} "
              f"min={ls.min().item():.3f} max={ls.max().item():.3f} "
              f"sigma_mean={ls.exp().mean().item():.4f}")

    results = {}
    for ablation in args.ablations.split(","):
        ablation = ablation.strip()
        env, base = build_env(ckpt_cfg, target, args.max_steps, args.device)
        apply_curriculum(env, ckpt_cfg, global_step)
        obs_ref_max = float(getattr(base, "_reference_fpi_max", 1.0))
        agent = build_agent(env, ckpt_cfg, sd, args.device)
        final_S = run_rollout(agent, env, base, obs_ref_max, target,
                              args.max_steps, args.device, args.seed,
                              ablation)
        results[ablation] = final_S
        del env, base, agent

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:>20s}  final_S={v:.4f}")


if __name__ == "__main__":
    main()
