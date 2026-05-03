#!/usr/bin/env python
"""Roll out the bilateral-DM checkpoints, one policy per inner-ring target.

Companion to ``rollout_static_dark_hole_grid.py``. Each per-target run
under ``<sweep_dir>/target_NN/`` is loaded against its own (target_NN,
inner-ring) geometry. The env is built as a v5 BatchedOptomechEnv
(num_envs=1) wrapped in ``BilateralDMVectorEnv``, matching the training
configuration exactly: the policy sees the blinded observation and
outputs only the n_dm // 2 controlled-half slice; the wrapper expands
to the full DM command.

Each GIF shows four panels per step:

  1. DM OPD (pupil plane, signed colour scale)
  2. raw PSF (pre-detector, log scale, target circle in cyan, blind
     region circle in magenta)
  3. detector observation -- the policy's *actual* input, with the
     blind region zeroed out
  4. detector observation -- *unblinded*, so the human can see what
     light leaked into the magenta blind region (the test-time
     verification signal the policy never had access to)

Plus contrast traces for the target and blind regions, evolving
side-by-side: a successful policy keeps both contrasts dropping
together; a policy that gamed visible reward by pushing flux into the
blind region shows the blind-region contrast diverging upward while
the target contrast falls.

Usage:
    # All 6 inner-ring targets:
    python train/ppo/rollout_bilateral_dm_grid.py \\
        --sweep-dir dark_hole_runs/dark_hole_bilateral_dm_<ts>

    # Single target (debug):
    python train/ppo/rollout_bilateral_dm_grid.py --target-id 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from glob import glob
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.ppo.bilateral_dm import BilateralDMVectorEnv                 # noqa: E402
from train.ppo.launch_static_dark_hole import build_grid, _ids_for_ring # noqa: E402
from train.ppo.ppo_models import RecurrentActorCritic                   # noqa: E402
from train.ppo.train_ppo_optomech import normalize_obs_fixed            # noqa: E402
from train.ppo.train_ppo_elf_dark_hole_bilateral_dm import ENV_KWARGS   # noqa: E402


_SWEEP_PREFIX = "dark_hole_bilateral_dm_"
_DEFAULT_SWEEP_ROOT = "dark_hole_runs"


# ---------------------------------------------------------------------------
# Env + checkpoint helpers
# ---------------------------------------------------------------------------

class _EnvShim:
    def __init__(self, env):
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space


def _latest_sweep_dir(root: str = _DEFAULT_SWEEP_ROOT) -> str | None:
    if not os.path.isdir(root):
        return None
    cands = [
        os.path.join(root, n) for n in os.listdir(root)
        if n.startswith(_SWEEP_PREFIX)
        and os.path.isdir(os.path.join(root, n))
    ]
    if not cands:
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def _resolve_checkpoint(sweep_dir: str, target_idx: int,
                        prefer_latest: bool = False) -> str:
    target_dir = os.path.join(sweep_dir, f"target_{target_idx:02d}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"target dir missing: {target_dir}")
    if prefer_latest:
        ck = sorted(glob(os.path.join(
            target_dir, "ppo_optomech_*", "checkpoints", "*update_*.pt")),
            key=os.path.getmtime)
        if ck:
            return ck[-1]
    best = sorted(glob(os.path.join(
        target_dir, "ppo_optomech_*", "checkpoints", "best.pt")))
    if not best:
        ck = sorted(glob(os.path.join(
            target_dir, "ppo_optomech_*", "checkpoints", "*update_*.pt")),
            key=os.path.getmtime)
        if not ck:
            raise FileNotFoundError(
                f"no checkpoints under {target_dir}/ppo_optomech_*/checkpoints/")
        return ck[-1]
    return sorted(best, key=os.path.getmtime)[-1]


def _build_env(target, max_steps, device):
    """v5 single-env + bilateral wrapper, matching training config."""
    angle, r_frac, s_frac = target
    kw = dict(ENV_KWARGS)
    kw["dark_hole"] = True
    kw["dark_hole_angular_location_degrees"] = float(angle)
    kw["dark_hole_location_radius_fraction"] = float(r_frac)
    kw["dark_hole_size_radius"] = float(s_frac)
    kw["dark_hole_randomize_on_reset"] = False
    kw["max_episode_steps"] = int(max_steps)
    kw["silence"] = True
    kw["observation_window_size"] = 1
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    with contextlib.redirect_stdout(io.StringIO()):
        base = BatchedOptomechEnv(num_envs=1, device=device, **kw)
        env = BilateralDMVectorEnv(base, freeze_segments=True)
    return env, base


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


# ---------------------------------------------------------------------------
# Per-step diagnostics
# ---------------------------------------------------------------------------

def _capture_diagnostics(base, blind_mask_t):
    """Return (DM OPD, raw PSF, target contrast, blind contrast)."""
    # DM OPD = sum_a (dm_actuators[a] * dm_basis[a, h, w]) (env 0 only).
    dm_act = base._dm_actuators_t[0]                              # [A]
    dm_basis_flat = base._dm_basis_t_flat                          # [A, H*W]
    dm_opd = torch.matmul(dm_act, dm_basis_flat).reshape(
        base._H, base._W).detach().cpu().numpy()

    raw_psf = base._last_raw_psf_t[0].detach().cpu().numpy()
    psf_max = float(np.max(raw_psf))
    if psf_max <= 0.0:
        return dm_opd, raw_psf, float("nan"), float("nan")

    target_mask = base._hole_mask_t[0].detach().cpu().numpy()      # [H, W] bool
    blind_mask = blind_mask_t[0].detach().cpu().numpy()
    if target_mask.any():
        target_ct = float(np.mean(raw_psf[target_mask]) / psf_max)
    else:
        target_ct = float("nan")
    if blind_mask.any():
        blind_ct = float(np.mean(raw_psf[blind_mask]) / psf_max)
    else:
        blind_ct = float("nan")
    return dm_opd, raw_psf, target_ct, blind_ct


def run_episode(agent, env, base, target, seed, device):
    """One deterministic rollout, with both blinded and unblinded
    detector frames captured per step."""
    angle, r_frac, s_frac = target
    th = np.deg2rad(angle)
    tv_np = np.array(
        [np.sin(th), np.cos(th), r_frac, s_frac], dtype=np.float32)
    tv = torch.from_numpy(tv_np).unsqueeze(0).to(device)

    obs_ref_max = float(getattr(base, "_reference_fpi_max", 1.0))

    obs_blind, _ = env.reset(seed=seed)
    # Unblinded copy of the initial frame (for the verification panel).
    obs_unblind_t = base._obs_history.detach().cpu().numpy()        # [1, 1, H, W]

    obs_norm = normalize_obs_fixed(obs_blind, obs_ref_max)

    h = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(1, agent.action_dim, device=device)
    prior_reward = torch.zeros(1, device=device)

    blind_mask_t = env._blind_mask
    opd0, psf0, tct0, bct0 = _capture_diagnostics(base, blind_mask_t)

    rewards = []
    actions = []
    obs_blind_list = [obs_blind.copy()]
    obs_unblind_list = [obs_unblind_t.copy()]
    opd_list = [opd0]
    psf_list = [psf0]
    target_ct_list = [tct0]
    blind_ct_list = [bct0]
    strehls = []

    done = False
    while not done:
        obs_t = torch.from_numpy(obs_norm).float().to(device)
        with torch.no_grad():
            action_t, (h, c) = agent.get_deterministic_action(
                obs_t, prior_action, prior_reward, (h, c), target_vec=tv)
        a_np = action_t.detach().cpu().numpy()                      # [1, n_half]
        next_obs_blind, reward, term, trunc, info = env.step(a_np)
        # Capture unblinded view from the env BEFORE the wrapper masks.
        next_obs_unblind = base._obs_history.detach().cpu().numpy()
        done = bool(term[0] or trunc[0])
        rewards.append(float(reward[0]))
        actions.append(a_np[0].copy())
        obs_blind_list.append(next_obs_blind.copy())
        obs_unblind_list.append(next_obs_unblind.copy())
        opd_t, psf_t, tct_t, bct_t = _capture_diagnostics(
            base, blind_mask_t)
        opd_list.append(opd_t)
        psf_list.append(psf_t)
        target_ct_list.append(tct_t)
        blind_ct_list.append(bct_t)
        if "strehl" in info:
            strehls.append(float(info["strehl"][0]))
        prior_action = action_t
        prior_reward = torch.tensor(
            [reward[0]], dtype=torch.float32, device=device)
        obs_norm = normalize_obs_fixed(next_obs_blind, obs_ref_max)

    return {
        "rewards": rewards,
        "actions": np.array(actions),
        "obs_blind": obs_blind_list,
        "obs_unblind": obs_unblind_list,
        "opd": opd_list,
        "raw_psf": psf_list,
        "target_contrast": target_ct_list,
        "blind_contrast": blind_ct_list,
        "strehls": strehls,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "seed": int(seed),
        "target": target,
    }


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------

def _prep_obs(o):
    a = np.asarray(o)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def render_gif(ep, save_path, dpi=110, frame_duration=0.10):
    """4-image-panels GIF with target + blind circles and dual contrast traces."""
    target = ep["target"]
    angle, r_frac, s_frac = target
    target_id = ep.get("target_id", -1)

    obs_blind = ep["obs_blind"]
    obs_unblind = ep["obs_unblind"]
    opds = ep["opd"]
    psfs = ep["raw_psf"]
    target_ct = np.array(ep["target_contrast"], dtype=np.float64)
    blind_ct = np.array(ep["blind_contrast"], dtype=np.float64)
    rewards = ep["rewards"]
    strehls = ep["strehls"]
    cumulative = np.cumsum(rewards)
    T = len(rewards)

    obs_b_imgs = [_prep_obs(o) for o in obs_blind]
    obs_u_imgs = [_prep_obs(o) for o in obs_unblind]
    H, W = obs_b_imgs[0].shape[-2:]
    th = np.deg2rad(angle)
    cx_t = W / 2.0 + r_frac * (W / 2.0) * np.cos(th)
    cy_t = H / 2.0 + r_frac * (H / 2.0) * np.sin(th)
    cx_b = W / 2.0 - r_frac * (W / 2.0) * np.cos(th)
    cy_b = H / 2.0 - r_frac * (H / 2.0) * np.sin(th)
    r_px = s_frac * (W / 2.0)

    def _draw_target(ax):
        ax.add_patch(Circle(
            (cx_t, cy_t), r_px, fill=False, edgecolor="cyan",
            linestyle=(0, (1.5, 1.5)), linewidth=1.2, alpha=0.95))

    def _draw_blind(ax):
        ax.add_patch(Circle(
            (cx_b, cy_b), r_px, fill=False, edgecolor="magenta",
            linestyle=(0, (3, 2)), linewidth=1.2, alpha=0.95))

    # Trace y-bounds (log).
    def _log_bounds(arr):
        x = arr[np.isfinite(arr) & (arr > 0)]
        if not x.size:
            return 1e-10, 1.0
        lo = max(float(x.min()) * 0.5, 1e-12)
        hi = float(x.max()) * 2.0
        return lo, hi

    ct_lo, ct_hi = _log_bounds(np.concatenate([target_ct, blind_ct]))
    timesteps = np.arange(T + 1)

    frames = []
    for t in range(T + 1):
        fig = plt.figure(figsize=(11.0, 5.5), dpi=dpi)
        gs = fig.add_gridspec(
            2, 8,
            height_ratios=[3.4, 1.4],
            hspace=0.35, wspace=0.30,
            left=0.025, right=0.96, top=0.90, bottom=0.10,
        )

        # --- DM OPD ---
        ax_opd = fig.add_subplot(gs[0, 0:2])
        opd = opds[t]
        opd_max = max(float(np.nanmax(np.abs(opd))), 1e-12)
        im_opd = ax_opd.imshow(
            opd, cmap="RdBu_r", origin="lower",
            vmin=-opd_max, vmax=opd_max, aspect="equal")
        ax_opd.set_xticks([]); ax_opd.set_yticks([])
        ax_opd.set_title("DM OPD (m)", fontsize=10, pad=3)
        cb = fig.colorbar(im_opd, ax=ax_opd, fraction=0.046, pad=0.018)
        cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()
        cb.ax.tick_params(labelsize=7)

        # --- raw PSF (with both circles) ---
        ax_psf = fig.add_subplot(gs[0, 2:4])
        psf = psfs[t] if psfs[t] is not None else np.zeros((H, W))
        pmax = max(float(np.max(psf)), 1e-30)
        pflo = max(pmax * 1e-8, 1e-30)
        im_psf = ax_psf.imshow(
            np.maximum(psf, pflo), cmap="inferno",
            norm=mcolors.LogNorm(vmin=pflo, vmax=pmax),
            origin="lower", aspect="equal")
        _draw_target(ax_psf); _draw_blind(ax_psf)
        ax_psf.set_xticks([]); ax_psf.set_yticks([])
        ax_psf.set_title("raw PSF (log)", fontsize=10, pad=3)
        cb = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=7)

        # --- blinded obs (what the policy saw) ---
        ax_b = fig.add_subplot(gs[0, 4:6])
        ob = obs_b_imgs[t]
        omax = max(float(np.max(ob)), 2.0)
        im_b = ax_b.imshow(
            np.maximum(ob, 1.0), cmap="inferno",
            norm=mcolors.LogNorm(vmin=1.0, vmax=omax),
            origin="lower", aspect="equal")
        _draw_target(ax_b); _draw_blind(ax_b)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        ax_b.set_title("policy obs (blinded, DN log)",
                       fontsize=10, pad=3)
        cb = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=7)

        # --- unblinded obs (verification view) ---
        ax_u = fig.add_subplot(gs[0, 6:8])
        ou = obs_u_imgs[t]
        umax = max(float(np.max(ou)), 2.0)
        im_u = ax_u.imshow(
            np.maximum(ou, 1.0), cmap="inferno",
            norm=mcolors.LogNorm(vmin=1.0, vmax=umax),
            origin="lower", aspect="equal")
        _draw_target(ax_u); _draw_blind(ax_u)
        ax_u.set_xticks([]); ax_u.set_yticks([])
        ax_u.set_title("verification view (unblinded)",
                       fontsize=10, pad=3)
        cb = fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=7)

        # --- contrast traces (target + blind, log y) ---
        ax_ct = fig.add_subplot(gs[1, 1:7])
        ax_ct.set_yscale("log")
        ax_ct.set_ylim(ct_lo, ct_hi)
        ax_ct.set_xlim(0, max(T, 1))
        ax_ct.plot(timesteps, target_ct,
                   color="#cccccc", lw=0.7, alpha=0.4)
        ax_ct.plot(timesteps[:t + 1], target_ct[:t + 1],
                   color="cyan", lw=1.6, label="target")
        ax_ct.plot(timesteps, blind_ct,
                   color="#cccccc", lw=0.7, alpha=0.4)
        ax_ct.plot(timesteps[:t + 1], blind_ct[:t + 1],
                   color="magenta", lw=1.6, label="blind (verification)")
        if np.isfinite(target_ct[t]):
            ax_ct.plot([t], [target_ct[t]], "o",
                       color="cyan", markersize=5, mec="black", mew=0.4)
        if np.isfinite(blind_ct[t]):
            ax_ct.plot([t], [blind_ct[t]], "o",
                       color="magenta", markersize=5, mec="black", mew=0.4)
        ax_ct.grid(True, which="both", alpha=0.25, lw=0.4)
        ax_ct.tick_params(labelsize=8)
        ax_ct.set_xlabel("step", fontsize=9)
        ax_ct.set_title(
            "Mean contrast in target (cyan) and blind (magenta) regions",
            fontsize=9, pad=2)
        ax_ct.legend(loc="upper right", fontsize=7, frameon=True)

        head = (f"target {target_id:02d}  angle={angle:5.1f}°  "
                f"r={r_frac:.3f}  size={s_frac:.3f}")
        if t == 0:
            sub = (f"t=0  (initial)  "
                   f"target_C={target_ct[0]:.2e}  blind_C={blind_ct[0]:.2e}")
        else:
            r = rewards[t - 1]
            cum = cumulative[t - 1]
            s = f"  S={strehls[t-1]:.3f}" if strehls else ""
            sub = (f"t={t:>3d}  r={r:+.3f}  Σ={cum:+.2f}{s}  "
                   f"target_C={target_ct[t]:.2e}  blind_C={blind_ct[t]:.2e}")
        fig.suptitle(f"{head}\n{sub}", fontsize=11, y=0.985)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(save_path, frames, duration=frame_duration)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Roll the bilateral-DM dark-hole sweep, one policy per "
                    "inner-ring target.")
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-id", type=int, default=None,
                        help="Limit to one inner-ring target id (0..5).")
    parser.add_argument("--frame-duration", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--prefer-latest", action="store_true")
    args = parser.parse_args()

    targets = build_grid()
    inner_ids = _ids_for_ring(0)

    if args.target_id is not None:
        if args.target_id not in inner_ids:
            print(f"Error: --target-id must be one of {inner_ids}")
            sys.exit(1)
        target_indices = [args.target_id]
    else:
        target_indices = list(inner_ids)

    sweep_dir = args.sweep_dir or _latest_sweep_dir()
    if sweep_dir is None or not os.path.isdir(sweep_dir):
        print(f"Error: sweep dir not found "
              f"(--sweep-dir or newest {_SWEEP_PREFIX}* under "
              f"{_DEFAULT_SWEEP_ROOT}/)")
        sys.exit(1)
    if args.sweep_dir is None:
        print(f"--sweep-dir not given; using newest: {sweep_dir}")

    output_dir = args.output_dir or os.path.join(
        "test_output", f"{os.path.basename(sweep_dir.rstrip('/'))}_grid")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sweep dir:  {sweep_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Targets:    {target_indices}")
    print()

    summary = []
    for i in target_indices:
        target = targets[i]
        try:
            ckpt_path = _resolve_checkpoint(
                sweep_dir, i, prefer_latest=args.prefer_latest)
        except FileNotFoundError as e:
            print(f"  target {i:>2}: SKIP -- {e}")
            continue

        env, base = _build_env(target, args.max_steps, args.device)
        agent, config = _load_agent(ckpt_path, env, args.device)
        td = int(config.get("target_dim", 0))
        if td == 0:
            print(f"  target {i:>2}: WARNING -- target_dim=0 in checkpoint")
        ep = run_episode(
            agent, env, base, target, args.seed, args.device)
        ep["target_id"] = i

        gif_path = os.path.join(output_dir, f"target_{i:02d}.gif")
        render_gif(ep, gif_path, dpi=args.dpi,
                   frame_duration=args.frame_duration)
        env.close()

        final_t = (ep["target_contrast"][-1]
                   if ep["target_contrast"] else float("nan"))
        final_b = (ep["blind_contrast"][-1]
                   if ep["blind_contrast"] else float("nan"))
        final_s = ep["strehls"][-1] if ep["strehls"] else float("nan")
        summary.append((i, target, ep["return"], final_s, final_t, final_b))
        print(f"  target {i:>2}: angle={target[0]:6.1f}  r={target[1]:.3f}  "
              f"size={target[2]:.3f}  R={ep['return']:+.3f}  "
              f"S={final_s:.4f}  target_C={final_t:.2e}  "
              f"blind_C={final_b:.2e}  -> {gif_path}")

    print(f"\nWrote {len(summary)} GIFs to {output_dir}")


if __name__ == "__main__":
    main()
