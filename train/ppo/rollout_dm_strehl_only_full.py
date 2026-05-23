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

def _build_env(base_env_kwargs, max_steps, device, ckpt_config=None):
    """v5 single-env, no bilateral wrapper, no dark hole.

    Mirrors the training-time env: same env_kwargs (pulled from the
    checkpoint's stored config), max_episode_steps overridden to
    max_steps + 1 so the loop can stop one step short of the env's
    auto-reset and never see a reset-transient terminal frame.

    When ``ckpt_config`` is provided and has ``zernike_dm=True``, the
    base env is wrapped with ``ZernikeDMVectorEnv`` using the same
    Zernike-basis parameters the training run used (read from the
    checkpoint config). This keeps the eval-time projection matrix
    bit-identical to the training-time one, which matters because the
    projection is built from the env's pupil mask + DM influence basis
    and any drift would alias into a learned-vs-eval action mismatch.
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
        if ckpt_config is not None and ckpt_config.get("zernike_dm", False):
            from train.ppo.zernike_dm import ZernikeDMVectorEnv
            env = ZernikeDMVectorEnv(
                env,
                n_zernike=ckpt_config.get("zernike_n_modes", 36),
                skip_piston=ckpt_config.get("zernike_skip_piston", False),
                freeze_segments=ckpt_config.get(
                    "zernike_freeze_segments", True),
                normalize=ckpt_config.get("zernike_normalize", "inf"))
    return env


def _default_out_dir(ckpt_path):
    """Build a test_output/ path for a checkpoint following the repo
    convention: ``test_output/<run_id>__<seed_dir>/``.

    Example: a checkpoint at
        dark_hole_runs/dm_strehl_only_zernike_1779499965/seed_1772816303/
            ppo_optomech_*/checkpoints/latest.pt
    lands at
        test_output/dm_strehl_only_zernike_1779499965__seed_1772816303/
    """
    p = Path(ckpt_path).resolve()
    parts = p.parts
    # Walk up looking for the seed_<n> directory and the parent run id.
    seed = None
    run_id = None
    for i, part in enumerate(parts):
        if part.startswith("seed_"):
            seed = part
            run_id = parts[i - 1] if i > 0 else "unknown_run"
            break
    if seed is None:
        # Fall back: just use the checkpoint basename and immediate
        # parent dir so we never collide between checkpoints.
        return os.path.join(
            "test_output",
            f"{p.parent.parent.name}__{p.stem}")
    return os.path.join("test_output", f"{run_id}__{seed}")


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

def _capture_dm_opd(env):
    """Compute current DM OPD on the pupil grid as ``[H, W]`` numpy.

    DM state is in meters of surface displacement; reflective optics
    double the path, but we report surface here to match the env's
    internal units (the env's reward is computed on the propagated
    field directly, so the factor-of-2 is implicit in the optics).
    Returns NaN of shape (1, 1) when the env has no DM basis (defensive).
    """
    dm_basis_flat = getattr(env, "_dm_basis_t_flat", None)
    if dm_basis_flat is None:
        return np.full((1, 1), np.nan, dtype=np.float32)
    H, W = int(env._H), int(env._W)
    dm_state = env._dm_actuators_t[0]                        # [n_dm], m
    opd = torch.matmul(dm_state, dm_basis_flat).reshape(H, W)
    return opd.detach().cpu().numpy()


def run_episode(agent, env, seed, device, max_steps, stochastic=False):
    """Single deterministic (or stochastic) episode.

    Returns dict with strehl trajectory, reward trajectory, action
    trajectory, and per-step focal-plane PSF + pupil-plane DM OPD
    snapshots used by the rollout GIF renderer.
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
    obs_frames = []        # focal-plane intensity, one [H, W] per step
    opd_frames = []        # pupil-plane DM OPD, one [H, W] per step

    obs_frames.append(obs_np[0, 0].copy())
    opd_frames.append(_capture_dm_opd(env))

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
        opd_frames.append(_capture_dm_opd(env))
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
        "opd_frames": opd_frames,
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
    """Render the median-final-Strehl episode as a 4-panel diagnostic GIF.

    Panels:
        top-left:    focal-plane PSF (log scale, inferno)
        top-right:   pupil-plane DM OPD (signed, RdBu_r, symmetric vlim
                     fixed across the whole episode so colour intensity
                     reads as absolute OPD magnitude over time)
        bottom-left: current policy action as a bar chart (Zernike
                     coefficients when the policy is Zernike-wrapped,
                     raw DM dims otherwise; ylim fixed at ±1 to match
                     the action bound)
        bottom-right: traces over time -- strehl on the primary y-axis,
                     cumulative |a|_1 (per-dim mean) on the secondary
                     y-axis; vertical marker tracks the current step.

    "Median" = sort episodes by final strehl, take the middle one. With
    8 episodes that's sort-rank 4, which lands close to the median by
    construction.
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    finals = np.array([ep["strehls"][-1] for ep in episodes])
    median_idx = int(np.argsort(finals)[len(finals) // 2])
    ep = episodes[median_idx]

    obs = ep["obs_frames"]                                     # T+1 frames
    opd = ep["opd_frames"]                                     # T+1 frames
    actions = np.asarray(ep["actions"])                        # [T, n_act]
    strehls = np.asarray(ep["strehls"], dtype=np.float64)      # [T]
    T = actions.shape[0]
    n_act = actions.shape[1]

    # Fixed scales across the episode so the eye reads absolute changes.
    obs_max = max(float(np.max(f)) for f in obs)
    obs_floor = max(obs_max * 1e-5, 1e-20)
    opd_max = max(
        max(float(np.max(np.abs(o))) for o in opd if np.isfinite(o).any()),
        1e-12)
    opd_um_max = opd_max * 1e6

    # Per-step cumulative |a|_1 trace (averaged over the action dims so
    # 24-D Zernike and 1225-D DM are on comparable y-axes). Mean(|a|)
    # is the same quantity the env's action_penalty consumes, which is
    # the most useful "is the policy doing anything?" diagnostic.
    abs_means = np.abs(actions).mean(axis=1) if T > 0 else np.zeros(0)

    fig_w, fig_h = 11.0, 8.4
    dpi = 100

    def _render_frame(t):
        # t indexes obs/opd as 0..T (T+1 frames). For action / strehl /
        # cumulative we have T values indexed 0..T-1 corresponding to
        # "the step that produced obs[t+1]". At t=0 there's no prior
        # action; the action bars panel shows zeros.
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi,
                         constrained_layout=False)
        gs = fig.add_gridspec(
            2, 2,
            left=0.06, right=0.96, top=0.92, bottom=0.07,
            wspace=0.30, hspace=0.30,
            width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.75])

        # --- Top-left: PSF on log scale -----------------------------------
        ax_psf = fig.add_subplot(gs[0, 0])
        psf_show = np.maximum(obs[t], obs_floor)
        im_psf = ax_psf.imshow(
            psf_show, cmap="inferno", origin="lower",
            norm=mcolors.LogNorm(vmin=obs_floor, vmax=max(obs_max, obs_floor*10)),
            aspect="equal")
        ax_psf.set_title(f"focal-plane PSF (log)", fontsize=10)
        ax_psf.set_xticks([]); ax_psf.set_yticks([])
        divider = make_axes_locatable(ax_psf)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(im_psf, cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("intensity (DN)", fontsize=8, labelpad=2)

        # --- Top-right: DM OPD --------------------------------------------
        ax_opd = fig.add_subplot(gs[0, 1])
        if np.isfinite(opd[t]).any():
            opd_um = opd[t] * 1e6
            im_opd = ax_opd.imshow(
                opd_um, cmap="RdBu_r", origin="lower",
                vmin=-opd_um_max, vmax=opd_um_max, aspect="equal")
        else:
            im_opd = ax_opd.imshow(
                np.zeros((2, 2)), cmap="RdBu_r", origin="lower",
                vmin=-1, vmax=1)
        ax_opd.set_title("DM surface OPD", fontsize=10)
        ax_opd.set_xticks([]); ax_opd.set_yticks([])
        divider = make_axes_locatable(ax_opd)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(im_opd, cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("surface (µm)", fontsize=8, labelpad=2)

        # --- Bottom-left: action bars -------------------------------------
        ax_act = fig.add_subplot(gs[1, 0])
        if t == 0:
            bars = np.zeros(n_act, dtype=np.float64)
            act_title = f"action (step 0; no prior action)"
        else:
            bars = actions[t - 1]
            act_title = f"action (step {t}, dim={n_act})"
        idx = np.arange(n_act)
        # For low-dim (e.g. 24 Zernike) draw thick bars; for very high-
        # dim (1225 DM) draw a thin stem so we don't render a black
        # rectangle. Threshold at 64 so anything past a few rows of
        # Noll Zernikes shows the dense rendering.
        if n_act <= 64:
            ax_act.bar(idx, bars, color="C0", edgecolor="black",
                       linewidth=0.4)
        else:
            ax_act.vlines(idx, 0, bars, color="C0", linewidth=0.6)
        ax_act.axhline(0, color="black", linewidth=0.6)
        ax_act.set_ylim(-1.05, 1.05)
        ax_act.set_xlim(-0.5, n_act - 0.5)
        ax_act.set_xlabel("action index", fontsize=8)
        ax_act.set_ylabel("action value (clamped to ±1)", fontsize=8)
        ax_act.set_title(act_title, fontsize=10)
        ax_act.tick_params(labelsize=7)
        ax_act.grid(True, axis="y", alpha=0.25)

        # --- Bottom-right: strehl + cumulative control work --------------
        ax_tr = fig.add_subplot(gs[1, 1])
        steps = np.arange(1, T + 1)
        ax_tr.plot(steps, strehls, color="C2", linewidth=1.6,
                   label="strehl")
        ax_tr.set_ylim(0, max(1.0, float(strehls.max()) * 1.1 if T else 1.0))
        ax_tr.set_ylabel("strehl", color="C2", fontsize=8)
        ax_tr.tick_params(axis="y", labelcolor="C2", labelsize=7)
        ax_tr.tick_params(axis="x", labelsize=7)
        ax_tr.set_xlabel("step", fontsize=8)
        ax_tr.set_xlim(0, T)
        ax_tr.grid(True, alpha=0.25)
        ax_tr2 = ax_tr.twinx()
        ax_tr2.plot(steps, abs_means, color="C3", linewidth=1.2,
                    linestyle="--", label="mean|action|")
        ax_tr2.set_ylabel("mean |action| (control work)",
                          color="C3", fontsize=8)
        ax_tr2.tick_params(axis="y", labelcolor="C3", labelsize=7)
        ax_tr2.set_ylim(0, max(0.05, float(abs_means.max()) * 1.1
                               if T else 0.05))
        # Current-step marker.
        if 0 < t <= T:
            ax_tr.axvline(t, color="0.3", linewidth=0.8, linestyle=":")
        ax_tr.set_title("strehl + control-work trace", fontsize=10)

        # --- Header -------------------------------------------------------
        if 0 < t <= T:
            r = float(ep["rewards"][t - 1])
            cum = float(np.sum(ep["rewards"][:t]))
            s = float(strehls[t - 1])
            header = (f"step {t:>3d} / {T}     "
                      f"strehl = {s:.4f}     "
                      f"reward = {r:+.4f}     "
                      f"Σreward = {cum:+.3f}     "
                      f"mean|a| = {float(abs_means[t-1]):.4f}")
        else:
            header = f"step 0 / {T}     (initial state, no action taken)"
        fig.suptitle(header, fontsize=11)

        # Render to RGBA array, drop alpha for GIF.
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    frames = [_render_frame(t) for t in range(T + 1)]
    imageio.mimsave(out_path, frames,
                    duration=frame_duration / 1000.0, loop=0)


# ---------------------------------------------------------------------------
# Per-checkpoint driver
# ---------------------------------------------------------------------------

def run_one_checkpoint(ckpt_path, args):
    print(f"\n=== {ckpt_path} ===")
    ek, ckpt_cfg = _pull_ckpt_config(ckpt_path)
    env = _build_env(
        base_env_kwargs=ek,
        max_steps=args.max_steps, device=args.device,
        ckpt_config=ckpt_cfg)
    agent, config, _ckpt = _load_agent(ckpt_path, env, args.device)
    # global_step / update / best_eval_return are top-level keys on the
    # checkpoint dict, NOT nested in config -- see CLAUDE.md "Checkpoint
    # structure". config holds the run's hyperparams.
    global_step = int(_ckpt.get("global_step", 0))
    update = int(_ckpt.get("update", 0))
    print(f"  global_step={global_step:,}  update={update}  "
          f"action_dim={agent.action_dim}  "
          f"target_dim={int(config.get('target_dim', 0))}")

    modes = []
    if args.stochastic in ("det", "both"):
        modes.append(("det", False))
    if args.stochastic in ("stoch", "both"):
        modes.append(("stoch", True))

    # Match the convention used by every other rollout in this repo
    # (rollout_bilateral_dm_grid, sweep_tiptilt, etc.): output lands
    # under top-level test_output/, NOT next to the checkpoint. Each
    # checkpoint gets its own subdir named after the run id + seed so
    # multi-seed sweeps don't collide. ``--output-dir`` overrides.
    out_root = args.output_dir or _default_out_dir(ckpt_path)
    os.makedirs(out_root, exist_ok=True)
    print(f"  -> {out_root}/")

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
                base_env_kwargs=ek,
                max_steps=args.max_steps, device=args.device,
                ckpt_config=ckpt_cfg)

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
            "global_step": global_step,
            "update": update,
            "modes": all_summaries,
        }, f, indent=2)
    print(f"  -> {json_path}")
    env.close()


def _pull_env_kwargs(ckpt_path):
    """Backward-compat: return just env_kwargs."""
    return _pull_ckpt_config(ckpt_path)[0]


def _pull_ckpt_config(ckpt_path):
    """Open the checkpoint and return ``(env_kwargs, full_config)``.

    Both are needed by ``_build_env``: env_kwargs go to the v5 env
    constructor; the full config carries top-level flags like
    ``zernike_dm`` / ``zernike_n_modes`` that drive the action-space
    wrapper choice. The checkpoint is loaded once per call (light --
    no model state is materialised) and may be cached by the caller.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    ek = cfg.get("env_kwargs", None)
    if ek is None:
        raise RuntimeError(
            f"{ckpt_path} has no env_kwargs in its config; cannot "
            f"reconstruct the training env.")
    return ek, cfg


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _resolve_checkpoints(args):
    """Either --checkpoint, or every checkpoint under --run-dir.

    Discovers checkpoints at any depth under each seed_* directory --
    run_main lands them at ``seed_<S>/ppo_optomech_<S>_<ts>/checkpoints/``
    (an extra timestamped subdir), so a single-glob ``seed_*/checkpoints``
    misses them. We walk every ``checkpoints/`` directory under each
    seed dir, prefer ``latest.pt``, and fall back to the highest-numbered
    ``update_*.pt`` if no ``latest.pt`` exists.
    """
    if args.checkpoint:
        return [args.checkpoint]
    if not args.run_dir:
        print("Error: provide --checkpoint or --run-dir.")
        sys.exit(1)
    found = []
    for seed_dir in sorted(Path(args.run_dir).glob("seed_*")):
        # rglob picks up checkpoints/ at seed_dir/checkpoints (older
        # layout) and seed_dir/ppo_optomech_*/checkpoints (current).
        for ckpts_dir in sorted(seed_dir.rglob("checkpoints")):
            if not ckpts_dir.is_dir():
                continue
            latest = ckpts_dir / "latest.pt"
            if latest.exists():
                found.append(str(latest))
            else:
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
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: "
                        "test_output/<run_id>__<seed_dir>/). When "
                        "--run-dir is used, every checkpoint reuses "
                        "this same flag value, so prefer the default "
                        "for multi-seed sweeps to avoid collisions.")
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
