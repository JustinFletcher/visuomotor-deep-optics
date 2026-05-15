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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    # History filename patterns. New runs write history_step_*.pt
    # (step-based naming, every N env steps). Legacy runs wrote
    # history_update_*.pt (slot-based on update count). Either matches.
    history_glob_step = os.path.join(
        target_dir, "ppo_optomech_*", "checkpoints", "history_step_*.pt")
    history_glob_update = os.path.join(
        target_dir, "ppo_optomech_*", "checkpoints", "*update_*.pt")

    def _newest_history():
        cands = (
            sorted(glob(history_glob_step), key=os.path.getmtime)
            + sorted(glob(history_glob_update), key=os.path.getmtime))
        if cands:
            cands.sort(key=os.path.getmtime)
            return cands[-1]
        return None

    if prefer_latest:
        # Prefer an explicit latest.pt (the running pointer the trainer
        # rewrites every save), fall back to the newest history file,
        # then to best.pt.
        latest_ptr = sorted(glob(os.path.join(
            target_dir, "ppo_optomech_*", "checkpoints", "latest.pt")),
            key=os.path.getmtime)
        if latest_ptr:
            return latest_ptr[-1]
        nh = _newest_history()
        if nh:
            return nh
    best = sorted(glob(os.path.join(
        target_dir, "ppo_optomech_*", "checkpoints", "best.pt")))
    if not best:
        nh = _newest_history()
        if nh is None:
            raise FileNotFoundError(
                f"no checkpoints under {target_dir}/ppo_optomech_*/checkpoints/")
        return nh
    return sorted(best, key=os.path.getmtime)[-1]


def _build_env(target, max_steps, device, bilateral=True,
               bilateral_mode="fixed_vertical", freeze_segments=True,
               base_env_kwargs=None, zero_init=False):
    """v5 single-env, optionally wrapped with the bilateral DM wrapper.

    bilateral=True matches the bilateral training track (action space
    halved by the symmetric expansion, obs has the blind-region mask).
    bilateral=False matches the full-DM debug track (no wrapper -- the
    policy sees the unmasked obs and outputs the full 1240-dim action).

    base_env_kwargs: the env kwargs to start from. Crucial that this
    matches what the policy was trained against -- otherwise the env
    will interpret actions under different rules (e.g., incremental
    vs absolute DM control, action_scale, init perturbation magnitude)
    and the policy will produce wildly off-distribution behaviour. The
    caller should pass the env_kwargs recovered from the checkpoint's
    stored config, NOT the module-level ENV_KWARGS imported at the
    top of this file (which references a different training script).
    """
    angle, r_frac, s_frac = target
    kw = dict(base_env_kwargs if base_env_kwargs is not None
              else ENV_KWARGS)
    kw["dark_hole"] = True
    kw["dark_hole_angular_location_degrees"] = float(angle)
    kw["dark_hole_location_radius_fraction"] = float(r_frac)
    kw["dark_hole_size_radius"] = float(s_frac)
    kw["dark_hole_randomize_on_reset"] = False
    # Extend the env's truncation horizon by one step beyond what the
    # rollout will actually take. v5 auto-resets on the truncating
    # step (zeroes _dm_actuators_t, re-simulates from a fresh state,
    # overwrites _obs_history) -- that reset is what was making the
    # last GIF frame look like an episode start. Loop control below
    # stops the rollout one step short of this, so the env never
    # auto-resets within the captured window.
    kw["max_episode_steps"] = int(max_steps) + 1
    if zero_init:
        # One-off sanity rollout: kill every randomised initial
        # disturbance so the rollout starts from a perfectly flat
        # state. Use this to test whether residual PSF / obs
        # asymmetry comes from the random init draw or from
        # downstream symmetry-breaking in the env pipeline (object
        # convolution, detector model, polychromatic accumulation).
        for k in (
            "init_dm_micron_std",
            "init_piston_micron_std",
            "init_piston_micron_mean",
            "init_piston_clip_micron",
            "init_tip_arcsec_std",
            "init_tilt_arcsec_std",
        ):
            if k in kw:
                kw[k] = 0.0
    kw["silence"] = True
    kw["observation_window_size"] = 1
    # Disable reward-vector overhead for rollouts.
    kw["reward_vector_enabled"] = False
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    with contextlib.redirect_stdout(io.StringIO()):
        base = BatchedOptomechEnv(num_envs=1, device=device, **kw)
        if bilateral:
            env = BilateralDMVectorEnv(
                base, freeze_segments=freeze_segments, mode=bilateral_mode)
        else:
            env = base
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
    """Return (DM OPD, raw PSF, target contrast, blind contrast).

    blind_mask_t may be None (full-DM, no wrapper); blind contrast is
    NaN in that case.
    """
    dm_act = base._dm_actuators_t[0]                              # [A]
    dm_basis_flat = base._dm_basis_t_flat                          # [A, H*W]
    dm_opd = torch.matmul(dm_act, dm_basis_flat).reshape(
        base._H, base._W).detach().cpu().numpy()

    raw_psf = base._last_raw_psf_t[0].detach().cpu().numpy()
    psf_max = float(np.max(raw_psf))
    if psf_max <= 0.0:
        return dm_opd, raw_psf, float("nan"), float("nan")

    # Inverted definition: higher = deeper hole. We report
    #     C(R, t) = max_p I_raw(p, t) / mean_{p in R} I_raw(p, t)
    # i.e. the peak-to-in-region brightness ratio. With no light in
    # the region the value diverges, so floor the mean at psf_max *
    # 1e-15 so the plotted value caps cleanly at 1e15 instead of inf.
    mean_floor = psf_max * 1e-15
    target_mask = base._hole_mask_t[0].detach().cpu().numpy()      # [H, W] bool
    if target_mask.any():
        mean_in = max(float(np.mean(raw_psf[target_mask])), mean_floor)
        target_ct = psf_max / mean_in
    else:
        target_ct = float("nan")
    if blind_mask_t is not None:
        blind_mask = blind_mask_t[0].detach().cpu().numpy()
        if blind_mask.any():
            mean_in = max(float(np.mean(raw_psf[blind_mask])), mean_floor)
            blind_ct = psf_max / mean_in
        else:
            blind_ct = float("nan")
    else:
        blind_ct = float("nan")
    return dm_opd, raw_psf, target_ct, blind_ct


def run_episode(agent, env, base, target, seed, device, max_steps,
                stochastic=False):
    """One rollout. If stochastic=False (default) the policy uses its
    deterministic mean action; if True it samples from Normal(mean,
    log_std.exp()), matching the training-time action distribution.
    """
    if stochastic:
        # Seed torch RNG so sampled rollouts are reproducible per
        # --seed. Without this each invocation produces different
        # action draws even with the same env reset seed.
        torch.manual_seed(int(seed))
    angle, r_frac, s_frac = target
    th = np.deg2rad(angle)
    tv_np = np.array(
        [np.sin(th), np.cos(th), r_frac, s_frac], dtype=np.float32)
    tv = torch.from_numpy(tv_np).unsqueeze(0).to(device)

    obs_ref_max = float(getattr(base, "_reference_fpi_max", 1.0))

    obs_unmasked, _ = env.reset(seed=seed)
    # Unblinded copy of the initial frame (for the verification panel).
    obs_unblind_t = base._obs_history.detach().cpu().numpy()        # [1, 1, H, W]

    # CRITICAL: training applies env.mask_obs() between the env's
    # raw observation and the policy input (the bilateral wrapper's
    # blind region is zeroed before the obs reaches the encoder).
    # If we skip that here, the policy sees a different obs
    # distribution than it was trained on -- non-zero pixels in the
    # blind region -- and produces wildly off-distribution actions.
    # Full-DM (unwrapped) envs have no mask_obs and this is a no-op.
    if hasattr(env, "mask_obs"):
        obs_for_policy = env.mask_obs(obs_unmasked)
    else:
        obs_for_policy = obs_unmasked
    obs_blind = obs_for_policy  # for filmstrip / capture lists
    obs_norm = normalize_obs_fixed(obs_for_policy, obs_ref_max)

    h = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    c = torch.zeros(
        agent.lstm_num_layers, 1, agent.lstm_hidden_dim, device=device)
    prior_action = torch.zeros(1, agent.action_dim, device=device)
    prior_reward = torch.zeros(1, device=device)

    # Bilateral wrapper exposes _blind_mask; full-DM (unwrapped) env
    # does not. Diagnostics handle the None case.
    blind_mask_t = getattr(env, "_blind_mask", None)
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

    # Stop manually after max_steps env steps so the env (whose
    # max_episode_steps was set to max_steps + 1 in _build_env) never
    # reaches truncation and never auto-resets within the captured
    # window. Without this, v5 would zero _dm_actuators_t and re-run
    # _batched_simulate from the reset state on the truncating step,
    # producing a final GIF frame that looks like an episode start.
    done = False
    steps_taken = 0
    while not done and steps_taken < max_steps:
        obs_t = torch.from_numpy(obs_norm).float().to(device)
        with torch.no_grad():
            if stochastic:
                # Sample from Normal(mean, std), same path as training.
                # get_action_and_value returns the raw sample; the
                # caller is responsible for scale_and_clamp.
                action_t, _logp, _ent, _v, (h, c) = (
                    agent.get_action_and_value(
                        obs_t, prior_action, prior_reward, (h, c),
                        target_vec=tv))
                action_t = agent.scale_and_clamp_action(action_t)
            else:
                action_t, (h, c) = agent.get_deterministic_action(
                    obs_t, prior_action, prior_reward, (h, c), target_vec=tv)
        a_np = action_t.detach().cpu().numpy()                      # [1, n_half]
        next_obs_unmasked, reward, term, trunc, info = env.step(a_np)
        steps_taken += 1
        # Unblinded copy (verification view); same env source either way.
        next_obs_unblind = base._obs_history.detach().cpu().numpy()
        done = bool(term[0] or trunc[0])
        rewards.append(float(reward[0]))
        actions.append(a_np[0].copy())
        # Apply the blind mask between env emission and policy input,
        # matching the training-time path (run_ppo_training calls
        # envs.mask_obs() before normalize_obs_fixed too). Without
        # this the policy sees a different obs distribution at eval
        # than it did at training (non-zero pixels in the blind
        # region) and produces off-distribution actions.
        if hasattr(env, "mask_obs"):
            next_obs_for_policy = env.mask_obs(next_obs_unmasked)
        else:
            next_obs_for_policy = next_obs_unmasked
        obs_blind_list.append(next_obs_for_policy.copy())
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
        obs_norm = normalize_obs_fixed(next_obs_for_policy, obs_ref_max)

    # Snapshot the actual env masks so render_gif can mark the target
    # and blind regions from their exact pixel positions rather than
    # re-deriving from (angle, r_frac, s_frac). The bilateral wrapper's
    # blind-mask geometry depends on the symmetry mode (fixed_vertical
    # mirrors only x; per_target_radial flips both x and y) and that
    # mode isn't carried in `target`, so the only mode-agnostic way to
    # locate the blind region is the mask itself.
    target_mask_np = base._hole_mask_t[0].detach().cpu().numpy() \
        if base._hole_mask_t is not None else None
    blind_mask_np = (blind_mask_t[0].detach().cpu().numpy()
                     if blind_mask_t is not None else None)

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
        "target_mask": target_mask_np,
        "blind_mask": blind_mask_np,
        # Focal-plane angular pixel scale (arcsec/pixel) for labelling
        # axes in physical units. Fallback to 1.0 if v5 was built
        # without ifov exposed (old envs).
        "ifov_arcsec": float(getattr(base, "_ifov_arcsec", 1.0)),
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
    """Animated GIF with target (and optionally blind) overlays and
    contrast trace(s).

    Detects bilateral vs full-DM mode from the captured episode data:
    if the blind-contrast trace is all-NaN (full-DM run, no wrapper)
    the blind circle, the second obs panel, and the magenta trace
    line are all suppressed.
    """
    target = ep["target"]
    angle, r_frac, s_frac = target
    target_id = ep.get("target_id", -1)

    obs_blind = ep["obs_blind"]
    obs_unblind = ep["obs_unblind"]
    opds = ep["opd"]
    psfs = ep["raw_psf"]
    target_ct = np.array(ep["target_contrast"], dtype=np.float64)
    blind_ct = np.array(ep["blind_contrast"], dtype=np.float64)
    has_blind = bool(np.isfinite(blind_ct).any())
    rewards = ep["rewards"]
    strehls = ep["strehls"]
    cumulative = np.cumsum(rewards)
    T = len(rewards)

    obs_b_imgs = [_prep_obs(o) for o in obs_blind]
    obs_u_imgs = [_prep_obs(o) for o in obs_unblind]
    H, W = obs_b_imgs[0].shape[-2:]

    # Physical scale for the focal-plane panels. The image is
    # [H, W] pixels with the optical axis at (H/2, W/2); arcsec extent
    # is ifov * H. We label axes in arcsec and centre at zero.
    ifov_arcsec = float(ep.get("ifov_arcsec", 1.0))
    foc_half = 0.5 * H * ifov_arcsec     # half-width in arcsec
    foc_extent = [-foc_half, foc_half, -foc_half, foc_half]

    # Pupil-plane (DM OPD) is normalised to pupil radius 1.
    pup_extent = [-1.0, 1.0, -1.0, 1.0]

    def _px_to_arcsec(px_x, px_y):
        return ((px_x - W / 2.0) * ifov_arcsec,
                (px_y - H / 2.0) * ifov_arcsec)

    # Marker radius in arcsec (from r_frac * H/2 pixels).
    marker_r_arcsec = s_frac * (H / 2.0) * ifov_arcsec

    # Derive marker centres from the actual env masks rather than from
    # (angle, r_frac). The bilateral wrapper's blind-mask placement
    # depends on the symmetry mode (fixed_vertical mirrors x only;
    # per_target_radial flips both), and re-deriving the geometry from
    # angle here mis-placed the blind circle on fixed_vertical runs.
    target_mask = ep.get("target_mask")
    blind_mask = ep.get("blind_mask")

    def _centroid_arcsec(mask):
        if mask is None or not mask.any():
            return None
        ys, xs = np.where(mask)
        return _px_to_arcsec(float(xs.mean()), float(ys.mean()))

    target_centre = _centroid_arcsec(target_mask)
    blind_centre = _centroid_arcsec(blind_mask)

    def _draw_target(ax):
        if target_centre is None:
            return
        ax.add_patch(Circle(
            target_centre, marker_r_arcsec, fill=False, edgecolor="cyan",
            linestyle=(0, (1.5, 1.5)), linewidth=1.2, alpha=0.95))

    def _draw_blind(ax):
        if blind_centre is None:
            return
        ax.add_patch(Circle(
            blind_centre, marker_r_arcsec, fill=False, edgecolor="magenta",
            linestyle=(0, (3, 2)), linewidth=1.2, alpha=0.95))

    # Trace y-bounds (log).
    def _log_bounds(arr):
        x = arr[np.isfinite(arr) & (arr > 0)]
        if not x.size:
            return 1e-10, 1.0
        lo = max(float(x.min()) * 0.5, 1e-12)
        hi = float(x.max()) * 2.0
        return lo, hi

    _ct_for_bounds = (np.concatenate([target_ct, blind_ct])
                      if has_blind else target_ct)
    ct_lo, ct_hi = _log_bounds(_ct_for_bounds)
    timesteps = np.arange(T + 1)

    # Static header — same on every frame.
    head_static = (f"target {target_id:02d}     "
                   f"angle = {angle:.1f}°     "
                   f"r = {r_frac:.3f}     "
                   f"size = {s_frac:.3f}")

    def _metrics_line(t):
        parts = [f"t = {t:>3d}"]
        if t > 0:
            parts.append(f"r = {rewards[t-1]:+.3f}")
            parts.append(f"Σr = {cumulative[t-1]:+.2f}")
            if strehls:
                parts.append(f"S = {strehls[t-1]:.3f}")
        parts.append(f"$C_\\mathrm{{target}}$ = {target_ct[t]:.2e}")
        if has_blind:
            parts.append(f"$C_\\mathrm{{blind}}$ = {blind_ct[t]:.2e}")
        return "     ".join(parts)

    # Layout: fixed positions, no constrained_layout (which recomputes
    # per-frame and produces visible jitter in the GIF). All four
    # imshow panels share the same axes-box width; flush colorbars are
    # appended via make_axes_locatable so colorbar size never changes
    # with content.
    # Figure widened a touch and inner_gap roomier so the in-between
    # tick labels don't crash into the next panel; panel titles kept
    # very short and units off-loaded to the colorbar label so they
    # never overflow the box width.
    fig_w, fig_h = 10.4, 9.6
    panel_h = 0.31
    row0_top, row0_bot = 0.90, 0.90 - panel_h
    row1_top, row1_bot = 0.55, 0.55 - panel_h
    trace_top, trace_bot = 0.175, 0.06
    left_margin, right_margin = 0.065, 0.985
    inner_gap = 0.095
    col_w = (right_margin - left_margin - inner_gap) / 2.0
    box_opd  = (left_margin,                   row0_bot, col_w, panel_h)
    box_psf  = (left_margin + col_w + inner_gap, row0_bot, col_w, panel_h)
    box_pol  = (left_margin,                   row1_bot, col_w, panel_h)
    box_ver  = (left_margin + col_w + inner_gap, row1_bot, col_w, panel_h)
    box_trace = (left_margin, trace_bot,
                 right_margin - left_margin, trace_top - trace_bot)

    def _add_flush_colorbar(fig, ax, im, label, fmt_powerlimits=None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.06)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(label, fontsize=7.5, labelpad=2)
        if fmt_powerlimits is not None:
            cb.formatter.set_powerlimits(fmt_powerlimits)
            cb.update_ticks()
        return cb

    def _style_focal(ax, title):
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlabel("x (arcsec)", fontsize=7.5, labelpad=2)
        ax.set_ylabel("y (arcsec)", fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=6.5)

    def _style_pupil(ax, title):
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlabel("pupil x (R)", fontsize=7.5, labelpad=2)
        ax.set_ylabel("pupil y (R)", fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    frames = []
    for t in range(T + 1):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        # --- Row 0: DM OPD + raw PSF ---
        ax_opd = fig.add_axes(box_opd)
        opd = opds[t]
        opd_max = max(float(np.nanmax(np.abs(opd))), 1e-12)
        im_opd = ax_opd.imshow(
            opd, cmap="RdBu_r", origin="lower", aspect="equal",
            vmin=-opd_max, vmax=opd_max, extent=pup_extent)
        _style_pupil(ax_opd, "DM OPD")
        _add_flush_colorbar(fig, ax_opd, im_opd,
                            label="surface (m)",
                            fmt_powerlimits=(-2, 2))

        ax_psf = fig.add_axes(box_psf)
        psf = psfs[t] if psfs[t] is not None else np.zeros((H, W))
        pmax = max(float(np.max(psf)), 1e-30)
        pflo = max(pmax * 1e-8, 1e-30)
        im_psf = ax_psf.imshow(
            np.maximum(psf, pflo), cmap="inferno",
            norm=mcolors.LogNorm(vmin=pflo, vmax=pmax),
            origin="lower", aspect="equal", extent=foc_extent)
        _draw_target(ax_psf)
        if has_blind:
            _draw_blind(ax_psf)
        _style_focal(ax_psf, "raw PSF")
        _add_flush_colorbar(fig, ax_psf, im_psf, label="intensity (log)")

        # --- Row 1: policy obs + verification (when wrapped) ---
        ax_b = fig.add_axes(box_pol)
        ob = obs_b_imgs[t]
        omax = max(float(np.max(ob)), 2.0)
        im_b = ax_b.imshow(
            np.maximum(ob, 1.0), cmap="inferno",
            norm=mcolors.LogNorm(vmin=1.0, vmax=omax),
            origin="lower", aspect="equal", extent=foc_extent)
        _draw_target(ax_b)
        if has_blind:
            _draw_blind(ax_b)
        _style_focal(ax_b, "policy obs (blinded)" if has_blind
                           else "policy obs")
        _add_flush_colorbar(fig, ax_b, im_b, label="detector DN (log)")

        if has_blind:
            ax_u = fig.add_axes(box_ver)
            ou = obs_u_imgs[t]
            umax = max(float(np.max(ou)), 2.0)
            im_u = ax_u.imshow(
                np.maximum(ou, 1.0), cmap="inferno",
                norm=mcolors.LogNorm(vmin=1.0, vmax=umax),
                origin="lower", aspect="equal", extent=foc_extent)
            _draw_target(ax_u); _draw_blind(ax_u)
            _style_focal(ax_u, "verification view (unblinded)")
            _add_flush_colorbar(fig, ax_u, im_u, label="detector DN (log)")

        # --- Row 2: contrast trace, axes-box aligned with imshow row ---
        ax_ct = fig.add_axes(box_trace)
        ax_ct.set_yscale("log")
        ax_ct.set_ylim(ct_lo, ct_hi)
        ax_ct.set_xlim(0, max(T, 1))
        ax_ct.plot(timesteps, target_ct,
                   color="#cccccc", lw=0.7, alpha=0.4)
        ax_ct.plot(timesteps[:t + 1], target_ct[:t + 1],
                   color="cyan", lw=1.6, label="target")
        if has_blind:
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
        ax_ct.tick_params(labelsize=7)
        ax_ct.set_xlabel("step", fontsize=8, labelpad=2)
        ax_ct.set_ylabel(r"$C(R, t)$", fontsize=8, labelpad=2)
        ax_ct.set_title(
            r"$C(R, t) \;=\; "
            r"\max_{p}\, I_\mathrm{raw}(p, t) \;/\; "
            r"\langle I_\mathrm{raw}(p, t)\rangle_{p \in R}$"
            r"   (dimensionless)",
            fontsize=9, pad=3)
        ax_ct.legend(loc="lower right", fontsize=7, frameon=True)

        # Header strip — two fixed-position lines so neither side wraps
        # into the other regardless of metric content. Identity on top,
        # dynamic state directly below.
        fig.text(left_margin, 0.975, head_static,
                 ha="left", va="center", fontsize=10.5,
                 weight="semibold")
        fig.text(left_margin, 0.945, _metrics_line(t),
                 ha="left", va="center", fontsize=9.5,
                 family="monospace")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(save_path, frames, duration=frame_duration)


# ---------------------------------------------------------------------------
# Bilateral-symmetry GIF
# ---------------------------------------------------------------------------

def _fold_diff_lr(img):
    """Anti-symmetric folded difference across the vertical axis x = W/2.

    For a perfectly bilaterally-symmetric image,
        img(x, y) == img(W - 1 - x, y)
    so this returns zeros everywhere. For asymmetric content, the
    result is anti-symmetric: D(x, y) == -D(W - 1 - x, y).
    """
    return img.astype(np.float64) - np.fliplr(img).astype(np.float64)


def _relative_l2_asymmetry(img):
    """Dimensionless asymmetry index in [0, sqrt(2)].

        L2 = ||I - flip_x(I)||_2 / ||I||_2

    0 = perfect bilateral symmetry; sqrt(2) ~ 1.414 = maximally
    one-sided. Uses the full image so the denominator is the same
    statistic as the numerator (an asymmetric content scale).
    """
    arr = img.astype(np.float64)
    diff_norm = float(np.sqrt(np.sum((arr - np.fliplr(arr)) ** 2)))
    img_norm = float(np.sqrt(np.sum(arr ** 2)))
    if img_norm <= 0:
        return 0.0
    return diff_norm / img_norm


def render_symmetry_gif(ep, save_path, dpi=110, frame_duration=0.10):
    """Bilateral-symmetry GIF: folded-difference images and L2 traces.

    For each step, computes the residual after subtracting the left-
    right mirror of the DM OPD, the raw PSF, and the detector
    observation, and plots them alongside a running L2 trace of each.
    The bilateral wrapper enforces an exact (x -> -x) symmetry on the
    commanded DM, so the OPD asymmetry should sit at the numerical
    floor; the PSF and obs asymmetries reveal whatever additional
    symmetry-breaking the env, detector, or noise injects between
    pupil and detector.
    """
    target = ep["target"]
    angle, r_frac, s_frac = target
    target_id = ep.get("target_id", -1)
    opds = ep["opd"]
    psfs = ep["raw_psf"]
    obs_unblind = ep["obs_unblind"]
    obs_b_imgs = [_prep_obs(o) for o in obs_unblind]
    # Use the unblinded view: the blinded one has the mirror region
    # zeroed by construction, which would trivially make it
    # asymmetric in a way that has nothing to do with the optics.
    H, W = obs_b_imgs[0].shape[-2:]
    T = len(ep["rewards"])
    rewards = ep["rewards"]
    cumulative = np.cumsum(rewards) if T else np.array([])
    strehls = ep["strehls"]

    ifov_arcsec = float(ep.get("ifov_arcsec", 1.0))
    foc_half = 0.5 * H * ifov_arcsec
    foc_extent = [-foc_half, foc_half, -foc_half, foc_half]
    pup_extent = [-1.0, 1.0, -1.0, 1.0]

    # Pre-compute traces so the y-axis bounds are stable across frames.
    l2_opd = np.array(
        [_relative_l2_asymmetry(opds[t]) for t in range(T + 1)],
        dtype=np.float64)
    l2_psf = np.array(
        [_relative_l2_asymmetry(psfs[t]) if psfs[t] is not None else 0.0
         for t in range(T + 1)], dtype=np.float64)
    l2_obs = np.array(
        [_relative_l2_asymmetry(obs_b_imgs[t]) for t in range(T + 1)],
        dtype=np.float64)

    def _log_bounds(arrs):
        cat = np.concatenate(arrs)
        x = cat[np.isfinite(cat) & (cat > 0)]
        if not x.size:
            return 1e-6, 1.0
        return max(float(x.min()) * 0.5, 1e-12), float(x.max()) * 2.0

    l2_lo, l2_hi = _log_bounds([l2_opd, l2_psf, l2_obs])
    timesteps = np.arange(T + 1)

    head_static = (f"target {target_id:02d}     "
                   f"angle = {angle:.1f}°     "
                   f"r = {r_frac:.3f}     "
                   f"size = {s_frac:.3f}   "
                   f"|   bilateral symmetry analysis")

    def _metrics_line(t):
        parts = [f"t = {t:>3d}"]
        if t > 0:
            parts.append(f"r = {rewards[t-1]:+.3f}")
            parts.append(f"Σr = {cumulative[t-1]:+.2f}")
            if strehls:
                parts.append(f"S = {strehls[t-1]:.3f}")
        parts.append(f"$L_2^\\mathrm{{OPD}}$ = {l2_opd[t]:.2e}")
        parts.append(f"$L_2^\\mathrm{{PSF}}$ = {l2_psf[t]:.2e}")
        parts.append(f"$L_2^\\mathrm{{obs}}$ = {l2_obs[t]:.2e}")
        return "     ".join(parts)

    # 3 rows × 2 cols. Left col = original image, right col = folded
    # difference. Row 0 = DM OPD, row 1 = raw PSF, row 2 = obs.
    # Fourth row (short) = L2 trace.
    fig_w, fig_h = 10.4, 12.4
    panel_h = 0.235
    row_tops = [0.88, 0.88 - panel_h - 0.02,
                0.88 - 2 * (panel_h + 0.02)]
    trace_top, trace_bot = 0.13, 0.045
    left_margin, right_margin = 0.065, 0.985
    inner_gap = 0.095
    col_w = (right_margin - left_margin - inner_gap) / 2.0
    box_left = lambda top: (left_margin, top - panel_h, col_w, panel_h)
    box_right = lambda top: (
        left_margin + col_w + inner_gap, top - panel_h, col_w, panel_h)
    box_trace = (left_margin, trace_bot,
                 right_margin - left_margin, trace_top - trace_bot)

    def _add_flush_colorbar(fig, ax, im, label, fmt_powerlimits=None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.06)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(label, fontsize=7.5, labelpad=2)
        if fmt_powerlimits is not None:
            cb.formatter.set_powerlimits(fmt_powerlimits)
            cb.update_ticks()
        return cb

    def _style_focal(ax, title):
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlabel("x (arcsec)", fontsize=7.5, labelpad=2)
        ax.set_ylabel("y (arcsec)", fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=6.5)

    def _style_pupil(ax, title):
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlabel("pupil x (R)", fontsize=7.5, labelpad=2)
        ax.set_ylabel("pupil y (R)", fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    def _imshow_signed(ax, img, extent, vlim=None):
        if vlim is None:
            vlim = max(float(np.nanmax(np.abs(img))), 1e-30)
        im = ax.imshow(
            img, cmap="RdBu_r", origin="lower", aspect="equal",
            vmin=-vlim, vmax=vlim, extent=extent)
        ax.axvline(0, color="black", lw=0.4, alpha=0.4)
        return im

    frames = []
    for t in range(T + 1):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        # --- Row 0: DM OPD + folded diff ---
        opd = opds[t]
        ax = fig.add_axes(box_left(row_tops[0]))
        opd_vmax = max(float(np.nanmax(np.abs(opd))), 1e-12)
        im = ax.imshow(opd, cmap="RdBu_r", origin="lower", aspect="equal",
                       vmin=-opd_vmax, vmax=opd_vmax, extent=pup_extent)
        _style_pupil(ax, "DM OPD")
        _add_flush_colorbar(fig, ax, im,
                            label="surface (m)",
                            fmt_powerlimits=(-2, 2))

        ax = fig.add_axes(box_right(row_tops[0]))
        d_opd = _fold_diff_lr(opd)
        im = _imshow_signed(ax, d_opd, pup_extent)
        _style_pupil(ax, r"DM OPD  $-$  flip$_x$(DM OPD)")
        _add_flush_colorbar(fig, ax, im,
                            label="Δ surface (m)",
                            fmt_powerlimits=(-2, 2))

        # --- Row 1: raw PSF + folded diff ---
        psf = psfs[t] if psfs[t] is not None else np.zeros((H, W))
        ax = fig.add_axes(box_left(row_tops[1]))
        pmax = max(float(np.max(psf)), 1e-30)
        pflo = max(pmax * 1e-8, 1e-30)
        im = ax.imshow(np.maximum(psf, pflo), cmap="inferno",
                       norm=mcolors.LogNorm(vmin=pflo, vmax=pmax),
                       origin="lower", aspect="equal", extent=foc_extent)
        _style_focal(ax, "raw PSF")
        _add_flush_colorbar(fig, ax, im, label="intensity (log)")

        ax = fig.add_axes(box_right(row_tops[1]))
        d_psf = _fold_diff_lr(psf)
        im = _imshow_signed(ax, d_psf, foc_extent)
        _style_focal(ax, r"raw PSF  $-$  flip$_x$(raw PSF)")
        _add_flush_colorbar(fig, ax, im, label="Δ intensity")

        # --- Row 2: detector obs + folded diff ---
        obs = obs_b_imgs[t]
        ax = fig.add_axes(box_left(row_tops[2]))
        omax = max(float(np.max(obs)), 2.0)
        im = ax.imshow(np.maximum(obs, 1.0), cmap="inferno",
                       norm=mcolors.LogNorm(vmin=1.0, vmax=omax),
                       origin="lower", aspect="equal", extent=foc_extent)
        _style_focal(ax, "detector obs (unblinded)")
        _add_flush_colorbar(fig, ax, im, label="DN (log)")

        ax = fig.add_axes(box_right(row_tops[2]))
        d_obs = _fold_diff_lr(obs)
        im = _imshow_signed(ax, d_obs, foc_extent)
        _style_focal(ax, r"obs  $-$  flip$_x$(obs)")
        _add_flush_colorbar(fig, ax, im, label="Δ DN")

        # --- Bottom row: L2 traces ---
        ax_l2 = fig.add_axes(box_trace)
        ax_l2.set_yscale("log")
        ax_l2.set_ylim(l2_lo, l2_hi)
        ax_l2.set_xlim(0, max(T, 1))
        for arr, color, label in (
                (l2_opd, "#1f77b4", "DM OPD"),
                (l2_psf, "#d62728", "raw PSF"),
                (l2_obs, "#2ca02c", "detector obs")):
            ax_l2.plot(timesteps, arr,
                       color=color, lw=0.7, alpha=0.35)
            ax_l2.plot(timesteps[:t + 1], arr[:t + 1],
                       color=color, lw=1.6, label=label)
            if np.isfinite(arr[t]):
                ax_l2.plot([t], [arr[t]], "o",
                           color=color, markersize=5,
                           mec="black", mew=0.4)
        ax_l2.grid(True, which="both", alpha=0.25, lw=0.4)
        ax_l2.tick_params(labelsize=7)
        ax_l2.set_xlabel("step", fontsize=8, labelpad=2)
        ax_l2.set_ylabel(r"$L_2$ asymmetry", fontsize=8, labelpad=2)
        ax_l2.set_title(
            r"$L_2(I, t) \;=\; "
            r"\|\,I(x,y,t) - I(W{-}1{-}x,y,t)\,\|_2 \;/\; "
            r"\|\,I(\cdot,t)\,\|_2$"
            r"     (dimensionless,  $0 = $ symmetric)",
            fontsize=9, pad=3)
        ax_l2.legend(loc="upper right", fontsize=7, frameon=True, ncol=3)

        fig.text(left_margin, 0.975, head_static,
                 ha="left", va="center", fontsize=10.5,
                 weight="semibold")
        fig.text(left_margin, 0.950, _metrics_line(t),
                 ha="left", va="center", fontsize=9, family="monospace")

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
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions from the policy's Normal(mean, std) "
             "instead of using the deterministic mean. Matches the "
             "training-time action distribution. Useful for "
             "disambiguating deterministic-mean failure from "
             "deeper train-eval mismatches: if stochastic rollout "
             "achieves training-level Strehl but deterministic does "
             "not, the policy mean is at a saddle.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint path. When set, --sweep-dir / "
             "--target-id selection and --prefer-latest are ignored; "
             "the rollout uses this exact file. Useful for rolling a "
             "specific update_NNN.pt step.")
    parser.add_argument(
        "--zero-init", action="store_true",
        help="Zero every randomised initial disturbance (DM "
             "perturbation, segment piston/tip/tilt). Sanity check "
             "for the bilateral-symmetry GIF: if PSF asymmetry "
             "survives zero-init the symmetry break is downstream of "
             "the random draw (env pipeline), not from the init.")
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

    # Explicit-checkpoint path: bypass sweep-dir / per-target search
    # and roll exactly one checkpoint (still routed through whichever
    # target_id was selected on the CLI for target geometry).
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"Error: --checkpoint {args.checkpoint!r} not found")
            sys.exit(1)
        if args.target_id is None:
            print("Error: --checkpoint requires --target-id so the env "
                  "can be built with the correct dark-hole geometry.")
            sys.exit(1)
        sweep_dir = args.sweep_dir or os.path.dirname(
            os.path.dirname(os.path.dirname(
                os.path.abspath(args.checkpoint))))
    else:
        sweep_dir = args.sweep_dir or _latest_sweep_dir()
        if sweep_dir is None or not os.path.isdir(sweep_dir):
            print(f"Error: sweep dir not found "
                  f"(--sweep-dir or newest {_SWEEP_PREFIX}* under "
                  f"{_DEFAULT_SWEEP_ROOT}/)")
            sys.exit(1)
        if args.sweep_dir is None:
            print(f"--sweep-dir not given; using newest: {sweep_dir}")

    suffix = "_grid" + ("_zeroinit" if args.zero_init else "")
    output_dir = args.output_dir or os.path.join(
        "test_output", f"{os.path.basename(sweep_dir.rstrip('/'))}{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sweep dir:  {sweep_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Targets:    {target_indices}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint} (explicit)")
    print()

    summary = []
    for i in target_indices:
        target = targets[i]
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            try:
                ckpt_path = _resolve_checkpoint(
                    sweep_dir, i, prefer_latest=args.prefer_latest)
            except FileNotFoundError as e:
                print(f"  target {i:>2}: SKIP -- {e}")
                continue

        # Peek at the checkpoint's config to decide whether the policy
        # was trained with the bilateral wrapper (action space halved,
        # obs blind mask) or against the unwrapped full-DM env, AND
        # to recover the exact env_kwargs the policy was trained
        # against. Importing the module-level ENV_KWARGS at the top of
        # this file is a train-eval mismatch trap: the kwargs there
        # are from a *different* training script and contain different
        # values for dm_incremental_control, env_action_scale,
        # init_dm_micron_std, init_dm_symmetric, and reward weights.
        # Running the policy under those produces wildly off-
        # distribution behaviour (e.g., Strehl crashes from 0.89
        # training-time to 0.02 eval-time).
        _ckpt_peek = torch.load(
            ckpt_path, map_location="cpu", weights_only=False)
        _ck_cfg = _ckpt_peek.get("config", {}) if _ckpt_peek else {}
        bilateral = bool(_ck_cfg.get("bilateral_dm", True))
        bilateral_mode = str(_ck_cfg.get(
            "bilateral_dm_mode", "fixed_vertical"))
        freeze_segments = bool(_ck_cfg.get(
            "bilateral_freeze_segments", True))
        ckpt_env_kwargs = _ck_cfg.get("env_kwargs", None)
        del _ckpt_peek  # release memory; _load_agent reloads
        if ckpt_env_kwargs is None:
            print(f"  target {i:>2}: WARNING -- checkpoint missing "
                  f"env_kwargs; falling back to module ENV_KWARGS "
                  f"(likely train/eval mismatch)")
        print(f"  target {i:>2}: mode = "
              f"{'bilateral (' + bilateral_mode + ')' if bilateral else 'full-DM'}")

        env, base = _build_env(
            target, args.max_steps, args.device,
            bilateral=bilateral, bilateral_mode=bilateral_mode,
            freeze_segments=freeze_segments,
            base_env_kwargs=ckpt_env_kwargs,
            zero_init=args.zero_init)
        agent, config = _load_agent(ckpt_path, env, args.device)
        td = int(config.get("target_dim", 0))
        if td == 0:
            print(f"  target {i:>2}: WARNING -- target_dim=0 in checkpoint")
        ep = run_episode(
            agent, env, base, target, args.seed, args.device,
            max_steps=args.max_steps, stochastic=args.stochastic)
        ep["target_id"] = i

        gif_path = os.path.join(output_dir, f"target_{i:02d}.gif")
        render_gif(ep, gif_path, dpi=args.dpi,
                   frame_duration=args.frame_duration)
        sym_path = os.path.join(output_dir, f"target_{i:02d}_symmetry.gif")
        render_symmetry_gif(ep, sym_path, dpi=args.dpi,
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
