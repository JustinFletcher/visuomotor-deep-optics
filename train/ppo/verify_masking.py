"""Visual verification of DM control / focal-plane masking for both
the bilateral_dm wrapper and the full-DM (no wrapper) scenarios.

Tests we need:
  1. Full DM: action space matches full DM, no mask in obs, contrast
     hole at the target only.
  2. Bilateral fixed_vertical: action space = n_dm // 2, each
     controlled actuator on the right pupil half (x > 0) is paired
     with a mirror partner on the left half. Blind mask sits at the
     bilateral mirror of the target hole (NOT on top of it).

Visual: render focal-plane PSF with the target hole and the blind
mask overlaid as ring outlines, plus a pupil-plane figure of the
DM actuator grid coloured by controlled / mirror / unused.

The critical geometric assertion that justifies the wrapper is that
a phase pattern symmetric under (x, y) -> (-x, y) on the DM (pupil
plane) produces a PSF intensity that is symmetric under the same
operation in the focal plane: that's just |FT[f]|^2 inheriting f's
mirror symmetry. The Fourier propagator in v5 is M1 @ E @ M2 * scale
with matrices inherited from HCIPy; it has no built-in pupil/focal
inversion (verified visually by this script's "DM tilt -> peak side"
check).

Run:

    poetry run python train/ppo/verify_masking.py --target-id 4
    poetry run python train/ppo/verify_masking.py --target-id 0
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.ppo.bilateral_dm import BilateralDMVectorEnv                 # noqa: E402
from train.ppo.launch_static_dark_hole import build_grid                # noqa: E402
from train.ppo.train_ppo_elf_dark_hole import ELF_DARK_HOLE_ENV_KWARGS  # noqa: E402


def build_env(angle_deg, r_frac, s_frac, with_wrapper, device="cpu"):
    """Build a fresh env at the given target. Always uses v5 absolute DM
    with command_secondaries=True (segments) and command_dm=True."""
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    env_kwargs = dict(ELF_DARK_HOLE_ENV_KWARGS)
    env_kwargs.update(dict(
        dark_hole=True,
        dark_hole_angular_location_degrees=float(angle_deg),
        dark_hole_location_radius_fraction=float(r_frac),
        dark_hole_size_radius=float(s_frac),
        dark_hole_randomize_on_reset=False,
        command_dm=True,
        dm_incremental_control=False,
        init_dm_micron_std=0.0,            # no noise: clean test patterns
        init_piston_micron_std=0.0,
        init_tip_arcsec_std=0.0,
        init_tilt_arcsec_std=0.0,
        silence=True,
        observation_window_size=1,
        reward_vector_enabled=False,
        reward_weight_centered_strehl=1.0,
    ))
    with contextlib.redirect_stdout(io.StringIO()):
        base = BatchedOptomechEnv(num_envs=1, device=device, **env_kwargs)
        if with_wrapper:
            env = BilateralDMVectorEnv(base, freeze_segments=True,
                                       mode="fixed_vertical")
        else:
            env = base
    return env, base


def make_test_dm_action(env, base, kind):
    """Return an action vector for the env. `kind` describes the DM
    test pattern in pupil-plane terms; segment slice (if present) is
    zero. Vector lives in the env's exposed action space (so for the
    wrapper it's the half-size action; for the bare base env it's the
    full action).
    """
    if isinstance(env, BilateralDMVectorEnv):
        n_seg = 0 if env._freeze_segments else env._n_seg
        n_dm_slot = env._n_half
        xy_ctrl = base._dm_actuator_xy_t[env._controlled_idx[0]].cpu().numpy()
        a_dm = _pattern_for_positions(xy_ctrl, kind)
        if env._freeze_segments:
            return a_dm.reshape(1, -1)
        return np.concatenate(
            [np.zeros(n_seg, dtype=np.float32), a_dm]
        ).reshape(1, -1)
    # Full-DM (no wrapper): action layout is [seg ..., dm ...].
    n_seg = base._n_seg_actions
    a_seg = np.zeros(n_seg, dtype=np.float32)
    xy_all = base._dm_actuator_xy_t.cpu().numpy()
    a_dm = _pattern_for_positions(xy_all, kind)
    return np.concatenate([a_seg, a_dm]).reshape(1, -1)


def _pattern_for_positions(xy, kind):
    """Compute a per-actuator action value based on actuator pupil
    positions xy [A, 2] in normalised pupil coords [-1, 1].
    kind:
      'rightside_positive_tip': a smooth +x tilt over the right pupil
          half (positive on the right side, zero on the left). Should
          deflect the PSF peak along +x_focal (or -x_focal if the MFT
          inverts).
      'leftside_positive_tip': mirror of above on the left.
      'zero':  flat (sanity baseline).
      'speckle_on_axis_for_target': a sinusoidal phase tuned to push
          flux off-axis along the target radial direction.
    """
    A = xy.shape[0]
    if kind == "zero":
        return np.zeros(A, dtype=np.float32)
    if kind == "rightside_positive_tip":
        # action units are fractional stroke (-1, 1). 0.2 over the
        # right pupil half is plenty to move the PSF visibly.
        v = np.zeros(A, dtype=np.float32)
        v[xy[:, 0] > 0] = 0.4
        return v
    if kind == "leftside_positive_tip":
        v = np.zeros(A, dtype=np.float32)
        v[xy[:, 0] < 0] = 0.4
        return v
    if kind == "linear_tip_x":
        # A clean linear x-tilt over the WHOLE DM. Should send the PSF
        # peak in a unique focal direction; comparing that direction to
        # +x_focal tells us whether the MFT has an axis inversion.
        # We push well past the nominal [-1, 1] stroke envelope (v5
        # doesn't clamp absolute-mode DM actions) so the PSF shift is
        # large enough to be unmistakable on a 256x256 frame — at amp 10
        # the peak moves ~63 px from centre, very hard to miss; at
        # amp 1 the shift is ~5 px and gets lost in the diffraction
        # rings around the original centre. Pure visualisation only;
        # not a physically achievable wavefront on a real DM.
        return (xy[:, 0]).astype(np.float32) * 10.0
    raise ValueError(kind)


def render_check(target_id, outdir):
    targets = build_grid()
    angle, r_frac, s_frac = targets[target_id]
    print(f"\n=== target {target_id}: angle={angle}, r={r_frac}, s={s_frac} ===")

    # --- Scenario A: full DM, no wrapper -----------------------------
    envA, baseA = build_env(angle, r_frac, s_frac, with_wrapper=False)
    envA.reset(seed=0)
    n_seg_A = baseA._n_seg_actions
    n_dm_A = baseA._n_dm_acts
    print(f"FULL DM:")
    print(f"  action_space.shape = {envA.single_action_space.shape}"
          f"  (= {n_seg_A} seg + {n_dm_A} dm)")
    print(f"  hole_mask sum (pixels in target hole): "
          f"{int(baseA._hole_mask_t[0].sum())}")
    has_mask_obs_A = hasattr(envA, "mask_obs")
    print(f"  has mask_obs? {has_mask_obs_A}")
    # Run a few clear test patterns
    frames_A = {}
    for kind in ("zero", "linear_tip_x"):
        envA, baseA = build_env(angle, r_frac, s_frac, with_wrapper=False)
        envA.reset(seed=0)
        a = make_test_dm_action(envA, baseA, kind)
        obs, _, _, _, info = envA.step(a)
        frames_A[kind] = (obs[0, 0].copy(),
                          baseA._hole_mask_t[0].cpu().numpy(),
                          baseA._dm_actuators_t[0].cpu().numpy().copy(),
                          baseA._dm_actuator_xy_t.cpu().numpy().copy())

    # --- Scenario B: bilateral_dm wrapper, fixed_vertical ------------
    envB, baseB = build_env(angle, r_frac, s_frac, with_wrapper=True)
    envB.reset(seed=0)
    n_half = envB._n_half
    print(f"BILATERAL (fixed_vertical, freeze_segments=True):")
    print(f"  action_space.shape = {envB.single_action_space.shape}"
          f"  (= n_dm // 2 = {n_half})")
    print(f"  has mask_obs? {hasattr(envB, 'mask_obs')}")
    print(f"  hole_mask sum:  {int(baseB._hole_mask_t[0].sum())}")
    print(f"  blind_mask sum: {int(envB._blind_mask[0].sum())}")
    # Check: blind and target masks shouldn't overlap
    overlap = (baseB._hole_mask_t[0] & envB._blind_mask[0]).sum().item()
    print(f"  hole ∩ blind overlap pixels: {overlap}"
          + ("  <-- BUG" if overlap > 0 else "  OK"))
    # Pull partition once (since target_vec is static)
    ctrl_idx = envB._controlled_idx[0].cpu().numpy()
    mirror_idx = envB._mirror_partner_idx[0].cpu().numpy()
    xy_all = baseB._dm_actuator_xy_t.cpu().numpy()
    ctrl_xy = xy_all[ctrl_idx]
    mirror_xy = xy_all[mirror_idx]
    print(f"  ctrl_xy.x range:   [{ctrl_xy[:, 0].min():+.3f}, "
          f"{ctrl_xy[:, 0].max():+.3f}]")
    print(f"  mirror_xy.x range: [{mirror_xy[:, 0].min():+.3f}, "
          f"{mirror_xy[:, 0].max():+.3f}]")
    # Confirm mirror_idx really reflects ctrl_idx in x
    refl_err = float(np.abs(ctrl_xy + mirror_xy * np.array([1, -1])).max())
    # Actually we want ctrl(x,y) <-> mirror(-x, y): test |ctrl.x + mirror.x| ~ 0 and |ctrl.y - mirror.y| ~ 0
    err_x = float(np.abs(ctrl_xy[:, 0] + mirror_xy[:, 0]).max())
    err_y = float(np.abs(ctrl_xy[:, 1] - mirror_xy[:, 1]).max())
    print(f"  partner mismatch:  max |x+x'|={err_x:.4f}  "
          f"max |y-y'|={err_y:.4f}  (grid pitch={2.0/35:.4f})")

    frames_B = {}
    for kind in ("zero", "rightside_positive_tip"):
        envB, baseB = build_env(angle, r_frac, s_frac, with_wrapper=True)
        envB.reset(seed=0)
        a = make_test_dm_action(envB, baseB, kind)
        obs_unmasked, _, _, _, info = envB.step(a)
        obs_masked = envB.mask_obs(obs_unmasked)
        frames_B[kind] = (
            obs_unmasked[0, 0].copy(),
            obs_masked[0, 0].copy(),
            baseB._hole_mask_t[0].cpu().numpy(),
            envB._blind_mask[0].cpu().numpy(),
            baseB._dm_actuators_t[0].cpu().numpy().copy(),
            xy_all, ctrl_idx, mirror_idx,
        )

    # --- Plot everything --------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    H = baseA._H
    loc_px = int(r_frac * H / 2)
    size_px = int(s_frac * H / 2)
    target_cx = int(H / 2 + loc_px * np.cos(np.deg2rad(angle)))
    target_cy = int(H / 2 + loc_px * np.sin(np.deg2rad(angle)))
    blind_cx_expected = int(H / 2 - loc_px * np.cos(np.deg2rad(angle)))
    blind_cy_expected = int(H / 2 + loc_px * np.sin(np.deg2rad(angle)))
    print(f"  target hole center (px): ({target_cx}, {target_cy})")
    print(f"  blind  mask center (px): ({blind_cx_expected}, "
          f"{blind_cy_expected})  [expected from fixed-vertical mirror]")

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        f"DM/mask verification | target {target_id}: "
        f"angle={angle}°, r={r_frac}, s={s_frac}",
        fontsize=12)

    # Row 0: full DM, zero action vs x-tilt
    for j, (kind, ax_title) in enumerate(
            [("zero", "Full DM: zero action"),
             ("linear_tip_x", "Full DM: linear +x tilt over WHOLE DM")]):
        ax_pupil = axes[0, j * 2]
        ax_focal = axes[0, j * 2 + 1]
        frame, hole, dm_acts, xy = frames_A[kind]
        # Pupil-plane: DM actuator commands. Auto-range each panel so a
        # small tilt fills the colormap instead of washing out against
        # the full stroke envelope. Floor the range at 1% stroke so the
        # zero-action panel doesn't try to expand machine-epsilon noise.
        vmax = max(float(np.abs(dm_acts).max()), 1e-8)
        scat = ax_pupil.scatter(xy[:, 0], xy[:, 1], c=dm_acts, cmap="RdBu_r",
                                s=10, vmin=-vmax, vmax=vmax)
        ax_pupil.set_aspect("equal")
        ax_pupil.set_title(f"PUPIL  {ax_title}")
        ax_pupil.set_xlabel("pupil x (norm)")
        ax_pupil.set_ylabel("pupil y (norm)")
        ax_pupil.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax_pupil.axhline(0, color="k", lw=0.5, alpha=0.4)
        plt.colorbar(scat, ax=ax_pupil, fraction=0.046)
        # Focal-plane. Linear colormap with a tight ceiling (99.9th
        # percentile) makes the displaced PSF peak pop visually --
        # log compresses the dynamic range too much and the central
        # diffraction speckle ends up the same brightness as the
        # actual shifted core.
        vmax_focal = float(np.percentile(frame, 99.9))
        ax_focal.imshow(frame, origin="lower", cmap="inferno",
                        vmin=0, vmax=vmax_focal)
        ax_focal.add_patch(Circle((target_cx, target_cy), size_px,
                                  fill=False, color="lime", lw=1.5,
                                  label="target hole"))
        ax_focal.axvline(H / 2, color="cyan", lw=0.6, alpha=0.5)
        ax_focal.axhline(H / 2, color="cyan", lw=0.6, alpha=0.5)
        ax_focal.set_title(f"FOCAL  {ax_title}")
        ax_focal.set_xlabel("focal x (px)")
        ax_focal.set_ylabel("focal y (px)")
        peak = np.unravel_index(int(np.argmax(frame)), frame.shape)
        if kind == "linear_tip_x":
            ax_focal.annotate(
                "", xy=(peak[1], peak[0]), xytext=(H / 2, H / 2),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
        ax_focal.plot(peak[1], peak[0], "x", color="cyan", ms=14, mew=2.5,
                      label=f"peak ({peak[1]}, {peak[0]})")
        ax_focal.legend(loc="upper right", fontsize=7)

    # Row 1: bilateral, zero action (just verify masks land right)
    frame_u, frame_m, hole, blind, dm_acts, xy, ctrl_idx, mirror_idx \
        = frames_B["zero"]
    # Pupil-plane: colour by partition role
    role = np.full(xy.shape[0], 2, dtype=int)   # 2 = unused (axis)
    role[ctrl_idx] = 0   # 0 = controlled
    role[mirror_idx] = 1   # 1 = mirror
    role_colours = np.array([[1.0, 0.4, 0.4],   # red = controlled
                             [0.4, 0.4, 1.0],   # blue = mirror
                             [0.7, 0.7, 0.7]])  # grey = unused
    ax_pupil = axes[1, 0]
    ax_pupil.scatter(xy[:, 0], xy[:, 1], c=role_colours[role], s=8)
    ax_pupil.set_aspect("equal")
    ax_pupil.set_title("PUPIL  Bilateral partition (zero action)\n"
                       "red=controlled, blue=mirror, grey=unused")
    ax_pupil.axvline(0, color="k", lw=0.8)
    ax_pupil.set_xlabel("pupil x (norm)")
    ax_pupil.set_ylabel("pupil y (norm)")

    ax = axes[1, 1]
    ax.imshow(np.log10(frame_u.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5, label="target hole"))
    ax.add_patch(Circle((blind_cx_expected, blind_cy_expected), size_px,
                        fill=False, color="magenta", lw=1.5,
                        label="blind mask (expected)"))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  Bilateral, zero action — UNMASKED")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1, 2]
    # Show the blind mask itself overlaid as a translucent red layer
    rgba = np.zeros((H, H, 4))
    rgba[..., 0] = 1.0
    rgba[..., 3] = blind.astype(float) * 0.6
    ax.imshow(np.log10(frame_u.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.imshow(rgba, origin="lower")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  red overlay = ACTUAL blind_mask\n"
                 "(should sit at the magenta ring above)")

    ax = axes[1, 3]
    ax.imshow(np.log10(frame_m.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5, label="target hole"))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  obs AFTER mask_obs (what policy sees)")
    ax.legend(loc="upper right", fontsize=7)

    # Row 2: bilateral, right-pupil tip, to see where focal energy lands
    frame_u, frame_m, hole, blind, dm_acts, xy, ctrl_idx, mirror_idx \
        = frames_B["rightside_positive_tip"]
    ax_pupil = axes[2, 0]
    vmax_b = max(float(np.abs(dm_acts).max()), 1e-8)
    sc = ax_pupil.scatter(xy[:, 0], xy[:, 1], c=dm_acts, cmap="RdBu_r",
                          s=10, vmin=-vmax_b, vmax=vmax_b)
    ax_pupil.set_aspect("equal")
    ax_pupil.axvline(0, color="k", lw=0.8)
    ax_pupil.set_title("PUPIL  Symmetric DM from right-half action\n"
                       "(wrapper has mirrored to the left half)")
    ax_pupil.set_xlabel("pupil x (norm)")
    ax_pupil.set_ylabel("pupil y (norm)")
    plt.colorbar(sc, ax=ax_pupil, fraction=0.046)

    ax = axes[2, 1]
    ax.imshow(np.log10(frame_u.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5, label="target hole"))
    ax.add_patch(Circle((blind_cx_expected, blind_cy_expected), size_px,
                        fill=False, color="magenta", lw=1.5,
                        label="blind mask"))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  UNMASKED — PSF after symmetric tip")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[2, 2]
    rgba = np.zeros((H, H, 4))
    rgba[..., 0] = 1.0
    rgba[..., 3] = blind.astype(float) * 0.6
    ax.imshow(np.log10(frame_u.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.imshow(rgba, origin="lower")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  actual blind_mask overlay")

    ax = axes[2, 3]
    ax.imshow(np.log10(frame_m.clip(min=1e-3) + 1), origin="lower",
              cmap="viridis")
    ax.add_patch(Circle((target_cx, target_cy), size_px, fill=False,
                        color="lime", lw=1.5, label="target hole"))
    ax.axvline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.axhline(H / 2, color="white", lw=0.5, alpha=0.4)
    ax.set_title("FOCAL  obs AFTER mask_obs (policy view)")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, f"verify_target_{target_id:02d}.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", type=int, nargs="*",
                    default=[0, 3, 4, 6, 13],
                    help="Targets to verify (default: a few spread "
                    "around the grid).")
    ap.add_argument("--outdir", type=str, default="test_output/verify_masking")
    args = ap.parse_args()

    paths = []
    for tid in args.target_id:
        paths.append(render_check(tid, args.outdir))
    print("\nAll figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
