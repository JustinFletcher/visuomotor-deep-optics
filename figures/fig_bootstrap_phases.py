#!/usr/bin/env python3
"""
Generate bootstrap phase gallery: for each of 15 incremental bootstrapping
phases, show the initial OPD (with excluded segments off-axis) and the
target focal-plane image (the goal state stored in the env).

Output: figures/bootstrap_gallery/ with individual PNGs for LaTeX subfigure,
        plus a combined strip figure.

Usage:
    poetry run python figures/fig_bootstrap_phases.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT_DIR = os.path.join(os.path.dirname(__file__), "bootstrap_gallery")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_PHASES = 15
FOCAL_PLANE_PIXELS = 256
SEED = 42


def make_bootstrap_env(phased_count):
    """Create an ELF OptomechEnv in bootstrap mode for the given phase."""
    from optomech.optomech.optomech_v4 import OptomechEnv

    kwargs = {
        "object_type": "single",
        "aperture_type": "elf",
        "focal_plane_image_size_pixels": FOCAL_PLANE_PIXELS,
        "observation_mode": "image_only",
        "observation_window_size": 1,
        "wavelength": 1e-6,
        "oversampling_factor": 8,
        "bandwidth_nanometers": 200.0,
        "bandwidth_sampling": 2,
        "object_plane_extent_meters": 1.0,
        "object_plane_distance_meters": 1.0,
        "num_atmosphere_layers": 0,
        "ao_interval_ms": 100.0,
        "control_interval_ms": 100.0,
        "frame_interval_ms": 100.0,
        "decision_interval_ms": 100.0,
        "max_episode_steps": 10,
        "command_secondaries": True,
        "command_tip_tilt": True,
        "command_tensioners": False,
        "command_dm": False,
        "ao_loop_active": False,
        "incremental_control": True,
        "discrete_control": False,
        "action_type": "none",
        "actuator_noise": False,
        "env_action_scale": 0.1,
        "max_piston_correction_micron": 10.0,
        "max_tip_correction_arcsec": 20.0,
        "max_tilt_correction_arcsec": 20.0,
        "get_disp_corr_max_piston_micron": 3.0,
        "get_disp_corr_max_tip_arcsec": 20.0,
        "get_disp_corr_max_tilt_arcsec": 20.0,
        "init_differential_motion": True,
        "simulate_differential_motion": False,
        "model_wind_diff_motion": True,
        "model_gravity_diff_motion": False,
        "model_temp_diff_motion": False,
        "model_ao": False,
        "init_wind_piston_micron_std": 3.0,
        "init_wind_piston_clip_m": 4e-6,
        "init_wind_tip_arcsec_std_tt": 0.05,
        "init_wind_tilt_arcsec_std_tt": 0.05,
        "reward_function": "factored",
        "reward_weight_strehl": 0.0,
        "reward_weight_shape": 0.0,
        "reward_weight_centered_strehl": 1.0,
        "centering_mode": "circular",
        "centering_radius_fraction": 0.25,
        "action_penalty": True,
        "action_penalty_weight": 0.5,
        "holding_bonus_weight": 1.0,
        "holding_bonus_min_reward": -1.0,
        "holding_bonus_threshold": -0.7,
        "detector_power_watts": 1e-12,
        "detector_quantum_efficiency": 0.8,
        "detector_gain_e_per_dn": 0.5,
        "detector_max_dn": 65535,
        "render": False,
        "silence": True,
        # Bootstrap flags
        "bootstrap_phase": True,
        "bootstrap_phased_count": phased_count,
    }
    return OptomechEnv(**kwargs)


def extract_bootstrap(env, phased_count):
    """Extract OPD and target image from a bootstrap environment.

    Returns dict with:
        opd_masked: OPD in microns (NaN outside aperture)
        target_image: the perfect/target focal plane image for this phase
        pupil_diameter_mm: pupil extent in mm
        focal_size_lod: focal plane extent in lambda/D
    """
    obs, info = env.reset(seed=SEED)
    osys = env.optical_system

    # --- OPD from current actuator state (post-reset, with perturbations) ---
    aperture = osys._aperture_t.cpu().numpy()
    ap_mask = np.abs(aperture) > 0.01
    actuators = osys._actuators_t.cpu().numpy()
    influence = osys._influence_t.cpu().numpy()
    opd = np.einsum("i,ihw->hw", actuators, influence)
    opd_masked = np.where(ap_mask, opd * 1e6, np.nan)  # microns

    # --- Target image: the env's internal perfect image for this phase ---
    # This is the goal state: phased_count+1 segments co-phased, rest off.
    target_image = env._perfect_image_dn.copy()

    # --- Physical extents ---
    pupil_diameter_mm = float(
        osys.pupil_grid.x.max() - osys.pupil_grid.x.min()) * 1e3
    pupil_diameter_m = pupil_diameter_mm * 1e-3
    focal_length_m = osys._focal_length
    wavelength_m = osys.wavelength
    focal_size_m = osys._focal_plane_image_size_meters
    lam_over_d_m = wavelength_m * focal_length_m / pupil_diameter_m
    focal_size_lod = focal_size_m / lam_over_d_m

    env.close()
    return {
        "opd_masked": opd_masked,
        "target_image": target_image,
        "pupil_diameter_mm": pupil_diameter_mm,
        "focal_size_lod": focal_size_lod,
    }


def save_img(data, path, cmap="gray", vmin=None, vmax=None,
             log_scale=False, extent=None,
             xlabel=None, ylabel=None, cb_label=None, title=None):
    """Save a single image with physical axes, colorbar, and LaTeX labels."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    d = data.copy()
    if log_scale:
        d = np.log10(d + 1e-12)
    im = ax.imshow(d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest", extent=extent)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    if cb_label is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(cb_label)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def main():
    print("Generating bootstrap phase gallery for ELF (15 phases)...\n")

    # Collect all phase data first
    all_data = []
    for phase in range(NUM_PHASES):
        print(f"Phase {phase:2d} (phased_count={phase})...")
        env = make_bootstrap_env(phased_count=phase)
        data = extract_bootstrap(env, phased_count=phase)
        all_data.append(data)

    # Use consistent extents from phase 0
    d0 = all_data[0]
    d_mm = d0["pupil_diameter_mm"]
    f_lod = d0["focal_size_lod"]
    pup_ext = [-d_mm / 2, d_mm / 2, -d_mm / 2, d_mm / 2]
    foc_ext = [-f_lod / 2, f_lod / 2, -f_lod / 2, f_lod / 2]
    pup_xl = r"$x$ (mm)"
    pup_yl = r"$y$ (mm)"
    foc_xl = r"$x$ ($\lambda/D$)"
    foc_yl = r"$y$ ($\lambda/D$)"

    # --- Save individual images ---
    print("\nSaving individual images...")
    for phase, data in enumerate(all_data):
        # OPD
        vm = np.nanmax(np.abs(data["opd_masked"]))
        vm = max(vm, 0.01)
        save_img(data["opd_masked"],
                 os.path.join(OUT_DIR, f"phase_{phase:02d}_opd.png"),
                 cmap="RdBu_r", vmin=-vm, vmax=vm, extent=pup_ext,
                 xlabel=pup_xl, ylabel=pup_yl,
                 cb_label=r"OPD ($\mu$m)",
                 title=f"Phase {phase}: initial OPD")

        # Target image (log scale for dynamic range)
        save_img(data["target_image"],
                 os.path.join(OUT_DIR, f"phase_{phase:02d}_target.png"),
                 cmap="inferno", log_scale=True, extent=foc_ext,
                 xlabel=foc_xl, ylabel=foc_yl,
                 cb_label=r"$\log_{10}$ intensity (arb.)",
                 title=f"Phase {phase}: target ({phase + 1} seg)")

    # --- Combined strip figure ---
    print("\nSaving combined strip figure...")
    fig, axes = plt.subplots(2, NUM_PHASES, figsize=(3.0 * NUM_PHASES, 6.5))

    for phase, data in enumerate(all_data):
        # Top row: OPD
        ax_opd = axes[0, phase]
        opd = data["opd_masked"]
        vm = np.nanmax(np.abs(opd))
        vm = max(vm, 0.01)
        im_opd = ax_opd.imshow(opd, cmap="RdBu_r", origin="lower",
                                vmin=-vm, vmax=vm, aspect="equal",
                                interpolation="nearest", extent=pup_ext)
        ax_opd.set_title(f"Phase {phase}", fontsize=9)
        ax_opd.tick_params(labelsize=6)
        if phase == 0:
            ax_opd.set_ylabel(r"OPD ($\mu$m)", fontsize=9)
        else:
            ax_opd.set_yticklabels([])

        # Bottom row: target image
        ax_tgt = axes[1, phase]
        tgt = np.log10(data["target_image"] + 1e-12)
        im_tgt = ax_tgt.imshow(tgt, cmap="inferno", origin="lower",
                                aspect="equal", interpolation="nearest",
                                extent=foc_ext)
        ax_tgt.tick_params(labelsize=6)
        if phase == 0:
            ax_tgt.set_ylabel(r"Target ($\log_{10}$ I)", fontsize=9)
        else:
            ax_tgt.set_yticklabels([])

    fig.tight_layout(h_pad=0.3, w_pad=0.1)
    strip_path = os.path.join(OUT_DIR, "bootstrap_strip.png")
    fig.savefig(strip_path, dpi=150, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  {strip_path}")

    print(f"\nAll images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
