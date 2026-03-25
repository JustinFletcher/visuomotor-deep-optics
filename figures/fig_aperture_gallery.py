#!/usr/bin/env python3
"""
Generate aperture gallery figure: nanoelf, nanoelfplus, elf pupil planes
with and without aberrations, plus corresponding focal-plane images.

Output: figures/aperture_gallery/ with individual PNGs for LaTeX subfigure.

Usage:
    poetry run python figures/fig_aperture_gallery.py
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
from matplotlib.colors import Normalize, TwoSlopeNorm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT_DIR = os.path.join(os.path.dirname(__file__), "aperture_gallery")
os.makedirs(OUT_DIR, exist_ok=True)


def make_env(aperture_type, tip_std=0.0, tilt_std=0.0, piston_std=0.0):
    """Create an OptomechEnv with the given aperture and optional aberrations."""
    from optomech.optomech.optomech_v4 import OptomechEnv

    kwargs = {
        "object_type": "single",
        "aperture_type": aperture_type,
        "focal_plane_image_size_pixels": 512,
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
        "max_tip_correction_arcsec": 10.0,
        "max_tilt_correction_arcsec": 10.0,
        "get_disp_corr_max_piston_micron": 3.0,
        "get_disp_corr_max_tip_arcsec": 20.0,
        "get_disp_corr_max_tilt_arcsec": 20.0,
        "init_differential_motion": False,
        "simulate_differential_motion": False,
        "model_wind_diff_motion": False,
        "model_gravity_diff_motion": False,
        "model_temp_diff_motion": False,
        "model_ao": False,
        "init_wind_piston_micron_std": piston_std,
        "init_wind_piston_clip_m": 4e-6,
        "init_wind_tip_arcsec_std_tt": tip_std,
        "init_wind_tilt_arcsec_std_tt": tilt_std,
        "reward_function": "factored",
        "reward_weight_strehl": 1.0,
        "reward_weight_shape": 0.0,
        "reward_weight_centered_strehl": 0.0,
        "action_penalty": False,
        "holding_bonus_weight": 0.0,
        "detector_power_watts": 1e-12,
        "detector_quantum_efficiency": 0.8,
        "detector_gain_e_per_dn": 0.5,
        "detector_max_dn": 65535,
        "render": False,
        "silence": True,
    }
    env = OptomechEnv(**kwargs)
    return env


def _resimulate(env, osys, integration_seconds):
    """Re-run simulation and detector model with current actuator state."""
    import torch
    frame_t = torch.zeros_like(osys._science_frame_t)
    for wl in env._cached_wavelengths:
        osys.simulate(wl)
        sci = osys.get_science_frame(integration_seconds=integration_seconds)
        frame_t += sci * (1.0 / len(env._cached_wavelengths))
    return env._apply_detector_model_gpu(frame_t, frame_t.device).cpu().numpy()


def extract(env, seed=42, force_perfect=False, integration_seconds=0.1,
            zero_tip_tilt=False, inject_tt_arcsec=0.0):
    """Reset env and extract pupil, OPD, PSF, focal-plane image.

    If force_perfect=True, zero out all actuators after reset and re-run
    the optical simulation to get a truly aberration-free state.
    If zero_tip_tilt=True, zero only the tip/tilt actuators (keep piston),
    then optionally inject a small known TT error via inject_tt_arcsec.
    """
    obs, info = env.reset(seed=seed)
    osys = env.optical_system

    resim = False
    if force_perfect:
        osys._actuators_t.zero_()
        resim = True
    elif zero_tip_tilt:
        # Zero tip and tilt actuators, keep piston
        # Layout: [p0, p1, ..., tip0, tip1, ..., tilt0, tilt1, ...]
        n_seg = osys._actuators_t.shape[0] // 3
        osys._actuators_t[n_seg:].zero_()
        # Inject small known TT if requested (in actuator units = arcsec)
        if inject_tt_arcsec > 0:
            import torch
            rng = np.random.RandomState(seed)
            tt_vals = rng.randn(n_seg * 2) * inject_tt_arcsec
            # Convert arcsec to radians for actuator space
            tt_rad = tt_vals * (np.pi / 180.0 / 3600.0)
            osys._actuators_t[n_seg:] = torch.tensor(
                tt_rad, dtype=osys._actuators_t.dtype,
                device=osys._actuators_t.device)
        resim = True

    if resim:
        focal_obs_resim = _resimulate(env, osys, integration_seconds)

    aperture = osys._aperture_t.cpu().numpy()
    ap_mask = np.abs(aperture) > 0.01

    actuators = osys._actuators_t.cpu().numpy()
    influence = osys._influence_t.cpu().numpy()
    opd = np.einsum("i,ihw->hw", actuators, influence)
    opd_masked = np.where(ap_mask, opd * 1e6, np.nan)  # in microns

    focal_field = osys._focal_field_t.cpu().numpy()
    psf = np.abs(focal_field) ** 2

    if resim:
        focal_obs = focal_obs_resim
    else:
        focal_obs = obs[-1] if obs.ndim == 3 else obs[0]

    strehl = info.get("strehl", 0.0) if isinstance(info, dict) else 0.0

    # Physical extents — pupil from grid, focal in λ/D
    pupil_diameter_mm = float(osys.pupil_grid.x.max() - osys.pupil_grid.x.min()) * 1e3
    pupil_diameter_m = pupil_diameter_mm * 1e-3
    _FOCAL_SIZE_M = {
        "nanoelf": 8.192e-5, "nanoelfplus": 8.192e-5,
        "elf": 3.611e-4, "circular": 8.192e-4,
    }
    _FOCAL_LENGTH_M = {
        "nanoelf": 1.018, "nanoelfplus": 1.018,
        "elf": 32.5, "circular": 200.0,
    }
    ap_key = osys._cfg["aperture_type"]
    focal_size_m = _FOCAL_SIZE_M.get(ap_key, 8.192e-5)
    focal_length_m = _FOCAL_LENGTH_M.get(ap_key, 1.018)
    wavelength_m = osys.wavelength
    # λ/D in meters at the focal plane = λ * f / D
    lam_over_d_m = wavelength_m * focal_length_m / pupil_diameter_m
    focal_size_lod = focal_size_m / lam_over_d_m  # extent in λ/D units

    env.close()
    return {
        "aperture": np.abs(aperture),
        "ap_mask": ap_mask,
        "opd_masked": opd_masked,
        "psf": psf,
        "focal_obs": focal_obs,
        "strehl": strehl,
        "pupil_diameter_mm": pupil_diameter_mm,
        "focal_size_lod": focal_size_lod,
    }


def save_img(data, path, cmap="gray", vmin=None, vmax=None,
             log_scale=False, norm=None, extent=None,
             xlabel=None, ylabel=None, cb_label=None):
    """Save a single image with physical axes, colorbar, and LaTeX labels."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    d = data.copy()
    if log_scale:
        d = np.log10(d + 1e-12)
    im = ax.imshow(d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                   norm=norm, aspect="equal", interpolation="nearest",
                   extent=extent)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if cb_label is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(cb_label)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def main():
    aperture_types = [
        ("nanoelf", "NanoELF (2 seg)"),
        ("nanoelfplus", "NanoELF+ (3 seg)"),
        ("elf", "ELF (15 seg)"),
    ]

    for ap_type, label in aperture_types:
        print(f"\n=== {label} ===")
        int_sec = 0.001 if ap_type == "elf" else 0.1

        # --- Unaberrated ---
        print("  Unaberrated...")
        env = make_env(ap_type, tip_std=0.0, tilt_std=0.0, piston_std=0.0)
        art = extract(env, seed=42, force_perfect=True, integration_seconds=int_sec)

        d_mm = art["pupil_diameter_mm"]
        f_lod = art["focal_size_lod"]
        pup_ext = [-d_mm / 2, d_mm / 2, -d_mm / 2, d_mm / 2]
        foc_ext = [-f_lod / 2, f_lod / 2, -f_lod / 2, f_lod / 2]
        pup_xl = r"$x$ (mm)"
        pup_yl = r"$y$ (mm)"
        foc_xl = r"$x$ ($\lambda/D$)"
        foc_yl = r"$y$ ($\lambda/D$)"

        save_img(art["aperture"],
                 os.path.join(OUT_DIR, f"{ap_type}_aperture.png"),
                 cmap="gray", extent=pup_ext,
                 xlabel=pup_xl, ylabel=pup_yl,
                 cb_label="Amplitude (norm.)")

        save_img(art["opd_masked"],
                 os.path.join(OUT_DIR, f"{ap_type}_opd_perfect.png"),
                 cmap="RdBu_r", vmin=-0.1, vmax=0.1, extent=pup_ext,
                 xlabel=pup_xl, ylabel=pup_yl,
                 cb_label=r"OPD ($\mu$m)")

        save_img(art["psf"],
                 os.path.join(OUT_DIR, f"{ap_type}_psf_perfect.png"),
                 cmap="inferno", log_scale=True, extent=foc_ext,
                 xlabel=foc_xl, ylabel=foc_yl,
                 cb_label=r"$\log_{10}$ intensity (arb.)")

        save_img(art["focal_obs"],
                 os.path.join(OUT_DIR, f"{ap_type}_focal_perfect.png"),
                 cmap="viridis", extent=foc_ext,
                 xlabel=foc_xl, ylabel=foc_yl,
                 cb_label="Detector (DN)")

        # --- Aberrated ---
        print("  Aberrated...")
        env_ab = make_env(ap_type, tip_std=0.0, tilt_std=0.0, piston_std=0.2)
        art_ab = extract(env_ab, seed=42, zero_tip_tilt=True,
                         inject_tt_arcsec=0.05, integration_seconds=int_sec)

        vm = np.nanmax(np.abs(art_ab["opd_masked"]))
        vm = max(vm, 0.01)
        save_img(art_ab["opd_masked"],
                 os.path.join(OUT_DIR, f"{ap_type}_opd_aberrated.png"),
                 cmap="RdBu_r", vmin=-vm, vmax=vm, extent=pup_ext,
                 xlabel=pup_xl, ylabel=pup_yl,
                 cb_label=r"OPD ($\mu$m)")

        save_img(art_ab["psf"],
                 os.path.join(OUT_DIR, f"{ap_type}_psf_aberrated.png"),
                 cmap="inferno", log_scale=True, extent=foc_ext,
                 xlabel=foc_xl, ylabel=foc_yl,
                 cb_label=r"$\log_{10}$ intensity (arb.)")

        save_img(art_ab["focal_obs"],
                 os.path.join(OUT_DIR, f"{ap_type}_focal_aberrated.png"),
                 cmap="viridis", extent=foc_ext,
                 xlabel=foc_xl, ylabel=foc_yl,
                 cb_label="Detector (DN)")

    print(f"\nAll images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
