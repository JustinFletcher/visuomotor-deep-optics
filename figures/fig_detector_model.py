#!/usr/bin/env python3
"""
Generate detector model figure showing the conversion chain:
  optical power -> photons -> electrons -> digital numbers

Also shows aligned vs misaligned detector images side-by-side.

Output: figures/detector_model/

Usage:
    poetry run python figures/fig_detector_model.py
"""

import os
import sys
import numpy as np
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

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT_DIR = os.path.join(os.path.dirname(__file__), "detector_model")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_KWARGS = {
    "object_type": "single", "aperture_type": "nanoelf",
    "focal_plane_image_size_pixels": 512,
    "observation_mode": "image_only", "observation_window_size": 1,
    "wavelength": 1e-6, "oversampling_factor": 8,
    "bandwidth_nanometers": 200.0, "bandwidth_sampling": 2,
    "object_plane_extent_meters": 1.0, "object_plane_distance_meters": 1.0,
    "num_atmosphere_layers": 0, "max_episode_steps": 10,
    "command_secondaries": True, "command_tip_tilt": True,
    "command_tensioners": False, "command_dm": False,
    "ao_loop_active": False, "incremental_control": True,
    "action_type": "none", "actuator_noise": False, "env_action_scale": 0.1,
    "init_differential_motion": False, "simulate_differential_motion": False,
    "model_wind_diff_motion": False, "model_gravity_diff_motion": False,
    "model_temp_diff_motion": False, "model_ao": False,
    "init_wind_piston_micron_std": 0.0,
    "init_wind_tip_arcsec_std_tt": 0.0, "init_wind_tilt_arcsec_std_tt": 0.0,
    "reward_function": "factored", "reward_weight_strehl": 1.0,
    "reward_weight_shape": 0.0, "action_penalty": False,
    "holding_bonus_weight": 0.0, "render": False, "silence": True,
}


def main():
    import torch
    from optomech.optomech.optomech_v4 import OptomechEnv

    # --- 1D conversion chain ---
    print("Perfect alignment: detector chain...")
    env = OptomechEnv(**BASE_KWARGS)
    obs, info = env.reset(seed=42)
    osys = env.optical_system

    # Force perfect alignment: zero all actuators and re-propagate
    osys._actuators_t.zero_()
    wl = env.cfg["wavelength"]
    osys.simulate(wl)
    # Use an integration time that yields DN below saturation (65535)
    # but large enough for realistic integer counts.
    integration_sec = 0.05
    science_t = osys.get_science_frame(integration_seconds=integration_sec)

    psf = (torch.abs(osys._focal_field_t) ** 2).cpu().numpy()
    science = science_t.cpu().numpy()

    cy = psf.shape[0] // 2

    # Use the env's actual detector parameters
    h, c = 6.62607015e-34, 2.99792458e8
    wl = env.cfg["wavelength"]
    photon_e = h * c / wl
    det_qe = env.cfg["detector_quantum_efficiency"]
    det_gain = env.cfg["detector_gain_e_per_dn"]
    det_max_dn = env.cfg["detector_max_dn"]

    # 1D cross-section through the conversion chain.
    # get_science_frame already includes integration_seconds in its output
    # (PSF = |E|^2 * grid_weight * dt), so the result has energy units.
    energy = science[cy, :]
    n_ph = energy / photon_e
    n_el = n_ph * det_qe
    dn_1d = np.floor(np.clip(n_el / det_gain, 0, det_max_dn)).astype(int)

    # Compute focal plane physical extent for x-axis
    _FOCAL_SIZE_M = {"nanoelf": 8.192e-5, "nanoelfplus": 8.192e-5,
                     "elf": 8.192e-4, "circular": 8.192e-4}
    f_um = _FOCAL_SIZE_M.get(osys._cfg["aperture_type"], 8.192e-5) * 1e6
    x_um = np.linspace(-f_um / 2, f_um / 2, len(energy))

    print(f"  Integration: {integration_sec}s, peak DN: {np.max(dn_1d):.1f}, "
          f"peak photons: {np.max(n_ph):.1f}")

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.5]})
    for ax, y, lbl, col in zip(axes,
            [energy, n_ph, n_el, dn_1d],
            [r"Energy (J per pixel)", r"Photons (per pixel)",
             r"Electrons (per pixel)", r"Digital numbers (DN)"],
            ["#D32F2F", "#1976D2", "#388E3C", "#7B1FA2"]):
        ax.plot(x_um, y, color=col, linewidth=1.5)
        ax.set_ylabel(lbl, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 4))
    axes[0].set_title("Detector Model Conversion Chain", fontsize=13,
                      fontweight="bold")
    axes[-1].set_xlabel(r"Focal plane position ($\mu$m)", fontsize=10)

    # Add zoom inset on the DN panel showing a sidelobe with integer-scale counts.
    # Use a separate inset axes above the DN panel to avoid all overlap.
    ax_dn = axes[-1]
    zoom_x0, zoom_x1 = 12.1, 13.55
    zoom_mask = (x_um >= zoom_x0) & (x_um <= zoom_x1)
    zoom_y_max = np.max(dn_1d[zoom_mask]) * 1.25

    # Place inset in the upper-right of the taller DN panel, above the
    # main curve data which peaks in the center.
    ax_ins = ax_dn.inset_axes([0.60, 0.45, 0.32, 0.50])
    ax_ins.plot(x_um[zoom_mask], dn_1d[zoom_mask], color="#7B1FA2",
                linewidth=1.2, marker="o", markersize=4.5,
                markerfacecolor="#7B1FA2", markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)
    zoom_y_min = np.min(dn_1d[zoom_mask])
    zoom_y_range = np.max(dn_1d[zoom_mask]) - zoom_y_min
    pad = zoom_y_range * 0.2
    ax_ins.set_xlim(zoom_x0, zoom_x1)
    ax_ins.set_ylim(zoom_y_min - pad, zoom_y_min + zoom_y_range + pad)
    ax_ins.set_ylabel("DN", fontsize=9, labelpad=2)
    ax_ins.set_xlabel(r"Position ($\mu$m)", fontsize=9, labelpad=2)
    ax_ins.tick_params(labelsize=8, length=3, pad=2)
    ax_ins.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax_ins.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax_ins.grid(True, alpha=0.25, linewidth=0.5)
    ax_ins.set_facecolor("#fafafa")
    for spine in ax_ins.spines.values():
        spine.set_edgecolor("0.4")
        spine.set_linewidth(0.8)

    # Connector lines from inset bottom corners to the zoom region on the main curve.
    from matplotlib.patches import ConnectionPatch
    for x_anchor in [zoom_x0, zoom_x1]:
        # Find the DN value at this x position for the main curve endpoint
        idx = np.argmin(np.abs(x_um - x_anchor))
        y_main = dn_1d[idx]
        con = ConnectionPatch(
            xyA=(x_anchor, zoom_y_min - pad), coordsA=ax_ins.transData,
            xyB=(x_anchor, y_main), coordsB=ax_dn.transData,
            color="0.4", linewidth=0.8, linestyle="--", alpha=0.6)
        fig.add_artist(con)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"detector_chain_1d.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  detector_chain_1d")
    env.close()

    # --- Aligned vs misaligned ---
    print("Alignment comparison...")
    configs = [
        ("aligned", 0.0, 0.0, 0.0),
        ("mild", 0.5, 0.5, 1.0),
        ("severe", 2.0, 2.0, 3.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, (name, tip, tilt, piston) in enumerate(configs):
        kw = dict(BASE_KWARGS)
        kw.update(init_wind_tip_arcsec_std_tt=tip,
                  init_wind_tilt_arcsec_std_tt=tilt,
                  init_wind_piston_micron_std=piston)
        if tip > 0 or piston > 0:
            kw["init_differential_motion"] = True
            kw["model_wind_diff_motion"] = True

        env = OptomechEnv(**kw)
        obs, info = env.reset(seed=42)
        osys = env.optical_system
        psf_i = np.abs(osys._focal_field_t.cpu().numpy()) ** 2
        dn_i = obs[0] if obs.ndim == 3 else obs
        strehl = info.get("strehl", 0.0) if isinstance(info, dict) else 0.0
        _FOCAL_SIZE_M = {"nanoelf": 8.192e-5, "nanoelfplus": 8.192e-5,
                         "elf": 8.192e-4, "circular": 8.192e-4}
        f_um = _FOCAL_SIZE_M.get(osys._cfg["aperture_type"], 8.192e-5) * 1e6
        fext = [-f_um / 2, f_um / 2, -f_um / 2, f_um / 2]

        im_psf = axes[0, col].imshow(np.log10(psf_i + 1e-12), cmap="inferno",
                                     origin="lower", extent=fext)
        axes[0, col].set_title(f"{name.capitalize()}\n$S = {strehl:.3f}$",
                               fontsize=11, fontweight="bold")
        fig.colorbar(im_psf, ax=axes[0, col], shrink=0.82, pad=0.02,
                     label=r"$\log_{10}$ intensity")
        if col == 0:
            axes[0, col].set_ylabel(r"PSF --- $y$ ($\mu$m)", fontsize=10)
        else:
            axes[0, col].set_yticklabels([])

        im_dn = axes[1, col].imshow(dn_i, cmap="viridis", origin="lower",
                                    extent=fext)
        fig.colorbar(im_dn, ax=axes[1, col], shrink=0.82, pad=0.02,
                     label="DN")
        axes[1, col].set_xlabel(r"$x$ ($\mu$m)", fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel(r"Detector --- $y$ ($\mu$m)", fontsize=10)
        else:
            axes[1, col].set_yticklabels([])
        env.close()

    fig.suptitle("Effect of Segment Misalignment", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"alignment_comparison.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  alignment_comparison")
    print(f"\nAll images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
