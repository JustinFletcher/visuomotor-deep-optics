#!/usr/bin/env python
"""Inner-ring static-rollout pipeline + publication-quality figures.

Walks the latest dark_hole_runs/dark_hole_static_* sweep, picks the
six target_NN/ppo_optomech_*/checkpoints/best.pt for the inner ring
(ids 0..5), rolls each policy against its own target geometry once,
and renders three publication-grade composites:

  1. inner_ring_psf_grid       2x3 panel of the t = final raw-PSF
                                images, dark-hole circle overlaid.
                                The headline figure for the chapter.
  2. inner_ring_contrast_traces all six contrast(t) curves overlaid
                                on a single log-y axis, one colour per
                                target id.
  3. inner_ring_before_after    2x6 grid: top row initial PSF, bottom
                                row final PSF, side-by-side per
                                target. Shows the agent's wavefront
                                shaping before/after.

Same scholarly style as the stress / hero figures: serif font, muted
palette, sentence-case titles, frameless legends, 600 dpi PNG +
vector PDF.

Per-target GIFs are also written so they can be inspected
individually; the publication composites are the main deliverable.

Usage:
    python train/ppo/rollout_inner_ring_publication.py
    python train/ppo/rollout_inner_ring_publication.py --sweep-dir <path>
    python train/ppo/rollout_inner_ring_publication.py --no-gifs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.ppo.launch_static_dark_hole import build_grid       # noqa: E402
from train.ppo.rollout_static_dark_hole_grid import (          # noqa: E402
    _latest_sweep_dir,
    _resolve_checkpoint,
)
from train.ppo.rollout_dark_hole_grid import (                 # noqa: E402
    _build_env,
    _load_agent,
    render_gif,
    run_episode,
)


# Inner ring is the first six grid targets.
INNER_IDS = list(range(6))

# Scholarly palette + serif rcParams (mirrors the stress / hero
# figures so all paper graphics have a coherent look).
NEURIPS_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "figure.dpi": 110,
}
LINE_C = "#1f3b5e"          # deep slate-navy
BAND_C = "#6c8ebf"          # paler navy
ACCENT_C = "#d94a4a"        # current-step / accent red
PSF_CMAP = "inferno"
OPD_CMAP = "RdBu_r"
HOLE_EDGE = "cyan"


def _saveboth(fig, basepath):
    fig.savefig(f"{basepath}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")


def _circle_for(target, H, W, lw=1.2, alpha=0.9):
    angle, radius_frac, size_frac = target
    th = np.deg2rad(angle)
    cx_px = W / 2.0 + radius_frac * (W / 2.0) * np.cos(th)
    cy_px = H / 2.0 + radius_frac * (H / 2.0) * np.sin(th)
    r_px = size_frac * (W / 2.0)
    return Circle((cx_px, cy_px), r_px, fill=False,
                  edgecolor=HOLE_EDGE,
                  linestyle=(0, (1.5, 1.5)),
                  linewidth=lw, alpha=alpha)


# --------------------------------------------------------------------------
# Composite figure: 2x3 grid of final-state raw PSFs.
# --------------------------------------------------------------------------

def render_psf_grid(rollouts, out_path, frame="final"):
    """2x3 grid of the t=final (or t=initial) raw PSF for each target.

    rollouts: list of (target_id, target, ep_data) — exactly 6 entries.
    """
    plt.rcParams.update(NEURIPS_RC)
    assert len(rollouts) == 6, "PSF grid expects 6 inner-ring rollouts"

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))

    # Shared color scale across the six panels for visual comparability.
    psfs = []
    for _, _, ep in rollouts:
        psf = ep["raw_psf"][-1] if frame == "final" else ep["raw_psf"][0]
        psfs.append(np.asarray(psf))
    vmax = max(float(np.max(p)) for p in psfs)
    vmin = max(vmax * 1e-8, 1e-30)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for ax, (tid, target, ep), psf in zip(axes.flat, rollouts, psfs):
        H, W = psf.shape[-2:]
        im = ax.imshow(np.maximum(psf, vmin), cmap=PSF_CMAP,
                       norm=norm, origin="lower", aspect="equal")
        ax.add_patch(_circle_for(target, H, W))
        ax.set_xticks([]); ax.set_yticks([])
        ct = ep["contrast"][-1]
        ax.set_title(
            f"target {tid:02d}   "
            f"$\\theta={target[0]:.0f}^\\circ$   "
            f"$C={ct:.1e}$",
            fontsize=8.5, pad=3)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_edgecolor("#333333")

    # Single shared colorbar on the right.
    cax = fig.add_axes([0.93, 0.15, 0.018, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("raw PSF intensity (log)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Six trained dark-hole shaping policies, "
                 "final-step focal-plane intensity",
                 fontsize=10.5, y=0.97)
    fig.tight_layout(rect=[0.01, 0.02, 0.92, 0.93])
    _saveboth(fig, str(out_path))
    plt.close(fig)


# --------------------------------------------------------------------------
# Composite figure: contrast traces overlaid.
# --------------------------------------------------------------------------

def render_contrast_traces(rollouts, out_path):
    """Six contrast(t) curves overlaid, one per target id."""
    plt.rcParams.update(NEURIPS_RC)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(rollouts) - 1, 1))
              for i in range(len(rollouts))]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for (tid, target, ep), color in zip(rollouts, colors):
        contrast = np.array(ep["contrast"], dtype=np.float64)
        steps = np.arange(len(contrast))
        ax.plot(steps, np.where(contrast > 0, contrast, np.nan),
                color=color, lw=1.4,
                label=f"target {tid:02d}  ({target[0]:.0f}$^\\circ$)")

    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("contrast = min(hole) / max(PSF)")
    ax.set_title("Per-target contrast trajectories",
                 fontsize=10, pad=6)
    ax.grid(True, which="both", alpha=0.3, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.5)
    ax.legend(loc="upper right", ncol=2, handlelength=1.6,
              handletextpad=0.5, columnspacing=0.9, fontsize=7.5)

    fig.tight_layout(pad=0.4)
    _saveboth(fig, str(out_path))
    plt.close(fig)


# --------------------------------------------------------------------------
# Composite figure: 2x6 before / after PSFs.
# --------------------------------------------------------------------------

def render_before_after(rollouts, out_path):
    """2x6 grid: top row initial PSF, bottom row final PSF, per target.

    Color scale is per-row (so rows are comparable across targets, and
    rows can highlight different intensity bands)."""
    plt.rcParams.update(NEURIPS_RC)
    n = len(rollouts)
    fig, axes = plt.subplots(
        2, n, figsize=(1.5 * n + 0.6, 3.5),
        gridspec_kw={"hspace": 0.05, "wspace": 0.04,
                     "left": 0.05, "right": 0.96,
                     "top": 0.86, "bottom": 0.05})

    init_imgs = [np.asarray(ep["raw_psf"][0]) for _, _, ep in rollouts]
    final_imgs = [np.asarray(ep["raw_psf"][-1]) for _, _, ep in rollouts]

    init_max = max(float(np.max(p)) for p in init_imgs)
    final_max = max(float(np.max(p)) for p in final_imgs)
    init_norm = mcolors.LogNorm(
        vmin=max(init_max * 1e-8, 1e-30), vmax=init_max)
    final_norm = mcolors.LogNorm(
        vmin=max(final_max * 1e-8, 1e-30), vmax=final_max)

    for col, ((tid, target, ep), p_init, p_final) in enumerate(
            zip(rollouts, init_imgs, final_imgs)):
        H, W = p_init.shape[-2:]
        # Initial.
        ax_i = axes[0, col]
        ax_i.imshow(np.maximum(p_init, init_norm.vmin),
                    cmap=PSF_CMAP, norm=init_norm,
                    origin="lower", aspect="equal")
        ax_i.add_patch(_circle_for(target, H, W, lw=1.0, alpha=0.85))
        ax_i.set_xticks([]); ax_i.set_yticks([])
        ax_i.set_title(f"target {tid:02d}", fontsize=8, pad=2)
        for sp in ax_i.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor("#333333")
        # Final.
        ax_f = axes[1, col]
        ax_f.imshow(np.maximum(p_final, final_norm.vmin),
                    cmap=PSF_CMAP, norm=final_norm,
                    origin="lower", aspect="equal")
        ax_f.add_patch(_circle_for(target, H, W, lw=1.0, alpha=0.85))
        ax_f.set_xticks([]); ax_f.set_yticks([])
        ct = ep["contrast"][-1]
        ax_f.set_xlabel(f"$C={ct:.1e}$", fontsize=7, labelpad=2)
        for sp in ax_f.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor("#333333")

    # Row labels.
    fig.text(0.012, 0.68, "initial", rotation=90, va="center",
             ha="center", fontsize=8)
    fig.text(0.012, 0.27, "final", rotation=90, va="center",
             ha="center", fontsize=8)
    fig.suptitle("Initial vs. trained dark-hole intensity, "
                 "inner-ring targets",
                 fontsize=10, y=0.99)
    _saveboth(fig, str(out_path))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main pipeline.
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inner-ring publication-quality dark-hole rollout.")
    parser.add_argument("--sweep-dir", type=str, default=None,
                        help="Static-sweep run dir. Default: newest "
                             "dark_hole_runs/dark_hole_static_*.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Destination for figures + GIFs. Default: "
                             "figures/dark_hole_inner_ring_<sweep_basename>/.")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip per-target GIF rendering, "
                             "produce only the publication composites.")
    parser.add_argument("--prefer-latest", action="store_true",
                        help="Use newest update_*.pt instead of best.pt.")
    cli = parser.parse_args()

    sweep_dir = cli.sweep_dir
    if sweep_dir is None:
        sweep_dir = _latest_sweep_dir()
        if sweep_dir is None:
            print("Error: no static sweep found; pass --sweep-dir.")
            sys.exit(1)
        print(f"Using sweep: {sweep_dir}")

    out_dir = cli.output_dir
    if out_dir is None:
        sweep_base = os.path.basename(sweep_dir.rstrip("/"))
        out_dir = os.path.join(
            "figures", f"dark_hole_inner_ring_{sweep_base}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print()

    targets = build_grid()
    rollouts = []
    for tid in INNER_IDS:
        target = targets[tid]
        try:
            ckpt_path = _resolve_checkpoint(
                sweep_dir, tid, prefer_latest=cli.prefer_latest)
        except FileNotFoundError as e:
            print(f"  target {tid:>2}: SKIP — {e}")
            continue

        env = _build_env(target, cli.max_steps)
        agent, _ = _load_agent(ckpt_path, env, cli.device)
        ep = run_episode(agent, env, target, cli.seed, cli.device)
        env.close()
        ep["target_id"] = tid

        # Per-target GIF (skipped if --no-gifs).
        if not cli.no_gifs:
            gif_path = os.path.join(out_dir, f"target_{tid:02d}.gif")
            render_gif(ep, gif_path, dpi=110)

        final_ct = ep["contrast"][-1] if ep["contrast"] else float("nan")
        print(f"  target {tid:>2}: angle={target[0]:6.1f}  "
              f"r={target[1]:.3f}  size={target[2]:.3f}  "
              f"final_C={final_ct:.2e}  ckpt={os.path.basename(ckpt_path)}")
        rollouts.append((tid, target, ep))

    if len(rollouts) != 6:
        print(f"\nWarning: only {len(rollouts)}/6 inner targets had "
              "checkpoints; some publication figures expect 6 entries.")

    if len(rollouts) > 0:
        print("\n[publication composites]")
        if len(rollouts) == 6:
            render_psf_grid(
                rollouts, Path(out_dir) / "inner_ring_psf_grid")
            print("  wrote inner_ring_psf_grid.{png,pdf}")
        render_contrast_traces(
            rollouts, Path(out_dir) / "inner_ring_contrast_traces")
        print("  wrote inner_ring_contrast_traces.{png,pdf}")
        render_before_after(
            rollouts, Path(out_dir) / "inner_ring_before_after")
        print("  wrote inner_ring_before_after.{png,pdf}")

    print(f"\nAll outputs under {out_dir}")


if __name__ == "__main__":
    main()
