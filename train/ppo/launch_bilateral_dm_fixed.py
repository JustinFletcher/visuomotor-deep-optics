#!/usr/bin/env python
"""Launch the 6 inner-ring fixed-axis bilateral-DM training runs.

Companion to ``launch_bilateral_dm.py``: same six inner-ring targets,
same env, same reward, same DM, same hyperparameters -- but uses the
``fixed_vertical`` symmetry mode (axis is x=0 for every target,
strict per-actuator bilateral symmetry, blind region is the horizontal
focal-plane mirror of the target).

Usage:
    python train/ppo/launch_bilateral_dm_fixed.py
    python train/ppo/launch_bilateral_dm_fixed.py --target-id 2
    python train/ppo/launch_bilateral_dm_fixed.py --illustrate
    python train/ppo/launch_bilateral_dm_fixed.py --local
"""
import argparse
import os
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

from train.ppo.launch_static_dark_hole import (
    HPC_WORKDIR, MAX_SEED, SLURM_ACCOUNT, SLURM_GRES,
    SLURM_PARTITION, SLURM_TIME,
    build_grid, _ids_for_ring,
)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_dark_hole_bilateral_dm_fixed.py"
_RUN_PREFIX = "dark_hole_bilateral_dm_fixed"


def make_sbatch_script(target_idx, run_id, run_dir_base, seed,
                       angle_deg, radius_frac, size_radius,
                       wall_time=SLURM_TIME):
    run_dir = os.path.join(run_dir_base, f"target_{target_idx:02d}")
    job_name = f"dhf-{run_id[-8:]}-{target_idx:02d}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={wall_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --gres={SLURM_GRES}
        #SBATCH --output=slurm-{run_id}-target{target_idx:02d}-%j.out
        #SBATCH --error=slurm-{run_id}-target{target_idx:02d}-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --dark-hole-angle {angle_deg:.4f} \\
            --dark-hole-radius-frac {radius_frac:.4f} \\
            --dark-hole-size {size_radius:.4f} \\
            --seed {seed} \\
            --run-dir {run_dir}
    """)


def render_illustration(targets, out_path: Path, only_ids):
    """Inner-ring targets + their fixed-vertical blind regions overlaid
    on the reference PSF. Differs from launch_bilateral_dm: blind
    region centres at (W - cx, cy) not (W - cx, H - cy).
    """
    import contextlib, io
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import gymnasium as gym

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": 200, "figure.dpi": 110,
    })

    from train.ppo.train_ppo_optomech import register_optomech
    from train.ppo.train_ppo_elf_dark_hole_bilateral_dm import ENV_KWARGS

    kw = dict(ENV_KWARGS)
    kw["dark_hole"] = False
    kw["init_piston_micron_mean"] = 0.0
    kw["init_piston_micron_std"] = 0.0
    kw["silence"] = True
    kw["max_episode_steps"] = 2
    kw["command_dm"] = False

    print("[illustrate] building one-off env for reference PSF...")
    register_optomech("optomech-v4", max_episode_steps=2)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make("optomech-v4", **kw)
        env.reset(seed=0)
    base = env.unwrapped
    psf = np.asarray(base.optical_system.perfect_image)
    H = W = int(np.sqrt(psf.size)) if psf.ndim == 1 else psf.shape[-1]
    if psf.ndim == 1:
        psf = psf.reshape(H, W)
    env.close()

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    psf_show = np.maximum(psf, max(float(psf.max()) * 1e-5, 1e-20))
    norm = mcolors.LogNorm(vmin=psf_show.min(), vmax=psf_show.max())
    im = ax.imshow(psf_show, cmap="inferno", norm=norm, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])

    keep = set(only_ids)
    for i, (angle, r, size) in enumerate(targets):
        if i not in keep:
            continue
        theta = np.deg2rad(angle)
        cx_t = W / 2.0 + r * (W / 2.0) * np.cos(theta)
        cy_t = H / 2.0 + r * (H / 2.0) * np.sin(theta)
        # Vertical-axis bilateral mirror: flip x, keep y.
        cx_b = W / 2.0 - r * (W / 2.0) * np.cos(theta)
        cy_b = cy_t
        size_px = size * (W / 2.0)

        ax.add_patch(Circle(
            (cx_t, cy_t), size_px, fill=False,
            edgecolor="cyan", linestyle=(0, (1.2, 1.2)),
            linewidth=1.1, alpha=0.95,
            label="target" if i == only_ids[0] else None))
        ax.add_patch(Circle(
            (cx_b, cy_b), size_px, fill=False,
            edgecolor="magenta", linestyle=(0, (3, 2)),
            linewidth=1.1, alpha=0.85,
            label="blind" if i == only_ids[0] else None))
        ax.text(cx_t, cy_t, str(i), color="white", fontsize=8,
                fontweight="bold", ha="center", va="center")

    # Symmetry axis line at x = W/2
    ax.axvline(W / 2.0, color="white", lw=0.5, alpha=0.4,
               linestyle=(0, (4, 3)))

    ax.legend(loc="lower right", fontsize=7, frameon=True,
              fancybox=True, framealpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("reference PSF (log DN)", fontsize=8)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(
        "Bilateral-DM dark-hole, fixed vertical axis (inner ring)\n"
        "cyan: target  magenta: blind region (horizontal mirror)",
        fontsize=9)
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[illustrate] wrote {out_path}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Launch the 6 inner-ring fixed-axis bilateral-DM runs.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--target-id", type=int, default=None,
                        help="Re-launch a single inner-ring target (0..5).")
    parser.add_argument("--ppo-seed", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time", type=str, default=SLURM_TIME)
    parser.add_argument("--illustrate", action="store_true")
    parser.add_argument("--illustrate-out", type=str,
                        default="dark_hole_runs/dark_hole_bilateral_dm_fixed_coverage")
    cli = parser.parse_args()

    targets = build_grid()
    inner_ids = _ids_for_ring(0)

    if cli.illustrate:
        render_illustration(
            targets, Path(cli.illustrate_out), only_ids=inner_ids)
        return

    run_id = cli.run_id or f"{_RUN_PREFIX}_{int(time.time())}"
    run_dir_base = os.path.join("dark_hole_runs", run_id)

    if cli.target_id is not None:
        if cli.target_id not in inner_ids:
            print(f"Error: --target-id must be one of {inner_ids} (inner ring)")
            sys.exit(1)
        target_indices = [cli.target_id]
    else:
        target_indices = list(inner_ids)

    if cli.ppo_seed is not None:
        seeds = {i: int(cli.ppo_seed) for i in inner_ids}
        seed_mode = f"explicit ({cli.ppo_seed})"
    else:
        seeds = {i: secrets.randbelow(MAX_SEED) for i in inner_ids}
        seed_mode = "random per-target"

    print(f"Run ID:      {run_id}")
    print(f"Output dir:  {run_dir_base}")
    print(f"Mode:        fixed_vertical (strict per-actuator bilateral symmetry)")
    print(f"Launching:   {len(target_indices)} of 6 inner-ring targets")
    print(f"PPO seeds:   {seed_mode}")
    print(f"Dispatch:    {'local' if cli.local else 'sbatch'}"
          f"{' (dry-run)' if cli.dry_run else ''}")
    print()
    print(f"{'id':>3}  {'angle':>7}  {'radius':>7}  {'size':>6}  {'seed':>12}")
    for i in target_indices:
        a, r, s = targets[i]
        print(f"{i:3d}  {a:7.2f}  {r:7.3f}  {s:6.3f}  {seeds[i]:12d}")
    print()

    if cli.local:
        for i in target_indices:
            a, r, s = targets[i]
            run_dir = os.path.join(run_dir_base, f"target_{i:02d}")
            cmd = [
                sys.executable, _TRAIN_SCRIPT,
                "--dark-hole-angle", f"{a:.4f}",
                "--dark-hole-radius-frac", f"{r:.4f}",
                "--dark-hole-size", f"{s:.4f}",
                "--seed", str(seeds[i]),
                "--run-dir", run_dir,
            ]
            print(f"Target {i:2d}: {' '.join(cmd)}")
            if not cli.dry_run:
                os.makedirs(run_dir, exist_ok=True)
                subprocess.run(cmd, check=True)
        return

    job_ids = []
    for i in target_indices:
        a, r, s = targets[i]
        script = make_sbatch_script(
            target_idx=i, run_id=run_id, run_dir_base=run_dir_base,
            seed=seeds[i], angle_deg=a, radius_frac=r,
            size_radius=s, wall_time=cli.time)
        if cli.dry_run:
            print(f"Target {i:2d}: would submit sbatch job")
            print(textwrap.indent(script, "    "))
            continue
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Target {i:2d}: FAILED - {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip()
        job_ids.append((i, job_id))
        print(f"Target {i:2d}: submitted job {job_id}")

    if job_ids and not cli.dry_run:
        print(f"\n{len(job_ids)} jobs submitted for run '{run_id}'")
        print(f"Monitor: squeue -u $USER | grep dhf-{run_id[-8:]}")


if __name__ == "__main__":
    main()
