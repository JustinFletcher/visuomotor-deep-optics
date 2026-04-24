#!/usr/bin/env python
"""Launch a deterministic 4x4 grid of dark-hole training runs.

Unlike ``launch_dark_hole.py`` (which draws hole geometry uniformly at
random), this launcher places 16 holes on a fixed polar grid that
covers the region between the PSF's first Airy ring and the outer
edge of the visible fringe pattern. Each hole is referenced by a
stable numeric target id 0..15, so individual failed jobs can be
re-submitted without re-running the rest of the sweep.

Grid layout (all values are focal-plane fractions of half-FOV):
  * Size radius: fixed at 1.15 x PSF core radius = 0.085
  * Radial centers (shells): linear 0.09 -> 0.38, 4 values
  * Angular positions: 4 per shell, rotated 45 degrees between shells
    so adjacent shells stagger their angular placement.

The grid is therefore:

    target     angle   radius_frac
       0          0       0.09
       1         90       0.09
       2        180       0.09
       3        270       0.09
       4         45       0.19
       5        135       0.19
       6        225       0.19
       7        315       0.19
       8          0       0.28
       9         90       0.28
      10        180       0.28
      11        270       0.28
      12         45       0.38
      13        135       0.38
      14        225       0.38
      15        315       0.38

Usage:
    # Launch the whole 16-target grid on HPC
    python train/ppo/launch_dark_hole_grid.py

    # Re-launch a single target (e.g. after a failed SLURM job)
    python train/ppo/launch_dark_hole_grid.py --target-id 5

    # Preview sbatch scripts without submitting
    python train/ppo/launch_dark_hole_grid.py --dry-run

    # Generate a paper-quality coverage figure and exit without
    # submitting anything. Saves the figure next to your current
    # working directory.
    python train/ppo/launch_dark_hole_grid.py --illustrate
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

# --------------------------------------------------------------------------
# Grid definition
# --------------------------------------------------------------------------

PSF_CORE_RADIUS_FRAC = 0.074          # first Airy zero in half-FOV units
SIZE_RADIUS_FRAC = 1.15 * PSF_CORE_RADIUS_FRAC   # 0.0851

SHELL_RADII = [0.09, 0.19, 0.28, 0.38]
SHELL_ANGLES_EVEN = [0.0, 90.0, 180.0, 270.0]
SHELL_ANGLES_ODD = [45.0, 135.0, 225.0, 315.0]


def build_grid():
    """Return 16 (angle_deg, radius_frac, size_radius) tuples, id-ordered."""
    targets = []
    for shell_idx, r in enumerate(SHELL_RADII):
        angles = SHELL_ANGLES_ODD if shell_idx % 2 else SHELL_ANGLES_EVEN
        for a in angles:
            targets.append((float(a), float(r), float(SIZE_RADIUS_FRAC)))
    assert len(targets) == 16
    return targets


# --------------------------------------------------------------------------
# SLURM scaffolding (matches launch_dark_hole.py)
# --------------------------------------------------------------------------

MAX_SEED = 2**31 - 1
SLURM_ACCOUNT = "MHPCC38870258"
SLURM_PARTITION = "standard"
SLURM_TIME = "72:00:00"
SLURM_GRES = "gpu"
HPC_WORKDIR = "/p/home/fletch/visuomotor-deep-optics"


def make_sbatch_script(target_idx, run_id, run_dir_base, seed,
                       angle_deg, radius_frac, size_radius,
                       wall_time=SLURM_TIME):
    run_dir = os.path.join(run_dir_base, f"target_{target_idx:02d}")
    job_name = f"dhg-{run_id[-8:]}-{target_idx:02d}"
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
        poetry run python train/ppo/train_ppo_elf_dark_hole.py \\
            --hpc \\
            --dark-hole-angle {angle_deg:.4f} \\
            --dark-hole-radius-frac {radius_frac:.4f} \\
            --dark-hole-size {size_radius:.4f} \\
            --seed {seed} \\
            --run-dir {run_dir}
    """)


# --------------------------------------------------------------------------
# --illustrate: render the coverage figure
# --------------------------------------------------------------------------

def render_illustration(targets, out_path: Path):
    """Render a paper-quality figure of the 16 holes overlaid on the PSF."""
    # Deferred imports so --dry-run / single-job re-launch don't pay
    # the cost of building an env just to validate.
    import contextlib, io
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import gymnasium as gym

    # Style: consistent with the stress / hero figures.
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": 200, "figure.dpi": 110,
    })

    from train.ppo.train_ppo_optomech import register_optomech
    from train.ppo.train_ppo_elf_dark_hole import ELF_DARK_HOLE_ENV_KWARGS

    kw = dict(ELF_DARK_HOLE_ENV_KWARGS)
    kw["dark_hole"] = False
    kw["init_piston_micron_mean"] = 0.0
    kw["init_piston_micron_std"] = 0.0
    kw["silence"] = True
    kw["max_episode_steps"] = 2

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

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    psf_show = np.maximum(psf, max(float(psf.max()) * 1e-5, 1e-20))
    norm = mcolors.LogNorm(vmin=psf_show.min(), vmax=psf_show.max())
    im = ax.imshow(psf_show, cmap="inferno", norm=norm, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_linewidth(0.6)

    # Overlay each hole (convert fractional coords to pixel coords).
    for i, (angle, r, size) in enumerate(targets):
        theta = np.deg2rad(angle)
        cx = W / 2.0 + r * (W / 2.0) * np.cos(theta)
        cy = H / 2.0 + r * (H / 2.0) * np.sin(theta)
        size_px = size * (W / 2.0)

        ax.add_patch(Circle(
            (cx, cy), size_px, fill=False,
            edgecolor="cyan", linestyle=(0, (1.2, 1.2)),
            linewidth=1.0, alpha=0.95))
        # Job id on top of each hole, white with a thin dark outline for
        # legibility against the log-scale inferno background.
        txt = ax.text(cx, cy, str(i), color="white", fontsize=8,
                      fontweight="bold", ha="center", va="center")
        try:
            import matplotlib.patheffects as pe
            txt.set_path_effects([
                pe.Stroke(linewidth=1.6, foreground="black"),
                pe.Normal()])
        except Exception:
            pass

    # Colorbar + title.
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("reference PSF (log DN)", fontsize=8)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(
        "Dark-hole grid coverage (16 targets, id 0-15)\n"
        "polar shells at r $\\in\\{0.09, 0.19, 0.28, 0.38\\}$, "
        r"size $=0.085$, 45$^\circ$ stagger",
        fontsize=9)
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[illustrate] wrote {out_path}.png / .pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch a 16-target grid of dark-hole training runs.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run ID (default: dark_hole_grid_{ts})")
    parser.add_argument("--target-id", type=int, default=None,
                        help="Re-launch a single target (0..15).")
    parser.add_argument("--ppo-seed", type=int, default=None,
                        help="If set, every launched job uses this same "
                             "PPO seed; otherwise each target gets a "
                             "distinct random seed.")
    parser.add_argument("--local", action="store_true",
                        help="Run jobs sequentially in-process instead "
                             "of submitting sbatch jobs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch scripts without executing.")
    parser.add_argument("--time", type=str, default=SLURM_TIME,
                        help=f"SLURM wall time (default: {SLURM_TIME})")
    parser.add_argument("--illustrate", action="store_true",
                        help="Render a paper-quality figure of the 16 "
                             "hole positions on the reference PSF (with "
                             "job ids overlaid) and exit without "
                             "submitting anything.")
    parser.add_argument("--illustrate-out", type=str,
                        default="dark_hole_runs/dark_hole_grid_coverage",
                        help="Destination basename (no extension) for the "
                             "illustration figure.")
    cli = parser.parse_args()

    targets = build_grid()

    if cli.illustrate:
        render_illustration(targets, Path(cli.illustrate_out))
        return

    run_id = cli.run_id or f"dark_hole_grid_{int(time.time())}"
    run_dir_base = os.path.join("dark_hole_runs", run_id)

    if cli.ppo_seed is not None:
        seeds = [int(cli.ppo_seed)] * len(targets)
        seed_mode = f"explicit ({cli.ppo_seed})"
    else:
        seeds = [secrets.randbelow(MAX_SEED) for _ in targets]
        seed_mode = "random per-target"

    if cli.target_id is not None:
        if not (0 <= cli.target_id < len(targets)):
            print(f"Error: --target-id must be in [0, {len(targets) - 1}]")
            sys.exit(1)
        target_indices = [cli.target_id]
    else:
        target_indices = list(range(len(targets)))

    print(f"Run ID:      {run_id}")
    print(f"Output dir:  {run_dir_base}")
    print(f"Launching:   {len(target_indices)} of 16 targets"
          f"{' (single re-launch)' if cli.target_id is not None else ''}")
    print(f"PPO seeds:   {seed_mode}")
    print(f"Mode:        {'local' if cli.local else 'sbatch'}"
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
                sys.executable,
                "train/ppo/train_ppo_elf_dark_hole.py",
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
        print(f"Monitor: squeue -u $USER | grep dhg-{run_id[-8:]}")


if __name__ == "__main__":
    main()
