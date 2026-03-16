"""
Tip/tilt error sweep: evaluate agent performance across a range of
initial tip/tilt disturbance magnitudes.

For each disturbance level, runs N rollouts and records the mean final
Strehl. Produces a scatter plot of mean Strehl vs tip/tilt error.

Usage:
    poetry run python train/ppo/sweep_tiptilt.py \
        --checkpoint runs/<run>/checkpoints/best.pt \
        --env-version v3

    poetry run python train/ppo/sweep_tiptilt.py \
        --checkpoint runs/<run>/checkpoints/best.pt \
        --tt-min 0.0 --tt-max 3.0 --tt-steps 13 \
        --num-episodes 16
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.rollout import run_rollouts, _DEFAULT_OUTPUT_DIR
from train.ppo.train_ppo_nanoelf_tt import NANOELF_TT_ENV_KWARGS


def run_sweep(args):
    tt_values = np.linspace(args.tt_min, args.tt_max, args.tt_steps)

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"sweep_tiptilt_{timestamp}_{int(time.time()) % 10000}"
    test_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output directory: {test_dir}")
    print(f"Sweep: tip/tilt {args.tt_min:.2f} -> {args.tt_max:.2f} "
          f"({args.tt_steps} steps, {args.num_episodes} episodes each)")

    # Use fixed seeds across all sweep points for comparability
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31, size=args.num_episodes).tolist()

    results = []
    for i, tt in enumerate(tt_values):
        print(f"\n--- Tip/tilt = {tt:.3f} arcsec ({i+1}/{len(tt_values)}) ---")

        env_kwargs = dict(NANOELF_TT_ENV_KWARGS)
        env_kwargs["init_wind_tip_arcsec_std_tt"] = float(tt)
        env_kwargs["init_wind_tilt_arcsec_std_tt"] = float(tt)

        episodes, metrics = run_rollouts(
            checkpoint_path=args.checkpoint,
            env_kwargs=env_kwargs,
            env_version=args.env_version,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            seeds=seeds,
        )

        final_strehls = [e["strehls"][-1] for e in episodes]
        entry = {
            "tip_tilt_arcsec": float(tt),
            "mean_final_strehl": float(np.mean(final_strehls)),
            "std_final_strehl": float(np.std(final_strehls)),
            "median_final_strehl": float(np.median(final_strehls)),
            "min_final_strehl": float(np.min(final_strehls)),
            "max_final_strehl": float(np.max(final_strehls)),
            "mean_return": metrics["mean_return"],
            "all_final_strehls": [float(s) for s in final_strehls],
        }
        results.append(entry)
        print(f"  Mean Strehl: {entry['mean_final_strehl']:.4f} "
              f"+/- {entry['std_final_strehl']:.4f}")

    # Save raw results
    results_path = os.path.join(test_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}")

    # Generate figures
    _fig_strehl_vs_tiptilt(results, test_dir)
    _fig_return_vs_tiptilt(results, test_dir)

    print(f"All outputs in: {test_dir}")


def _savefig(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _fig_strehl_vs_tiptilt(results, output_dir):
    """Scatter plot: mean final Strehl vs initial tip/tilt error."""
    tt = [r["tip_tilt_arcsec"] for r in results]
    mean_s = [r["mean_final_strehl"] for r in results]
    std_s = [r["std_final_strehl"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual points as scatter
    for r in results:
        tt_val = r["tip_tilt_arcsec"]
        for s in r["all_final_strehls"]:
            ax.scatter(tt_val, s, color="steelblue", alpha=0.2, s=15,
                       edgecolors="none", zorder=2)

    # Mean +/- std as errorbars
    ax.errorbar(tt, mean_s, yerr=std_s, fmt="o-", color="darkblue",
                linewidth=1.5, markersize=6, capsize=4, capthick=1.5,
                label="Mean +/- std", zorder=3)

    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Final Strehl Ratio")
    ax.set_title("Agent Performance vs Initial Tip/Tilt Disturbance")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "strehl_vs_tiptilt")
    print(f"  Figure: strehl_vs_tiptilt.png/pdf")


def _fig_return_vs_tiptilt(results, output_dir):
    """Mean return vs tip/tilt error."""
    tt = [r["tip_tilt_arcsec"] for r in results]
    returns = [r["mean_return"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tt, returns, "o-", color="forestgreen", linewidth=1.5, markersize=6)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Mean Return vs Initial Tip/Tilt Disturbance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "return_vs_tiptilt")
    print(f"  Figure: return_vs_tiptilt.png/pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep tip/tilt error and plot agent performance")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--env-version", type=str, default="v3",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--num-episodes", type=int, default=8,
                        help="Episodes per sweep point")
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--tt-min", type=float, default=0.0,
                        help="Min tip/tilt error in arcsec")
    parser.add_argument("--tt-max", type=float, default=3.0,
                        help="Max tip/tilt error in arcsec")
    parser.add_argument("--tt-steps", type=int, default=13,
                        help="Number of sweep points")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
