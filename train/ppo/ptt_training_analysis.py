"""
Training analysis for a PTT run: extract tensorboard metrics and evaluate
checkpoints to produce training-progress figures.

Usage:
    poetry run python train/ppo/ptt_training_analysis.py \
        runs/ppo_optomech_1_1773905144

    # Skip rollout eval (just plot TB metrics + previously cached evals):
    poetry run python train/ppo/ptt_training_analysis.py \
        runs/ppo_optomech_1_1773905144 --skip-eval

    # Subsample checkpoints (default: every 10th = ~12 evals):
    poetry run python train/ppo/ptt_training_analysis.py \
        runs/ppo_optomech_1_1773905144 --eval-every 5

    # More episodes per checkpoint (default: 4):
    poetry run python train/ppo/ptt_training_analysis.py \
        runs/ppo_optomech_1_1773905144 --eval-episodes 8
"""

import os
import sys
import json
import glob
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import Normalize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _savefig(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


# ══════════════════════════════════════════════════════════════════════
# Tensorboard extraction
# ══════════════════════════════════════════════════════════════════════

def extract_tb_scalars(tb_dir):
    """Extract all scalar time-series from a tensorboard log directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalars[tag] = {"steps": steps, "values": values}
    return scalars


# ══════════════════════════════════════════════════════════════════════
# Lightweight checkpoint evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_checkpoints(run_dir, eval_every=10, num_episodes=4,
                         max_episode_steps=128, env_version="v4"):
    """Evaluate a subsampled set of checkpoints.

    Returns a list of dicts, each with:
        update, global_step, mean_return, mean_final_strehl,
        mean_median_strehl, per_episode_strehls (list of lists)
    """
    import torch
    import gymnasium as gym
    from train.ppo.rollout import load_agent, run_single_episode
    from train.ppo.train_ppo_optomech import register_optomech

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "update_*.pt")),
                         key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))

    # Subsample
    selected = ckpt_files[::eval_every]
    # Always include first and last
    if ckpt_files[0] not in selected:
        selected.insert(0, ckpt_files[0])
    if ckpt_files[-1] not in selected:
        selected.append(ckpt_files[-1])

    print(f"Evaluating {len(selected)}/{len(ckpt_files)} checkpoints "
          f"({num_episodes} episodes × {max_episode_steps} steps each)")

    # Load config from first checkpoint to get env_kwargs
    first_ckpt = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    config = first_ckpt["config"]
    env_kwargs = config.env_kwargs if hasattr(config, "env_kwargs") else config["env_kwargs"]
    env_kwargs = dict(env_kwargs)
    env_kwargs["max_episode_steps"] = max_episode_steps

    register_optomech(f"optomech-{env_version}", max_episode_steps=max_episode_steps)
    env = gym.make(f"optomech-{env_version}", **env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Fixed seeds for comparability across checkpoints
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31, size=num_episodes).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_results = []
    for ci, ckpt_path in enumerate(selected):
        update_num = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        global_step = ckpt.get("global_step", update_num * 8192)

        agent, _, obs_ref_max = load_agent(ckpt_path, env, device)

        returns = []
        final_strehls = []
        median_strehls = []
        all_strehls = []

        for seed in seeds:
            ep = run_single_episode(agent, env, seed, obs_ref_max, device)
            returns.append(ep["return"])
            final_strehls.append(ep["strehls"][-1])
            median_strehls.append(float(np.median(ep["strehls"])))
            all_strehls.append(ep["strehls"])

        entry = {
            "update": update_num,
            "global_step": global_step,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_final_strehl": float(np.mean(final_strehls)),
            "std_final_strehl": float(np.std(final_strehls)),
            "mean_median_strehl": float(np.mean(median_strehls)),
            "all_returns": [float(r) for r in returns],
            "all_final_strehls": [float(s) for s in final_strehls],
            "all_median_strehls": [float(s) for s in median_strehls],
        }
        eval_results.append(entry)

        print(f"  [{ci+1}/{len(selected)}] update={update_num:>6d}  "
              f"step={global_step:>10d}  "
              f"return={entry['mean_return']:+.3f}  "
              f"final_S={entry['mean_final_strehl']:.3f}  "
              f"median_S={entry['mean_median_strehl']:.3f}")

    env.close()
    return eval_results


# ══════════════════════════════════════════════════════════════════════
# Figures: Tensorboard metrics
# ══════════════════════════════════════════════════════════════════════

def fig_tb_training_curves(scalars, output_dir):
    """Core training curves: reward, losses, entropy, KL."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    plots = [
        ("train/step_reward", "Step Reward", "forestgreen"),
        ("train/episodic_return", "Episodic Return", "darkblue"),
        ("train/step_strehl", "Step Strehl", "darkorange"),
        ("losses/policy_loss", "Policy Loss", "crimson"),
        ("losses/value_loss", "Value Loss", "purple"),
        ("losses/entropy", "Entropy", "teal"),
    ]

    for ax, (tag, title, color) in zip(axes.flat, plots):
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], color=color, linewidth=0.5, alpha=0.6)
            # Rolling mean
            if len(s["values"]) > 50:
                window = max(len(s["values"]) // 50, 10)
                smooth = np.convolve(s["values"], np.ones(window)/window, mode="valid")
                smooth_steps = s["steps"][window-1:]
                ax.plot(smooth_steps, smooth, color=color, linewidth=2.0)
            ax.set_title(title)
            ax.set_xlabel("Global Step")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{tag}\n(not found)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(title)

    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "tb_training_curves")


def fig_tb_performance(scalars, output_dir):
    """Performance metrics: SPS, rollout/optimize time split."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    plots = [
        ("performance/current_SPS", "Steps Per Second", "steelblue"),
        ("performance/rollout_pct", "Rollout % of Update", "darkorange"),
        ("performance/optimize_pct", "Optimize % of Update", "purple"),
    ]

    for ax, (tag, title, color) in zip(axes.flat, plots):
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], color=color, linewidth=0.5, alpha=0.5)
            if len(s["values"]) > 20:
                window = max(len(s["values"]) // 30, 5)
                smooth = np.convolve(s["values"], np.ones(window)/window, mode="valid")
                ax.plot(s["steps"][window-1:], smooth, color=color, linewidth=2.0)
            ax.set_title(title)
            ax.set_xlabel("Global Step")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(title)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    fig.suptitle("Performance Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir, "tb_performance")


def fig_tb_curriculum(scalars, output_dir):
    """Curriculum progression: tip/tilt std over training."""
    tag = "curriculum/tip_tilt_std"
    if tag not in scalars:
        print("  (no curriculum data found, skipping)")
        return

    s = scalars[tag]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s["steps"], s["values"], color="darkorange", linewidth=2)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Tip/Tilt Std (arcsec)")
    ax.set_title("Curriculum: Tip/Tilt Error Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "tb_curriculum")


def fig_tb_reward_vs_curriculum(scalars, output_dir):
    """Overlay episodic return with curriculum schedule on twin axes."""
    ret_tag = "train/episodic_return"
    cur_tag = "curriculum/tip_tilt_std"
    if ret_tag not in scalars or cur_tag not in scalars:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Episodic return
    s_ret = scalars[ret_tag]
    ax1.plot(s_ret["steps"], s_ret["values"], color="steelblue", linewidth=0.3, alpha=0.3)
    if len(s_ret["values"]) > 50:
        w = max(len(s_ret["values"]) // 50, 10)
        smooth = np.convolve(s_ret["values"], np.ones(w)/w, mode="valid")
        ax1.plot(s_ret["steps"][w-1:], smooth, color="darkblue", linewidth=2, label="Episodic Return")
    ax1.set_ylabel("Episodic Return", color="darkblue")
    ax1.tick_params(axis="y", labelcolor="darkblue")

    # Curriculum
    s_cur = scalars[cur_tag]
    ax2.plot(s_cur["steps"], s_cur["values"], color="darkorange", linewidth=2,
             linestyle="--", label="TT Curriculum")
    ax2.set_ylabel("Tip/Tilt Std (arcsec)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_xlabel("Global Step")
    ax1.set_title("Reward Progression vs Curriculum Schedule")
    ax1.grid(True, alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    fig.tight_layout()
    _savefig(fig, output_dir, "tb_reward_vs_curriculum")


def fig_tb_kl_entropy(scalars, output_dir):
    """KL divergence and entropy over training — stability diagnostics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, tag, title, color in [
        (ax1, "losses/approx_kl", "Approx KL Divergence", "crimson"),
        (ax2, "losses/entropy", "Policy Entropy", "teal"),
    ]:
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], color=color, linewidth=0.4, alpha=0.4)
            if len(s["values"]) > 50:
                w = max(len(s["values"]) // 50, 10)
                smooth = np.convolve(s["values"], np.ones(w)/w, mode="valid")
                ax.plot(s["steps"][w-1:], smooth, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Stability Diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir, "tb_kl_entropy")


# ══════════════════════════════════════════════════════════════════════
# Figures: Checkpoint evaluations
# ══════════════════════════════════════════════════════════════════════

def fig_eval_learning_curve(eval_results, output_dir):
    """Mean return and Strehl vs training step from checkpoint evals."""
    steps = [e["global_step"] for e in eval_results]
    returns = [e["mean_return"] for e in eval_results]
    final_s = [e["mean_final_strehl"] for e in eval_results]
    median_s = [e["mean_median_strehl"] for e in eval_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Return
    ax1.plot(steps, returns, "o-", color="darkblue", linewidth=1.5, markersize=4)
    # Scatter individual episodes
    for e in eval_results:
        for r in e["all_returns"]:
            ax1.scatter(e["global_step"], r, color="steelblue", alpha=0.2, s=10,
                        edgecolors="none")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Evaluation Return Over Training")
    ax1.grid(True, alpha=0.3)

    # Strehl
    ax2.plot(steps, final_s, "o-", color="darkorange", linewidth=1.5, markersize=4,
             label="Final Strehl")
    ax2.plot(steps, median_s, "s--", color="forestgreen", linewidth=1.5, markersize=4,
             label="Median Strehl")
    for e in eval_results:
        for s in e["all_final_strehls"]:
            ax2.scatter(e["global_step"], s, color="darkorange", alpha=0.2, s=10,
                        edgecolors="none")
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Strehl Ratio")
    ax2.set_title("Evaluation Strehl Over Training")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, output_dir, "eval_learning_curve")


def fig_eval_return_distribution(eval_results, output_dir):
    """Box/violin of return at each evaluated checkpoint."""
    n = len(eval_results)
    if n < 3:
        return

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 5))
    data = [e["all_returns"] for e in eval_results]
    labels = [f"{e['update']}" for e in eval_results]
    positions = range(n)

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_xlabel("Update #")
    ax.set_ylabel("Episode Return")
    ax.set_title("Return Distribution Across Training")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "eval_return_distribution")


def fig_eval_strehl_heatmap(eval_results, output_dir):
    """Heatmap: columns = checkpoints, rows = episodes, color = final Strehl."""
    n_ckpts = len(eval_results)
    n_eps = len(eval_results[0]["all_final_strehls"])
    grid = np.zeros((n_eps, n_ckpts))
    for j, e in enumerate(eval_results):
        grid[:, j] = sorted(e["all_final_strehls"])

    fig, ax = plt.subplots(figsize=(max(10, n_ckpts * 0.3), 4))
    im = ax.imshow(grid, aspect="auto", cmap="inferno", vmin=0, vmax=1,
                    origin="lower")
    ax.set_xlabel("Checkpoint (update #)")
    ax.set_ylabel("Episode (sorted by Strehl)")
    ax.set_xticks(range(0, n_ckpts, max(1, n_ckpts // 15)))
    ax.set_xticklabels([f"{eval_results[i]['update']}" for i in range(0, n_ckpts, max(1, n_ckpts // 15))],
                        fontsize=7, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Final Strehl", shrink=0.8)
    ax.set_title("Per-Episode Final Strehl Across Training")
    fig.tight_layout()
    _savefig(fig, output_dir, "eval_strehl_heatmap")


def fig_eval_consistency(eval_results, output_dir):
    """Std of final Strehl across episodes vs training step — measures consistency."""
    steps = [e["global_step"] for e in eval_results]
    stds = [e["std_final_strehl"] for e in eval_results]
    ret_stds = [e["std_return"] for e in eval_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(steps, stds, "o-", color="crimson", linewidth=1.5, markersize=4)
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Std of Final Strehl")
    ax1.set_title("Strehl Consistency Over Training")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, ret_stds, "o-", color="purple", linewidth=1.5, markersize=4)
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Std of Episode Return")
    ax2.set_title("Return Consistency Over Training")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, output_dir, "eval_consistency")


def fig_combined_tb_eval(scalars, eval_results, output_dir):
    """Overlay TB training metrics with checkpoint eval on same axes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: episodic return from TB + eval return
    ret_tag = "train/episodic_return"
    if ret_tag in scalars:
        s = scalars[ret_tag]
        ax1.plot(s["steps"], s["values"], color="lightsteelblue", linewidth=0.3, alpha=0.4,
                 label="TB episodic return (raw)")
        if len(s["values"]) > 50:
            w = max(len(s["values"]) // 50, 10)
            smooth = np.convolve(s["values"], np.ones(w)/w, mode="valid")
            ax1.plot(s["steps"][w-1:], smooth, color="steelblue", linewidth=1.5,
                     label="TB episodic return (smoothed)")

    if eval_results:
        eval_steps = [e["global_step"] for e in eval_results]
        eval_returns = [e["mean_return"] for e in eval_results]
        ax1.plot(eval_steps, eval_returns, "D-", color="darkred", linewidth=2,
                 markersize=5, label="Eval return (fixed seeds)", zorder=5)

    ax1.set_ylabel("Episode Return")
    ax1.set_title("Training vs Evaluation Return")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom: step Strehl from TB + eval Strehl
    strehl_tag = "train/step_strehl"
    if strehl_tag in scalars:
        s = scalars[strehl_tag]
        ax2.plot(s["steps"], s["values"], color="moccasin", linewidth=0.3, alpha=0.4,
                 label="TB step Strehl (raw)")
        if len(s["values"]) > 50:
            w = max(len(s["values"]) // 50, 10)
            smooth = np.convolve(s["values"], np.ones(w)/w, mode="valid")
            ax2.plot(s["steps"][w-1:], smooth, color="darkorange", linewidth=1.5,
                     label="TB step Strehl (smoothed)")

    if eval_results:
        eval_steps = [e["global_step"] for e in eval_results]
        eval_final = [e["mean_final_strehl"] for e in eval_results]
        ax2.plot(eval_steps, eval_final, "D-", color="darkred", linewidth=2,
                 markersize=5, label="Eval final Strehl (fixed seeds)", zorder=5)

    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Strehl Ratio")
    ax2.set_title("Training vs Evaluation Strehl")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, output_dir, "combined_tb_eval")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate training analysis figures for a PTT run")
    parser.add_argument("run_dir", type=str,
                        help="Path to run directory (e.g. runs/ppo_optomech_1_...)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip checkpoint evaluation (only plot TB + cached evals)")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate every Nth checkpoint (default: 10)")
    parser.add_argument("--eval-episodes", type=int, default=4,
                        help="Episodes per checkpoint eval (default: 4)")
    parser.add_argument("--max-episode-steps", type=int, default=128,
                        help="Steps per eval episode (default: 128, shorter than full 256)")
    parser.add_argument("--env-version", type=str, default="v4",
                        help="Env version for eval (default: v4)")
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip("/")
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Output: {analysis_dir}\n")

    # ── Tensorboard metrics ──────────────────────────────────────────
    tb_dir = os.path.join(run_dir, "tensorboard")
    scalars = {}
    if os.path.isdir(tb_dir):
        print("Extracting tensorboard metrics...")
        scalars = extract_tb_scalars(tb_dir)
        print(f"  Found {len(scalars)} scalar tags\n")

        fig_tb_training_curves(scalars, analysis_dir)
        fig_tb_performance(scalars, analysis_dir)
        fig_tb_curriculum(scalars, analysis_dir)
        fig_tb_reward_vs_curriculum(scalars, analysis_dir)
        fig_tb_kl_entropy(scalars, analysis_dir)
    else:
        print(f"No tensorboard directory found at {tb_dir}\n")

    # ── Checkpoint evaluation ────────────────────────────────────────
    eval_cache = os.path.join(analysis_dir, "eval_results.json")
    eval_results = []

    if not args.skip_eval:
        print("Evaluating checkpoints...")
        t0 = time.time()
        eval_results = evaluate_checkpoints(
            run_dir,
            eval_every=args.eval_every,
            num_episodes=args.eval_episodes,
            max_episode_steps=args.max_episode_steps,
            env_version=args.env_version,
        )
        elapsed = time.time() - t0
        print(f"\nEval complete: {len(eval_results)} checkpoints in {elapsed:.0f}s\n")

        # Cache results
        with open(eval_cache, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"  Cached: {eval_cache}")

    elif os.path.exists(eval_cache):
        print(f"Loading cached eval results from {eval_cache}")
        with open(eval_cache) as f:
            eval_results = json.load(f)
        print(f"  Loaded {len(eval_results)} entries\n")
    else:
        print("No eval results (use without --skip-eval to generate)\n")

    # ── Eval figures ─────────────────────────────────────────────────
    if eval_results:
        fig_eval_learning_curve(eval_results, analysis_dir)
        fig_eval_return_distribution(eval_results, analysis_dir)
        fig_eval_strehl_heatmap(eval_results, analysis_dir)
        fig_eval_consistency(eval_results, analysis_dir)

    # ── Combined figures ─────────────────────────────────────────────
    if scalars and eval_results:
        fig_combined_tb_eval(scalars, eval_results, analysis_dir)

    n_figs = len(glob.glob(os.path.join(analysis_dir, "*.png")))
    print(f"\nDone. {n_figs} figures in: {analysis_dir}")


if __name__ == "__main__":
    main()
