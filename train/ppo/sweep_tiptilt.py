"""
Tip/tilt error sweep: evaluate agent performance across a range of
initial tip/tilt disturbance magnitudes.

For each disturbance level, runs N rollouts and records the median
Strehl ratio (across episode steps), comparing agent vs zero-action
baseline.

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

import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import imageio

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.rollout import run_rollouts, _DEFAULT_OUTPUT_DIR
from train.ppo.train_ppo_nanoelf_ptt import NANOELF_TT_ENV_KWARGS
from train.ppo.train_ppo_optomech import register_optomech, _prepare_obs_raw


def _run_zero_action_episodes(env_kwargs, env_version, seeds, max_episode_steps):
    """Run zero-action episodes and collect per-step Strehl ratios.

    Returns list of dicts with 'strehls' (per-step list) per episode.
    """
    register_optomech(f"optomech-{env_version}", max_episode_steps=max_episode_steps)
    env_kw = dict(env_kwargs)
    env_kw["max_episode_steps"] = max_episode_steps
    env = gym.make(f"optomech-{env_version}", **env_kw)

    episodes = []
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        strehls = []
        while not done:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            strehls.append(float(info.get("strehl", 0.0)))
        episodes.append({"strehls": strehls})
    env.close()
    return episodes


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
    all_sweep_episodes = []  # list of (tt_val, episodes_list) for GIF rendering
    for i, tt in enumerate(tt_values):
        print(f"\n--- Tip/tilt = {tt:.3f} arcsec ({i+1}/{len(tt_values)}) ---")

        env_kwargs = dict(NANOELF_TT_ENV_KWARGS)
        env_kwargs["init_wind_tip_arcsec_std_tt"] = float(tt)
        env_kwargs["init_wind_tilt_arcsec_std_tt"] = float(tt)

        # Agent rollouts
        episodes, metrics = run_rollouts(
            checkpoint_path=args.checkpoint,
            policy_spec_path=getattr(args, 'policy_spec', None),
            env_kwargs=env_kwargs,
            env_version=args.env_version,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            seeds=seeds,
        )

        # Zero-action baseline (with per-step Strehl tracking)
        print(f"  Running zero-action baseline...")
        zero_episodes = _run_zero_action_episodes(
            env_kwargs, args.env_version, seeds[:args.num_episodes],
            args.max_episode_steps,
        )

        # Raw per-step Strehl traces for every episode
        agent_strehls_raw = [e["strehls"] for e in episodes]
        zero_strehls_raw = [e["strehls"] for e in zero_episodes]

        entry = {
            "tip_tilt_arcsec": float(tt),
            # Raw per-step Strehls: list of episodes, each a list of floats
            "agent_strehls": agent_strehls_raw,
            "zero_strehls": zero_strehls_raw,
            # Summary stats for quick inspection
            "mean_final_strehl": float(np.mean([e["strehls"][-1] for e in episodes])),
            "mean_return": metrics["mean_return"],
        }
        results.append(entry)
        all_sweep_episodes.append((float(tt), episodes))
        _agent_med = float(np.median([np.median(t) for t in agent_strehls_raw]))
        _zero_med = float(np.median([np.median(t) for t in zero_strehls_raw]))
        print(f"  Agent median Strehl: {_agent_med:.4f}  "
              f"Zero baseline: {_zero_med:.4f}")

    # Save raw results
    results_path = os.path.join(test_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}")

    # Derive summary metrics from raw traces and generate figures
    _derive_metrics(results)
    _fig_strehl_vs_tiptilt(results, test_dir)
    _fig_return_vs_tiptilt(results, test_dir)
    _fig_frac_above_threshold(results, test_dir, threshold="0.9")
    _fig_frac_above_threshold(results, test_dir, threshold="0.8")
    _fig_strehl_decile_bars(results, test_dir)
    _fig_cumulative_strehl_bars(results, test_dir)
    _fig_strehl_violins(results, test_dir)

    # Generate sweep GIFs (best, median, worst across tt levels)
    if not args.no_gifs:
        gif_dir = os.path.join(test_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        _render_sweep_gifs(all_sweep_episodes, gif_dir)

    print(f"All outputs in: {test_dir}")


def _savefig(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _derive_metrics(results):
    """Derive per-episode summary metrics from raw Strehl traces.

    Adds computed fields to each result entry in-place and returns results.
    """
    _thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for r in results:
        agent_traces = r["agent_strehls"]
        zero_traces = r["zero_strehls"]

        r["agent_all_median_strehls"] = [float(np.median(t)) for t in agent_traces]
        r["zero_all_median_strehls"] = [float(np.median(t)) for t in zero_traces]
        r["agent_median_strehl"] = float(np.median(r["agent_all_median_strehls"]))
        r["zero_median_strehl"] = float(np.median(r["zero_all_median_strehls"]))

        agent_frac = {}
        zero_frac = {}
        for thr in _thresholds:
            key = f"{thr:.1f}"
            agent_frac[key] = [float(np.mean(np.array(t) >= thr)) for t in agent_traces]
            zero_frac[key] = [float(np.mean(np.array(t) >= thr)) for t in zero_traces]
        r["agent_frac_above"] = agent_frac
        r["zero_frac_above"] = zero_frac
    return results


def _fig_strehl_vs_tiptilt(results, output_dir):
    """Median Strehl ratio vs initial tip/tilt error: agent vs zero-action."""
    tt = [r["tip_tilt_arcsec"] for r in results]
    agent_med = [r["agent_median_strehl"] for r in results]
    zero_med = [r["zero_median_strehl"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual episode medians as scatter
    for r in results:
        tt_val = r["tip_tilt_arcsec"]
        for s in r["agent_all_median_strehls"]:
            ax.scatter(tt_val, s, color="steelblue", alpha=0.15, s=12,
                       edgecolors="none", zorder=2)
        for s in r["zero_all_median_strehls"]:
            ax.scatter(tt_val, s, color="salmon", alpha=0.15, s=12,
                       edgecolors="none", zorder=2)

    # Median lines
    ax.plot(tt, agent_med, "o-", color="darkblue", linewidth=2, markersize=6,
            label="Agent (median)", zorder=3)
    ax.plot(tt, zero_med, "s--", color="firebrick", linewidth=2, markersize=6,
            label="Zero-action (median)", zorder=3)

    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Median Strehl Ratio")
    ax.set_title("Agent vs Zero-Action Baseline")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "strehl_vs_tiptilt")
    print(f"  Figure: strehl_vs_tiptilt.png/pdf")


def _fig_frac_above_threshold(results, output_dir, threshold="0.9"):
    """Fraction of episode steps with Strehl >= threshold, vs tip/tilt error."""
    tt = [r["tip_tilt_arcsec"] for r in results]

    # Check that frac_above data exists
    if "agent_frac_above" not in results[0]:
        print(f"  Skipping frac_above plot (data not in results; re-run sweep)")
        return

    agent_all = [r["agent_frac_above"][threshold] for r in results]
    zero_all = [r["zero_frac_above"][threshold] for r in results]
    agent_means = [float(np.mean(vals)) for vals in agent_all]
    zero_means = [float(np.mean(vals)) for vals in zero_all]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual episode points
    for i, tt_val in enumerate(tt):
        for v in agent_all[i]:
            ax.scatter(tt_val, v, color="steelblue", alpha=0.15, s=12,
                       edgecolors="none", zorder=2)
        for v in zero_all[i]:
            ax.scatter(tt_val, v, color="salmon", alpha=0.15, s=12,
                       edgecolors="none", zorder=2)

    # Mean lines
    ax.plot(tt, agent_means, "o-", color="darkblue", linewidth=2, markersize=6,
            label="Agent (mean)", zorder=3)
    ax.plot(tt, zero_means, "s--", color="firebrick", linewidth=2, markersize=6,
            label="Zero-action (mean)", zorder=3)

    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel(f"Fraction of Steps with Strehl ≥ {threshold}")
    ax.set_title(f"Time Above Strehl {threshold}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, f"frac_above_{threshold.replace('.', '')}")
    print(f"  Figure: frac_above_{threshold.replace('.', '')}.png/pdf")


def _fig_strehl_decile_bars(results, output_dir):
    """Stacked bar chart: fraction of all Strehl samples in each decile bin.

    Two bars per TT level (zero-action baseline, agent), stacked by decile
    [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
    """
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    n_tt = len(tt_vals)
    bin_edges = np.arange(0.0, 1.1, 0.1)  # 0.0, 0.1, ..., 1.0
    n_bins = len(bin_edges) - 1
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    # Colormap: red (low) → yellow (mid) → green (high)
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    # Compute fractions for each TT level
    zero_fracs = np.zeros((n_tt, n_bins))
    agent_fracs = np.zeros((n_tt, n_bins))
    for i, r in enumerate(results):
        # Pool ALL steps from ALL episodes into one histogram
        zero_all = np.concatenate(r["zero_strehls"])
        agent_all = np.concatenate(r["agent_strehls"])
        zero_fracs[i] = np.histogram(zero_all, bins=bin_edges)[0] / len(zero_all)
        agent_fracs[i] = np.histogram(agent_all, bins=bin_edges)[0] / len(agent_all)

    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n_tt * 0.9), 5))
    x = np.arange(n_tt)

    # Stacked bars: zero-action (left), agent (right)
    for data, offset, group_label in [
        (zero_fracs, -bar_width / 2, "Zero"),
        (agent_fracs, bar_width / 2, "Agent"),
    ]:
        bottom = np.zeros(n_tt)
        for b in range(n_bins):
            label = f"{bin_labels[b]} ({group_label})" if b == n_bins - 1 else None
            ax.bar(x + offset, data[:, b], bar_width, bottom=bottom,
                   color=colors[b], edgecolor="white", linewidth=0.3,
                   label=None)
            bottom += data[:, b]

    # Legend for decile colors (shared across both groups)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[b], label=bin_labels[b])
                      for b in range(n_bins)]
    leg = ax.legend(handles=legend_patches, title="Strehl range",
                    bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                    title_fontsize=8)

    # Labels for zero/agent groups
    ax.text(x[0] - bar_width / 2, -0.06, "Z", ha="center", fontsize=6,
            color="gray", transform=ax.get_xaxis_transform())
    ax.text(x[0] + bar_width / 2, -0.06, "A", ha="center", fontsize=6,
            color="gray", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Fraction of Strehl Samples")
    ax.set_title("Strehl Distribution by Decile: Zero-Action (left) vs Agent (right)")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _savefig(fig, output_dir, "strehl_decile_bars")
    print(f"  Figure: strehl_decile_bars.png/pdf")


def _fig_cumulative_strehl_bars(results, output_dir):
    """Stacked bar chart: cumulative Strehl contribution by decile.

    Instead of fractions, each decile segment shows the *sum* of Strehl values
    from samples in that bin, normalized by the total number of samples.
    This means the maximum possible bar height is 1.0 (every sample at Strehl=1.0).
    Bar height = mean Strehl, with color showing which decile range contributes.
    """
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    n_tt = len(tt_vals)
    bin_edges = np.arange(0.0, 1.1, 0.1)
    n_bins = len(bin_edges) - 1
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    cmap = plt.cm.RdYlGn
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    # For each TT level, sum Strehl values contributed by each decile bin (no normalization)
    zero_contrib = np.zeros((n_tt, n_bins))
    agent_contrib = np.zeros((n_tt, n_bins))
    max_possible = 0.0
    for i, r in enumerate(results):
        zero_all = np.concatenate(r["zero_strehls"])
        agent_all = np.concatenate(r["agent_strehls"])
        max_possible = max(max_possible, len(zero_all), len(agent_all))  # N_episodes * N_steps
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            z_mask = (zero_all >= lo) & (zero_all < hi) if b < n_bins - 1 else (zero_all >= lo) & (zero_all <= hi)
            a_mask = (agent_all >= lo) & (agent_all < hi) if b < n_bins - 1 else (agent_all >= lo) & (agent_all <= hi)
            zero_contrib[i, b] = np.sum(zero_all[z_mask])
            agent_contrib[i, b] = np.sum(agent_all[a_mask])

    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n_tt * 0.9), 5))
    x = np.arange(n_tt)

    for data, offset, group_label in [
        (zero_contrib, -bar_width / 2, "Zero"),
        (agent_contrib, bar_width / 2, "Agent"),
    ]:
        bottom = np.zeros(n_tt)
        for b in range(n_bins):
            ax.bar(x + offset, data[:, b], bar_width, bottom=bottom,
                   color=colors[b], edgecolor="white", linewidth=0.3)
            bottom += data[:, b]

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[b], label=bin_labels[b])
                      for b in range(n_bins)]
    ax.legend(handles=legend_patches, title="Strehl range",
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              title_fontsize=8)

    ax.text(x[0] - bar_width / 2, -0.04, "Z", ha="center", fontsize=6,
            color="gray", transform=ax.get_xaxis_transform())
    ax.text(x[0] + bar_width / 2, -0.04, "A", ha="center", fontsize=6,
            color="gray", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Cumulative Strehl")
    ax.set_title("Cumulative Strehl by Decile: Zero-Action (left) vs Agent (right)")
    ax.set_ylim(0, max_possible * 1.05)
    fig.tight_layout()
    _savefig(fig, output_dir, "cumulative_strehl_bars")
    print(f"  Figure: cumulative_strehl_bars.png/pdf")


def _fig_strehl_violins(results, output_dir):
    """Side-by-side violin plots of ALL per-step Strehl values at each TT level.

    Every single step across all episodes is pooled into one distribution per
    TT level. Zero-action (left/red) vs Agent (right/blue).
    """
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    n_tt = len(tt_vals)

    zero_pools = []
    agent_pools = []
    for r in results:
        zero_pools.append(np.concatenate(r["zero_strehls"]))
        agent_pools.append(np.concatenate(r["agent_strehls"]))

    fig, ax = plt.subplots(figsize=(max(9, n_tt * 1.0), 5.5))
    x = np.arange(n_tt)
    width = 0.38  # half-width of each violin

    for i in range(n_tt):
        for data, side, color, label in [
            (zero_pools[i], -1, "firebrick", "Zero-action"),
            (agent_pools[i], +1, "steelblue", "Agent"),
        ]:
            parts = ax.violinplot(
                data, positions=[x[i] + side * width / 2],
                widths=width, showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                # Clip each violin to its own side
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                if side == -1:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], -np.inf, m)
                else:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], m, np.inf)
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.5)
                pc.set_alpha(0.7)

            # Add median line
            med = np.median(data)
            hw = width * 0.35
            if side == -1:
                ax.plot([x[i] - hw, x[i]], [med, med], color=color,
                        linewidth=1.5, solid_capstyle="round")
            else:
                ax.plot([x[i], x[i] + hw], [med, med], color=color,
                        linewidth=1.5, solid_capstyle="round")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="firebrick", alpha=0.7, label="Zero-action"),
        Patch(facecolor="steelblue", alpha=0.7, label="Agent"),
    ], loc="lower left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Strehl Ratio")
    ax.set_title("Per-Step Strehl Distribution: Zero-Action vs Agent")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "strehl_violins")
    print(f"  Figure: strehl_violins.png/pdf")


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


def _select_episodes(episodes, rank):
    """Select best, median, or worst episode by median Strehl.

    rank: 'best', 'median', or 'worst'
    """
    scored = [(np.median(e["strehls"]), e) for e in episodes]
    scored.sort(key=lambda x: x[0])
    if rank == "worst":
        return scored[0][1]
    elif rank == "best":
        return scored[-1][1]
    else:  # median
        return scored[len(scored) // 2][1]


def _render_sweep_gifs(all_sweep_episodes, output_dir, dpi=72, frame_duration=0.2):
    """Render 3 GIFs (best, median, worst) across all sweep points.

    Each GIF has one column per tt level (increasing left to right).
    Row 1: focal-plane observation image.
    Row 2: action bar chart.
    """
    n_cols = len(all_sweep_episodes)

    for rank in ("best", "median", "worst"):
        print(f"  Rendering {rank} sweep GIF ({n_cols} columns)...")

        # Pick the episode for each tt level
        selected = []
        for tt_val, episodes in all_sweep_episodes:
            ep = _select_episodes(episodes, rank)
            selected.append((tt_val, ep))

        # Find the max episode length across all selected episodes
        max_T = max(len(ep["rewards"]) for _, ep in selected)

        # Pre-render all observations
        all_obs = []
        for _, ep in selected:
            obs_raw = ep["obs_raw"]
            T = len(ep["rewards"])
            rendered = [_prepare_obs_raw(obs_raw[t]) for t in range(T + 1)]
            all_obs.append(rendered)

        # Global colormap normalization across all columns
        global_max = 1.0
        for rendered in all_obs:
            for img in rendered:
                global_max = max(global_max, float(np.max(img)))
        norm = mcolors.LogNorm(vmin=1.0, vmax=global_max)

        # Action dim from first episode
        action_dim = np.array(selected[0][1]["actions"]).shape[1]
        act_colors = (["#4a90d9", "#d94a4a", "#4ad94a",
                       "#d9d94a", "#d94ad9", "#4ad9d9"] * 2)[:action_dim]

        col_width = 2.5
        fig_w = col_width * n_cols + 0.5
        fig_h = 4.5

        frames = []
        for t in range(max_T + 1):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            gs = gridspec.GridSpec(
                2, n_cols, figure=fig,
                height_ratios=[3, 1], hspace=0.35, wspace=0.3,
                left=0.02, right=0.98, top=0.88, bottom=0.05,
            )

            fig.suptitle(
                f"{rank.upper()} episode per TT level  |  t = {t} / {max_T}",
                fontsize=9, fontweight="bold",
            )

            for col, (tt_val, ep) in enumerate(selected):
                T_ep = len(ep["rewards"])
                strehls = ep["strehls"]
                actions = np.array(ep["actions"])
                obs_rendered = all_obs[col]

                # Clamp t to this episode's length
                t_ep = min(t, T_ep)

                # Row 1: observation image
                ax_obs = fig.add_subplot(gs[0, col])
                img_dn = obs_rendered[t_ep]
                ax_obs.imshow(np.maximum(img_dn, 1.0), cmap="inferno", norm=norm)
                ax_obs.axis("off")
                strehl_txt = (f"S={strehls[t_ep - 1]:.3f}" if t_ep > 0
                              else "S=--")
                ax_obs.set_title(
                    f"TT={tt_val:.2f}\"\n{strehl_txt}",
                    fontsize=7,
                )

                # Row 2: action bar chart
                ax_act = fig.add_subplot(gs[1, col])
                if t_ep > 0 and t_ep <= T_ep:
                    act = actions[t_ep - 1]
                    ax_act.barh(range(action_dim), act, color=act_colors)
                    ax_act.set_xlim(-1.1, 1.1)
                    ax_act.set_yticks(range(action_dim))
                    ax_act.set_yticklabels(
                        [f"a{i}" for i in range(action_dim)], fontsize=5)
                    ax_act.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
                    ax_act.tick_params(axis="x", labelsize=5)
                else:
                    ax_act.text(0.5, 0.5, "t=0", ha="center", va="center",
                                fontsize=6, transform=ax_act.transAxes)
                    ax_act.set_xticks([])
                    ax_act.set_yticks([])

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())
            plt.close(fig)

        save_path = os.path.join(output_dir, f"sweep_{rank}.gif")
        imageio.mimsave(save_path, frames, duration=frame_duration)
        print(f"    {save_path}")


def replot(results_path):
    """Re-generate analysis plots from a previously saved sweep_results.json."""
    with open(results_path, "r") as f:
        results = json.load(f)
    output_dir = os.path.dirname(results_path)
    print(f"Re-plotting from: {results_path}")
    print(f"Output directory: {output_dir}")
    _derive_metrics(results)
    _fig_strehl_vs_tiptilt(results, output_dir)
    _fig_return_vs_tiptilt(results, output_dir)
    _fig_frac_above_threshold(results, output_dir, threshold="0.9")
    _fig_frac_above_threshold(results, output_dir, threshold="0.8")
    _fig_strehl_decile_bars(results, output_dir)
    _fig_cumulative_strehl_bars(results, output_dir)
    _fig_strehl_violins(results, output_dir)
    print(f"All outputs in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep tip/tilt error and plot agent performance")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--policy-spec", type=str, default=None,
                        help="Path to policy specification YAML (alternative to --checkpoint)")
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing sweep_results.json to re-plot without re-running")
    parser.add_argument("--env-version", type=str, default="v3",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--num-episodes", type=int, default=8,
                        help="Episodes per sweep point")
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--tt-min", type=float, default=0.0,
                        help="Min tip/tilt error in arcsec")
    parser.add_argument("--tt-max", type=float, default=2.0,
                        help="Max tip/tilt error in arcsec")
    parser.add_argument("--tt-steps", type=int, default=13,
                        help="Number of sweep points")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip GIF generation")
    args = parser.parse_args()

    if args.replot:
        replot(args.replot)
    else:
        if args.checkpoint is None and args.policy_spec is None:
            parser.error("--checkpoint or --policy-spec is required when not using --replot")
        run_sweep(args)


if __name__ == "__main__":
    main()
