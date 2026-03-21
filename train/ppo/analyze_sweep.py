"""
Generate a comprehensive set of analysis figures from a sweep_results.json.

Usage:
    poetry run python train/ppo/analyze_sweep.py \
        test_output/sweep_tiptilt_20260320_094953_6193/sweep_results.json

All figures are saved alongside the input JSON.
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib import cm


def _savefig(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def load_results(path):
    with open(path) as f:
        results = json.load(f)
    # Ensure numpy arrays for convenience
    for r in results:
        r["_agent"] = [np.array(ep, dtype=np.float64) for ep in r["agent_strehls"]]
        r["_zero"] = [np.array(ep, dtype=np.float64) for ep in r["zero_strehls"]]
    return results


# ──────────────────────────────────────────────────────────────────────
# 1. Convergence curves: mean Strehl vs timestep, one line per TT level
# ──────────────────────────────────────────────────────────────────────
def fig_convergence_curves(results, output_dir):
    """Mean Strehl vs timestep within episode, one curve per TT level."""
    fig, (ax_a, ax_z) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = cm.viridis
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    norm = Normalize(vmin=min(tt_vals), vmax=max(tt_vals))

    for r in results:
        tt = r["tip_tilt_arcsec"]
        color = cmap(norm(tt))

        # Pad episodes to equal length, then average
        agent_traces = r["_agent"]
        zero_traces = r["_zero"]
        max_len = max(max(len(t) for t in agent_traces), max(len(t) for t in zero_traces))

        def padded_mean(traces, length):
            mat = np.full((len(traces), length), np.nan)
            for i, t in enumerate(traces):
                mat[i, :len(t)] = t
            return np.nanmean(mat, axis=0)

        agent_mean = padded_mean(agent_traces, max_len)
        zero_mean = padded_mean(zero_traces, max_len)

        ax_a.plot(agent_mean, color=color, linewidth=1.0, alpha=0.8)
        ax_z.plot(zero_mean, color=color, linewidth=1.0, alpha=0.8)

    for ax, title in [(ax_a, "Agent"), (ax_z, "Zero-Action")]:
        ax.set_xlabel("Timestep")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    ax_a.set_ylabel("Mean Strehl Ratio")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_a, ax_z], label="Initial TT Error (arcsec)", shrink=0.8)
    fig.suptitle("Strehl Convergence Over Episode", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "convergence_curves")


# ──────────────────────────────────────────────────────────────────────
# 2. Convergence heatmap: TT level (y) vs timestep (x), color = mean Strehl
# ──────────────────────────────────────────────────────────────────────
def fig_convergence_heatmap(results, output_dir):
    """2D heatmap: x=timestep, y=TT level, color=mean Strehl."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    max_len = max(max(len(t) for t in r["_agent"]) for r in results)

    agent_grid = np.full((len(results), max_len), np.nan)
    zero_grid = np.full((len(results), max_len), np.nan)

    for i, r in enumerate(results):
        for traces, grid in [(r["_agent"], agent_grid), (r["_zero"], zero_grid)]:
            mat = np.full((len(traces), max_len), np.nan)
            for j, t in enumerate(traces):
                mat[j, :len(t)] = t
            grid[i] = np.nanmean(mat, axis=0)

    fig, (ax_a, ax_z) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, grid, title in [(ax_a, agent_grid, "Agent"), (ax_z, zero_grid, "Zero-Action")]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="inferno",
                        vmin=0, vmax=1,
                        extent=[0, max_len, tt_vals[0], tt_vals[-1]])
        ax.set_ylabel("Initial TT Error (arcsec)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Mean Strehl", shrink=0.8)

    ax_z.set_xlabel("Timestep")
    fig.suptitle("Strehl Convergence Heatmap", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "convergence_heatmap")


# ──────────────────────────────────────────────────────────────────────
# 3. Agent advantage heatmap: agent - zero at each (TT, timestep)
# ──────────────────────────────────────────────────────────────────────
def fig_advantage_heatmap(results, output_dir):
    """2D heatmap of agent Strehl minus zero-action Strehl."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    max_len = max(max(len(t) for t in r["_agent"]) for r in results)

    diff_grid = np.full((len(results), max_len), np.nan)
    for i, r in enumerate(results):
        def padded_mean(traces):
            mat = np.full((len(traces), max_len), np.nan)
            for j, t in enumerate(traces):
                mat[j, :len(t)] = t
            return np.nanmean(mat, axis=0)
        diff_grid[i] = padded_mean(r["_agent"]) - padded_mean(r["_zero"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    vmax = max(abs(np.nanmin(diff_grid)), abs(np.nanmax(diff_grid)))
    im = ax.imshow(diff_grid, aspect="auto", origin="lower", cmap="RdBu",
                    vmin=-vmax, vmax=vmax,
                    extent=[0, max_len, tt_vals[0], tt_vals[-1]])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Initial TT Error (arcsec)")
    ax.set_title("Agent Advantage (Agent Strehl − Zero-Action Strehl)")
    fig.colorbar(im, ax=ax, label="Strehl Difference", shrink=0.8)
    fig.tight_layout()
    _savefig(fig, output_dir, "advantage_heatmap")


# ──────────────────────────────────────────────────────────────────────
# 4. Steady-state Strehl (last N steps) vs TT
# ──────────────────────────────────────────────────────────────────────
def fig_steady_state(results, output_dir, tail_steps=64):
    """Mean Strehl over the last N steps of each episode, vs TT level."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    agent_means = []
    zero_means = []

    for r in results:
        a_tails = [np.mean(t[-tail_steps:]) for t in r["_agent"] if len(t) >= tail_steps]
        z_tails = [np.mean(t[-tail_steps:]) for t in r["_zero"] if len(t) >= tail_steps]

        # Scatter individual episodes
        for v in a_tails:
            ax.scatter(r["tip_tilt_arcsec"], v, color="steelblue", alpha=0.12,
                       s=12, edgecolors="none", zorder=2)
        for v in z_tails:
            ax.scatter(r["tip_tilt_arcsec"], v, color="salmon", alpha=0.12,
                       s=12, edgecolors="none", zorder=2)

        agent_means.append(np.mean(a_tails))
        zero_means.append(np.mean(z_tails))

    ax.plot(tt_vals, agent_means, "o-", color="darkblue", linewidth=2, markersize=6,
            label="Agent (mean)", zorder=3)
    ax.plot(tt_vals, zero_means, "s--", color="firebrick", linewidth=2, markersize=6,
            label="Zero-action (mean)", zorder=3)

    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel(f"Mean Strehl (last {tail_steps} steps)")
    ax.set_title(f"Steady-State Performance (last {tail_steps} steps)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, f"steady_state_{tail_steps}")


# ──────────────────────────────────────────────────────────────────────
# 5. Time-to-threshold: steps until Strehl first exceeds 0.8
# ──────────────────────────────────────────────────────────────────────
def fig_time_to_threshold(results, output_dir, threshold=0.8):
    """Box plot: how many steps until Strehl first reaches threshold."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    agent_times = []
    zero_times = []

    for r in results:
        a_t = []
        z_t = []
        for traces, out in [(r["_agent"], a_t), (r["_zero"], z_t)]:
            for t in traces:
                above = np.where(t >= threshold)[0]
                if len(above) > 0:
                    out.append(above[0])
                else:
                    out.append(len(t))  # never reached
        agent_times.append(a_t)
        zero_times.append(z_t)

    fig, ax = plt.subplots(figsize=(max(9, len(tt_vals) * 0.9), 5))
    x = np.arange(len(tt_vals))
    width = 0.35

    bp_a = ax.boxplot([agent_times[i] for i in range(len(tt_vals))],
                       positions=x - width/2, widths=width*0.8,
                       patch_artist=True, showfliers=False)
    bp_z = ax.boxplot([zero_times[i] for i in range(len(tt_vals))],
                       positions=x + width/2, widths=width*0.8,
                       patch_artist=True, showfliers=False)

    for bp, color in [(bp_a, "steelblue"), (bp_z, "salmon")]:
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for elem in ["whiskers", "caps"]:
            for line in bp[elem]:
                line.set_color(color)
        for line in bp["medians"]:
            line.set_color("black")

    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Agent"),
        Patch(facecolor="salmon", alpha=0.7, label="Zero-action"),
    ])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel(f"Steps to Reach Strehl ≥ {threshold}")
    ax.set_title(f"Convergence Speed (Time to Strehl ≥ {threshold})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, f"time_to_{str(threshold).replace('.', '')}")


# ──────────────────────────────────────────────────────────────────────
# 6. Strehl trajectory ribbon: mean ± std over time, per TT level
# ──────────────────────────────────────────────────────────────────────
def fig_trajectory_ribbons(results, output_dir):
    """Individual panels per TT level showing mean ± 1σ ribbon."""
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                              sharex=True, sharey=True, squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        tt = r["tip_tilt_arcsec"]

        for traces, color, label in [
            (r["_agent"], "steelblue", "Agent"),
            (r["_zero"], "firebrick", "Zero"),
        ]:
            max_len = max(len(t) for t in traces)
            mat = np.full((len(traces), max_len), np.nan)
            for j, t in enumerate(traces):
                mat[j, :len(t)] = t
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            steps = np.arange(max_len)
            ax.plot(steps, mean, color=color, linewidth=1.2, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(f"TT = {tt:.2f}\"", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.supxlabel("Timestep")
    fig.supylabel("Strehl Ratio")
    fig.suptitle("Strehl Trajectories (Mean ± 1σ)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    _savefig(fig, output_dir, "trajectory_ribbons")


# ──────────────────────────────────────────────────────────────────────
# 7. Success rate: fraction of episodes where final Strehl > threshold
# ──────────────────────────────────────────────────────────────────────
def fig_success_rate(results, output_dir):
    """Fraction of episodes achieving final Strehl above various thresholds."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    cmap = cm.cool

    fig, (ax_a, ax_z) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ti, thr in enumerate(thresholds):
        color = cmap(ti / (len(thresholds) - 1))
        agent_rates = []
        zero_rates = []
        for r in results:
            a_final = [t[-1] for t in r["_agent"]]
            z_final = [t[-1] for t in r["_zero"]]
            agent_rates.append(np.mean(np.array(a_final) >= thr))
            zero_rates.append(np.mean(np.array(z_final) >= thr))

        ax_a.plot(tt_vals, agent_rates, "o-", color=color, linewidth=1.5,
                  markersize=5, label=f"≥ {thr}")
        ax_z.plot(tt_vals, zero_rates, "s--", color=color, linewidth=1.5,
                  markersize=5, label=f"≥ {thr}")

    for ax, title in [(ax_a, "Agent"), (ax_z, "Zero-Action")]:
        ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Final Strehl", fontsize=7)
        ax.grid(True, alpha=0.3)

    ax_a.set_ylabel("Fraction of Episodes (Success Rate)")
    fig.suptitle("Success Rate by Final Strehl Threshold", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "success_rate")


# ──────────────────────────────────────────────────────────────────────
# 8. Improvement factor: agent / zero-action mean Strehl at each timestep
# ──────────────────────────────────────────────────────────────────────
def fig_improvement_factor(results, output_dir):
    """Ratio of agent mean Strehl to zero-action mean Strehl over time."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    cmap = cm.viridis
    norm = Normalize(vmin=min(tt_vals), vmax=max(tt_vals))

    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        tt = r["tip_tilt_arcsec"]
        if tt == 0.0:
            continue  # skip zero TT (ratio ill-defined early on)
        color = cmap(norm(tt))
        max_len = max(max(len(t) for t in r["_agent"]), max(len(t) for t in r["_zero"]))

        def padded_mean(traces):
            mat = np.full((len(traces), max_len), np.nan)
            for j, t in enumerate(traces):
                mat[j, :len(t)] = t
            return np.nanmean(mat, axis=0)

        a_mean = padded_mean(r["_agent"])
        z_mean = padded_mean(r["_zero"])
        ratio = np.where(z_mean > 0.01, a_mean / z_mean, np.nan)
        ax.plot(ratio, color=color, linewidth=1.0, alpha=0.8)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Agent / Zero-Action Strehl Ratio")
    ax.set_title("Improvement Factor Over Episode")
    ax.set_ylim(0, 3.0)
    ax.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Initial TT Error (arcsec)", shrink=0.8)
    fig.tight_layout()
    _savefig(fig, output_dir, "improvement_factor")


# ──────────────────────────────────────────────────────────────────────
# 9. Episode outcome scatter: initial Strehl vs final Strehl
# ──────────────────────────────────────────────────────────────────────
def fig_initial_vs_final(results, output_dir):
    """Scatter of step-0 Strehl vs final Strehl, colored by TT level."""
    cmap = cm.viridis
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    norm = Normalize(vmin=min(tt_vals), vmax=max(tt_vals))

    fig, (ax_a, ax_z) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for r in results:
        tt = r["tip_tilt_arcsec"]
        color = cmap(norm(tt))
        for traces, ax in [(r["_agent"], ax_a), (r["_zero"], ax_z)]:
            for t in traces:
                if len(t) >= 2:
                    ax.scatter(t[0], t[-1], color=color, alpha=0.25, s=14,
                               edgecolors="none")

    for ax, title in [(ax_a, "Agent"), (ax_z, "Zero-Action")]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Initial Strehl (step 0)")
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    ax_a.set_ylabel("Final Strehl (last step)")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_a, ax_z], label="Initial TT Error (arcsec)", shrink=0.8)
    fig.suptitle("Initial vs Final Strehl Per Episode", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "initial_vs_final")


# ──────────────────────────────────────────────────────────────────────
# 10. Cumulative time above threshold curves
# ──────────────────────────────────────────────────────────────────────
def fig_time_above_curves(results, output_dir):
    """For each TT level: fraction of total episode time spent above Strehl=X,
    plotted as a curve X → fraction, for agent and zero-action."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    thresholds = np.linspace(0, 1, 101)

    # Select a few representative TT levels
    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
    indices = sorted(set(indices))

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = cm.viridis
    norm = Normalize(vmin=min(tt_vals), vmax=max(tt_vals))

    for idx in indices:
        r = results[idx]
        tt = r["tip_tilt_arcsec"]
        color = cmap(norm(tt))

        agent_all = np.concatenate(r["_agent"])
        zero_all = np.concatenate(r["_zero"])

        agent_curve = [np.mean(agent_all >= thr) for thr in thresholds]
        zero_curve = [np.mean(zero_all >= thr) for thr in thresholds]

        ax.plot(thresholds, agent_curve, color=color, linewidth=2, label=f"TT={tt:.2f}")
        ax.plot(thresholds, zero_curve, color=color, linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xlabel("Strehl Threshold")
    ax.set_ylabel("Fraction of Time Above Threshold")
    ax.set_title("Complementary CDF of Strehl (solid=Agent, dashed=Zero)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "time_above_curves")


# ──────────────────────────────────────────────────────────────────────
# 11. Per-episode total reward: agent vs zero, violin comparison
# ──────────────────────────────────────────────────────────────────────
def fig_episode_total_strehl(results, output_dir):
    """Per-episode sum-of-Strehls (a proxy for total useful observation time),
    shown as side-by-side violins."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    n_tt = len(tt_vals)

    fig, ax = plt.subplots(figsize=(max(9, n_tt * 1.0), 5.5))
    x = np.arange(n_tt)
    width = 0.38

    for i, r in enumerate(results):
        agent_sums = [np.sum(t) for t in r["_agent"]]
        zero_sums = [np.sum(t) for t in r["_zero"]]

        for data, side, color in [
            (zero_sums, -1, "firebrick"),
            (agent_sums, +1, "steelblue"),
        ]:
            parts = ax.violinplot(
                data, positions=[x[i] + side * width / 2],
                widths=width, showmedians=True, showextrema=False,
            )
            for pc in parts["bodies"]:
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

    ax.legend(handles=[
        Patch(facecolor="firebrick", alpha=0.7, label="Zero-action"),
        Patch(facecolor="steelblue", alpha=0.7, label="Agent"),
    ])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Episode Sum of Strehl")
    ax.set_title("Total Useful Observation (Σ Strehl per Episode)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "episode_total_strehl")


# ──────────────────────────────────────────────────────────────────────
# 12. Strehl percentile envelope: show 10/25/50/75/90 percentiles
# ──────────────────────────────────────────────────────────────────────
def fig_percentile_envelope(results, output_dir):
    """Percentile envelopes of Strehl over time, for a few TT levels."""
    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
    indices = sorted(set(indices))
    n = len(indices)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True, squeeze=False)
    percentiles = [10, 25, 50, 75, 90]
    fills = [(10, 90, 0.15), (25, 75, 0.25)]

    for col, idx in enumerate(indices):
        ax = axes[0][col]
        r = results[idx]
        tt = r["tip_tilt_arcsec"]
        max_len = max(len(t) for t in r["_agent"])
        mat = np.full((len(r["_agent"]), max_len), np.nan)
        for j, t in enumerate(r["_agent"]):
            mat[j, :len(t)] = t

        pcts = {}
        for p in percentiles:
            pcts[p] = np.nanpercentile(mat, p, axis=0)

        steps = np.arange(max_len)
        for lo, hi, alpha in fills:
            ax.fill_between(steps, pcts[lo], pcts[hi], color="steelblue", alpha=alpha)
        ax.plot(steps, pcts[50], color="darkblue", linewidth=1.5, label="Median")

        ax.set_title(f"TT = {tt:.2f}\"", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_ylabel("Strehl Ratio")
            ax.legend(fontsize=7)

    fig.suptitle("Agent Strehl Percentile Envelopes (10/25/50/75/90)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir, "percentile_envelope")


# ──────────────────────────────────────────────────────────────────────
# 13. Agent advantage bar chart: mean agent Strehl minus zero at key timepoints
# ──────────────────────────────────────────────────────────────────────
def fig_advantage_bars(results, output_dir):
    """Bar chart: agent mean Strehl minus zero-action at t=1, 16, 64, 128, 256."""
    tt_vals = [r["tip_tilt_arcsec"] for r in results]
    timepoints = [1, 16, 64, 128, 256]
    # Filter to timepoints that exist
    max_ep_len = max(max(len(t) for t in r["_agent"]) for r in results)
    timepoints = [t for t in timepoints if t <= max_ep_len]

    n_tp = len(timepoints)
    n_tt = len(tt_vals)
    x = np.arange(n_tt)
    bar_width = 0.8 / n_tp

    cmap = cm.plasma
    fig, ax = plt.subplots(figsize=(max(9, n_tt * 0.9), 5))

    for ti, tp in enumerate(timepoints):
        diffs = []
        for r in results:
            def get_mean_at(traces, t):
                vals = [tr[t-1] for tr in traces if len(tr) >= t]
                return np.mean(vals) if vals else np.nan
            a = get_mean_at(r["_agent"], tp)
            z = get_mean_at(r["_zero"], tp)
            diffs.append(a - z)

        color = cmap(ti / max(n_tp - 1, 1))
        ax.bar(x + ti * bar_width - 0.4 + bar_width/2, diffs, bar_width,
               color=color, label=f"t={tp}", edgecolor="white", linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tt_vals], fontsize=7)
    ax.set_xlabel("Initial Tip/Tilt Error (arcsec)")
    ax.set_ylabel("Agent − Zero-Action (Mean Strehl)")
    ax.set_title("Agent Advantage at Key Timepoints")
    ax.legend(title="Timestep", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "advantage_bars")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis figures from sweep results")
    parser.add_argument("results_json", type=str,
                        help="Path to sweep_results.json")
    args = parser.parse_args()

    results = load_results(args.results_json)
    output_dir = os.path.dirname(os.path.abspath(args.results_json))
    print(f"Generating figures in: {output_dir}\n")

    fig_convergence_curves(results, output_dir)
    fig_convergence_heatmap(results, output_dir)
    fig_advantage_heatmap(results, output_dir)
    fig_steady_state(results, output_dir, tail_steps=64)
    fig_steady_state(results, output_dir, tail_steps=32)
    fig_time_to_threshold(results, output_dir, threshold=0.8)
    fig_time_to_threshold(results, output_dir, threshold=0.7)
    fig_trajectory_ribbons(results, output_dir)
    fig_success_rate(results, output_dir)
    fig_improvement_factor(results, output_dir)
    fig_initial_vs_final(results, output_dir)
    fig_time_above_curves(results, output_dir)
    fig_episode_total_strehl(results, output_dir)
    fig_percentile_envelope(results, output_dir)
    fig_advantage_bars(results, output_dir)

    print(f"\nDone. {15} figures generated in: {output_dir}")


if __name__ == "__main__":
    main()
