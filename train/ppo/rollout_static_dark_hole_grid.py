#!/usr/bin/env python
"""Roll out the static-sweep checkpoints, one policy per grid target.

Companion to ``rollout_dark_hole_grid.py``. Where that script evaluates
a *single* (dynamic-trained) target-aware policy against all 16 grid
targets, this script picks the *matching* static policy for each
target — i.e. it loads
``<sweep_dir>/target_NN/ppo_optomech_*/checkpoints/best.pt`` and runs
that checkpoint against target NN's geometry.

Same layout and per-step diagnostics (OPD | raw PSF | observation +
contrast trace) as the dynamic rollout, so side-by-side comparison
with the dynamic GIFs is straightforward.

Usage:
    # Roll all 16 static checkpoints against their own targets:
    python train/ppo/rollout_static_dark_hole_grid.py \\
        --sweep-dir dark_hole_runs/dark_hole_static_<ts> \\
        --output-dir test_output/dh_static_<ts>_grid/

    # Just one target id (debug):
    python train/ppo/rollout_static_dark_hole_grid.py \\
        --sweep-dir dark_hole_runs/dark_hole_static_<ts> \\
        --output-dir test_output/dh_static_<ts>_one/ \\
        --target-id 7
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.ppo.launch_dark_hole_grid import build_grid                 # noqa: E402
from train.ppo.rollout_dark_hole_grid import (                         # noqa: E402
    _build_env,
    _load_agent,
    render_gif,
    run_episode,
)


_DEFAULT_SWEEP_ROOT = "dark_hole_runs"
_SWEEP_PREFIX = "dark_hole_static_"


def _latest_sweep_dir(root: str = _DEFAULT_SWEEP_ROOT) -> str | None:
    """Return the newest ``<root>/dark_hole_static_*`` dir by mtime."""
    if not os.path.isdir(root):
        return None
    candidates = [
        os.path.join(root, name) for name in os.listdir(root)
        if name.startswith(_SWEEP_PREFIX)
        and os.path.isdir(os.path.join(root, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _resolve_checkpoint(sweep_dir: str, target_idx: int,
                        prefer_latest: bool = False) -> str:
    """Find the best.pt under <sweep_dir>/target_<NN>/ppo_optomech_*/.

    Each static sweep job lives in target_NN/ with a single
    ppo_optomech_<seed>_<ts>/ subdir holding the checkpoints/.

    prefer_latest: if True, use the newest update_*.pt (or history_*)
    instead of best.pt. Falls back to best.pt if no numbered
    checkpoint is found.
    """
    target_dir = os.path.join(sweep_dir, f"target_{target_idx:02d}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"target dir missing: {target_dir} "
            f"(expected layout: <sweep_dir>/target_NN/ppo_optomech_*/checkpoints/)")

    if prefer_latest:
        ck_glob = os.path.join(
            target_dir, "ppo_optomech_*", "checkpoints", "*update_*.pt")
        latest = sorted(glob(ck_glob), key=os.path.getmtime)
        if latest:
            return latest[-1]
        # fall through to best.pt below

    candidates = sorted(glob(os.path.join(
        target_dir, "ppo_optomech_*", "checkpoints", "best.pt")))
    if not candidates:
        candidates = sorted(glob(os.path.join(
            target_dir, "ppo_optomech_*", "checkpoints", "*update_*.pt")))
        if not candidates:
            raise FileNotFoundError(
                f"no checkpoints under {target_dir}/ppo_optomech_*/checkpoints/")
        return sorted(candidates, key=os.path.getmtime)[-1]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Roll the static dark-hole sweep, one policy per target.")
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Path to the static-sweep run dir, e.g. "
             "dark_hole_runs/dark_hole_static_<timestamp>. If omitted, "
             "the newest dark_hole_runs/dark_hole_static_* dir is used.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Destination for target_NN.gif files. Default: "
             "test_output/<sweep_basename>_grid/.")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--target-id", type=int, default=None,
        help="Limit to one target id (0..15) for quick iteration.")
    parser.add_argument("--frame-duration", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--prefer-latest", action="store_true",
        help="Use the newest update_*.pt instead of best.pt for "
             "checkpoint selection (per target). Falls back to "
             "best.pt if no numbered checkpoint exists.")
    args = parser.parse_args()

    targets = build_grid()
    if args.target_id is not None:
        if not (0 <= args.target_id < len(targets)):
            print(f"Error: --target-id must be in [0, {len(targets) - 1}]")
            sys.exit(1)
        target_indices = [args.target_id]
    else:
        target_indices = list(range(len(targets)))

    sweep_dir = args.sweep_dir
    if sweep_dir is None:
        sweep_dir = _latest_sweep_dir()
        if sweep_dir is None:
            print(f"Error: no {_SWEEP_PREFIX}* dir found under "
                  f"{_DEFAULT_SWEEP_ROOT}/. Pass --sweep-dir explicitly.")
            sys.exit(1)
        print(f"--sweep-dir not given; using newest: {sweep_dir}")
    if not os.path.isdir(sweep_dir):
        print(f"Error: --sweep-dir {sweep_dir} does not exist")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "test_output", f"{os.path.basename(sweep_dir.rstrip('/'))}_grid")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Sweep dir:  {sweep_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Targets:    {target_indices}")
    print()

    summary = []
    for i in target_indices:
        target = targets[i]
        try:
            ckpt_path = _resolve_checkpoint(
                sweep_dir, i, prefer_latest=args.prefer_latest)
        except FileNotFoundError as e:
            print(f"  target {i:>2}: SKIP — {e}")
            continue

        env = _build_env(target, args.max_steps)
        agent, config = _load_agent(ckpt_path, env, args.device)
        td = int(config.get("target_dim", 0))
        if td == 0:
            # Static checkpoints from the target-aware grid should have
            # target_dim=4. Older checkpoints would be target-blind;
            # warn loudly.
            print(f"  target {i:>2}: WARNING — checkpoint target_dim=0 "
                  f"(checkpoint: {ckpt_path})")

        ep = run_episode(agent, env, target, args.seed, args.device)
        ep["target_id"] = i
        gif_path = os.path.join(output_dir, f"target_{i:02d}.gif")
        render_gif(ep, gif_path, dpi=args.dpi,
                   frame_duration=args.frame_duration)
        env.close()

        final_s = ep["strehls"][-1] if ep["strehls"] else float("nan")
        final_ct = ep["contrast"][-1] if ep["contrast"] else float("nan")
        best_ct = (
            min(c for c in ep["contrast"] if np.isfinite(c) and c > 0)
            if any(np.isfinite(c) and c > 0 for c in ep["contrast"])
            else float("nan"))
        summary.append((i, target, ep["return"], final_s, final_ct, best_ct))
        print(f"  target {i:>2}: angle={target[0]:6.1f}  "
              f"r={target[1]:.3f}  size={target[2]:.3f}  "
              f"R={ep['return']:+.3f}  final_S={final_s:.4f}  "
              f"final_C={final_ct:.2e}  best_C={best_ct:.2e}  "
              f"-> {gif_path}")

    print(f"\nWrote {len(summary)} GIFs to {output_dir}")


if __name__ == "__main__":
    main()
