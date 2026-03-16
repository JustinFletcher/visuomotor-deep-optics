"""
Lightweight PPO nanoelf training test.

Runs a fast smoke test by importing training infrastructure from
train/ppo/train_ppo_optomech.py and verifying the trained policy
outperforms baselines.  Uses the piston-only config by default.

Usage:
    poetry run python tests/test_ppo_nanoelf.py                          # full run (v4)
    poetry run python tests/test_ppo_nanoelf.py --fast                   # quick smoke test
    poetry run python tests/test_ppo_nanoelf.py --env-version v3         # use optomech-v3
    poetry run python tests/test_ppo_nanoelf.py --env-version v3 --fast  # v3 smoke test
    poetry run python tests/test_ppo_nanoelf.py --run-dir ./output       # specify output dir
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train.ppo.train_ppo_optomech as train_module
from train.ppo.train_ppo_optomech import (
    register_optomech,
    run_ppo_training,
    evaluate_zero_policy,
    evaluate_random_policy,
)
from train.ppo.train_ppo_nanoelf_piston import (
    FULL_CONFIG,
    FAST_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(description="PPO Nanoelf Optomech Test")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke test with fewer timesteps (no pass/fail threshold)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for outputs (default: temp dir)",
    )
    parser.add_argument(
        "--env-version",
        type=str,
        default="v4",
        choices=["v1", "v2", "v3", "v4"],
        help="Optomech environment version (default: v4)",
    )
    parser.add_argument(
        "--action-penalty-weight",
        type=float,
        default=None,
        help="L1 action penalty weight (overrides env config)",
    )
    cli = parser.parse_args()

    # Set the module-level env ID in the training module
    train_module._ENV_ID = f"optomech-{cli.env_version}"

    config = dict(FAST_CONFIG if cli.fast else FULL_CONFIG)

    # Override action penalty weight if specified on command line
    if cli.action_penalty_weight is not None:
        config["env_kwargs"] = dict(config["env_kwargs"])
        config["env_kwargs"]["action_penalty_weight"] = cli.action_penalty_weight

    # Disable periodic checkpointing in test mode
    config["model_save_interval"] = 0

    # Generate fixed eval seeds deterministically from the main seed.
    rng = np.random.RandomState(config["seed"])
    config["eval_seeds"] = rng.randint(0, 2**31, size=config["eval_episodes"]).tolist()

    print(f"Using environment: {train_module._ENV_ID}")
    print(f"Fixed eval seeds:  {config['eval_seeds']}")

    # Register environment
    register_optomech(train_module._ENV_ID, max_episode_steps=config["max_episode_steps"])

    # Output directory
    if cli.run_dir:
        run_dir = cli.run_dir
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = tempfile.mkdtemp(prefix="ppo_nanoelf_test_")

    print(f"Output directory: {run_dir}")

    # Baselines
    print("\nEvaluating zero-action baseline...")
    zero_return = evaluate_zero_policy(config, num_episodes=3)
    print(f"Zero-action policy mean return: {zero_return:.4f}")

    print("Evaluating random baseline...")
    random_return = evaluate_random_policy(config, num_episodes=3)
    print(f"Random policy mean return: {random_return:.4f}")

    # Train
    best_eval_return, this_run_dir = run_ppo_training(config, run_dir)

    # Pass / Fail
    threshold = config.get("pass_threshold_ratio")
    if threshold is not None:
        # For nanoelf, higher return = better alignment.
        # The trained policy should beat the better of (zero, random) baselines.
        baseline = max(zero_return, random_return)
        passed = best_eval_return > baseline * threshold

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Zero-action baseline:  {zero_return:.4f}")
        print(f"  Random baseline:       {random_return:.4f}")
        print(f"  Best trained policy:   {best_eval_return:.4f}")
        print(f"  Threshold (ratio):     {threshold}")
        print(f"  Required (baseline x {threshold}): {baseline * threshold:.4f}")

        if passed:
            print(
                f"\n  \u2713 TEST PASSED \u2014 trained policy outperforms baselines"
            )
        else:
            print(
                f"\n  \u2717 TEST FAILED \u2014 trained policy did not sufficiently outperform baselines"
            )
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {this_run_dir}")

        sys.exit(0 if passed else 1)
    else:
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST COMPLETE (no pass/fail threshold in fast mode)")
        print(f"  Best eval return:      {best_eval_return:.4f}")
        print(f"  Zero-action baseline:  {zero_return:.4f}")
        print(f"  Random baseline:       {random_return:.4f}")
        print(f"{'='*60}")
        print(f"\nTensorBoard: tensorboard --logdir {this_run_dir}")


if __name__ == "__main__":
    main()
