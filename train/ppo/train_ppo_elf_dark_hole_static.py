"""
PPO training: ELF piston-only dark-hole shaping, STATIC target-aware.

Same env/reward setup as ``train_ppo_elf_dark_hole.py`` but the policy
is target-aware: its LSTM input includes an auxiliary 4-vector
[sin(θ), cos(θ), radius_frac, size_frac] that encodes the dark-hole
geometry. The target stays fixed for the whole run (same hole in every
episode), matching the current single-geometry-per-run workflow, but
with an architecture that can be dropped into a composite pipeline
alongside the dynamic variant without code change.

For a dynamically varying target (random hole per episode), use
``train_ppo_elf_dark_hole_dynamic.py`` instead.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole import (
    ELF_DARK_HOLE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


LOCAL_CONFIG = dict(BASE_LOCAL_CONFIG)
LOCAL_CONFIG["env_kwargs"] = dict(ELF_DARK_HOLE_ENV_KWARGS)
LOCAL_CONFIG["target_dim"] = 4
# Entropy regulation: with norm_adv=True the PG gradient on log_std
# is roughly O(1) but noisy, while ent_coef contributes a constant
# +0.005 per dim per update — when advantages don't reliably
# distinguish actions (this task), the entropy term wins on average
# and log_std drifts up monotonically until the policy is uniformly
# random. Drop ent_coef ~3x and clamp log_std at -1 (sigma <= 0.37,
# plenty of exploration room for piston shaping but not enough to
# sample uniformly across the [-1, 1] action range).
LOCAL_CONFIG["ent_coef"] = 0.0015
LOCAL_CONFIG["log_std_max"] = -1.0

HPC_CONFIG = dict(BASE_HPC_CONFIG)
HPC_CONFIG["env_kwargs"] = dict(ELF_DARK_HOLE_ENV_KWARGS)
HPC_CONFIG["target_dim"] = 4
HPC_CONFIG["ent_coef"] = 0.0015
HPC_CONFIG["log_std_max"] = -1.0


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--dark-hole-angle", type=float, default=None,
        help="Dark-hole angular location, degrees [0, 360).")
    pre_parser.add_argument(
        "--dark-hole-radius-frac", type=float, default=None,
        help="Dark-hole radial location as fraction of FOV.")
    pre_parser.add_argument(
        "--dark-hole-size", type=float, default=None,
        help="Dark-hole size (radius), same units as radius-frac.")
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config = 1).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        if pre_args.dark_hole_angle is not None:
            cfg["env_kwargs"]["dark_hole_angular_location_degrees"] = float(
                pre_args.dark_hole_angle)
        if pre_args.dark_hole_radius_frac is not None:
            cfg["env_kwargs"]["dark_hole_location_radius_fraction"] = float(
                pre_args.dark_hole_radius_frac)
        if pre_args.dark_hole_size is not None:
            cfg["env_kwargs"]["dark_hole_size_radius"] = float(
                pre_args.dark_hole_size)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
