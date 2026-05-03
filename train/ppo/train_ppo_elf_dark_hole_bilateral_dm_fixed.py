"""PPO training: bilaterally-symmetric DM dark-hole shaping, FIXED axis.

Companion to ``train_ppo_elf_dark_hole_bilateral_dm.py``. Same env,
same reward, same DM, same hyperparameters -- the ONLY difference is
the bilateral wrapper's symmetry mode:

  * ``per_target_radial`` (the other script): per episode, the
    symmetry axis rotates so it sits perpendicular to the target's
    radial direction. The actuator partition is rebuilt per target,
    and for off-cardinal angles the mirror partners are nearest-
    neighbour matches on the Cartesian DM grid (approximate symmetry).

  * ``fixed_vertical`` (this script): the symmetry axis is the
    vertical line x=0 for every target. The 35x35 actuator grid is
    exactly bijective under (x, y) -> (-x, y), so the resulting DM
    command is bit-exact bilaterally symmetric. Blind region is the
    horizontal mirror of the target (same y, opposite x in the focal
    plane), not the diametric opposite.

Use this variant when strict bilateral symmetry of the action is more
important than aligning the symmetry axis with the target's radial
direction.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole_bilateral_dm import (
    ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


def _patch_fixed(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["bilateral_dm_mode"] = "fixed_vertical"
    return cfg


LOCAL_CONFIG = _patch_fixed(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch_fixed(BASE_HPC_CONFIG)


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dark-hole-angle", type=float, default=None)
    pre_parser.add_argument("--dark-hole-radius-frac", type=float, default=None)
    pre_parser.add_argument("--dark-hole-size", type=float, default=None)
    pre_parser.add_argument("--seed", type=int, default=None)
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
