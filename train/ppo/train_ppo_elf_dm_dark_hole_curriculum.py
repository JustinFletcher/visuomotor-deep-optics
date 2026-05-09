"""PPO training: dark-hole shaping with a Strehl pre-training curriculum.

Builds on ``train_ppo_elf_dm_strehl_only.py`` (which established that
the bilateral-DM policy can do closed-loop wavefront control under a
centred-Strehl reward) by introducing a dark-hole reward gradually
once the wavefront-control prior has been learned. The intent is to
avoid the degenerate "darken the entire frame" attractor that plagued
the cold-start dark-hole training.

Schedule
--------
  * 0          -> warmup_timesteps (default 1M)
        Pure centred-Strehl reward. Same regime as the strehl-only
        sanity check. The policy learns wavefront control under the
        small symmetric init perturbation.

  * warmup     -> warmup + anneal (default 10M after warmup)
        Linear ramp of reward_weight_dark_hole from 0 to 1.0.
        centred-Strehl weight stays at 1.0 throughout, so the
        wavefront-control signal is preserved while the dark-hole
        objective is introduced gradually. The policy is forced to
        find DM shapes that do BOTH (Strehl-positive AND dark-hole-
        deep) rather than a darken-everything shortcut.

  * warmup + anneal onward
        reward_weight_dark_hole = 1.0, reward_weight_centered_strehl
        = 1.0. Equal weighting. Training proceeds against the joint
        objective for the remainder of the budget.

Env / wrapper layout matches the bilateral-DM-fixed track: absolute
DM control, fixed_vertical bilateral mode, segments frozen at zero,
50 nm symmetric init perturbation. Per-target geometry is supplied
via CLI flags by the launcher.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_strehl_only import (
    ENV_KWARGS as STREHL_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs
# ----------------------------------------------------------------------
ENV_KWARGS = dict(STREHL_ENV_KWARGS)

# Re-enable the dark-hole geometry. The reward weight starts at 0 and
# is ramped up by the curriculum below; the geometry has to be live
# from step 0 so the bilateral wrapper's blind mask is computed
# correctly (otherwise the wrapper sees target_vec == 0 and produces
# a one-pixel mask, which would be wrong post-curriculum).
ENV_KWARGS["dark_hole"] = True
ENV_KWARGS["dark_hole_angular_location_degrees"] = 0.0    # set by CLI
ENV_KWARGS["dark_hole_location_radius_fraction"] = 0.16
ENV_KWARGS["dark_hole_size_radius"] = 0.095

# Reward composition. centered_strehl stays at 1.0 throughout; the
# dark-hole weight starts at 0 and is ramped to 1.0 by the curriculum.
ENV_KWARGS["reward_weight_centered_strehl"] = 1.0
ENV_KWARGS["reward_weight_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_log_mean_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_strehl"] = 0.0
ENV_KWARGS["holding_bonus_weight"] = 0.0


# ----------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------
# Strehl-only sanity check showed the policy learns wavefront control
# in roughly 1M env steps. Hold dark-hole weight at zero for that
# window, then ramp linearly over 10M steps. Past the ramp the policy
# trains on the joint reward for the rest of the budget.
DARK_HOLE_CURRICULUM = {
    "warmup_timesteps": 1_000_000,
    "anneal_timesteps": 10_000_000,
    "start_value": 0.0,
    "end_value": 1.0,
}


# ----------------------------------------------------------------------
# PPO config patch
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["target_dim"] = 4                       # target-aware policy
    cfg["bilateral_dm"] = True
    cfg["bilateral_dm_mode"] = "fixed_vertical"
    cfg["bilateral_freeze_segments"] = True
    cfg["dark_hole_curriculum"] = DARK_HOLE_CURRICULUM
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
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
        "--curriculum-warmup", type=int, default=None,
        help="Override warmup_timesteps for the dark-hole curriculum.")
    pre_parser.add_argument(
        "--curriculum-anneal", type=int, default=None,
        help="Override anneal_timesteps for the dark-hole curriculum.")
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        cfg["dark_hole_curriculum"] = dict(cfg["dark_hole_curriculum"])
        if pre_args.dark_hole_angle is not None:
            cfg["env_kwargs"]["dark_hole_angular_location_degrees"] = float(
                pre_args.dark_hole_angle)
        if pre_args.dark_hole_radius_frac is not None:
            cfg["env_kwargs"]["dark_hole_location_radius_fraction"] = float(
                pre_args.dark_hole_radius_frac)
        if pre_args.dark_hole_size is not None:
            cfg["env_kwargs"]["dark_hole_size_radius"] = float(
                pre_args.dark_hole_size)
        if pre_args.curriculum_warmup is not None:
            cfg["dark_hole_curriculum"]["warmup_timesteps"] = int(
                pre_args.curriculum_warmup)
        if pre_args.curriculum_anneal is not None:
            cfg["dark_hole_curriculum"]["anneal_timesteps"] = int(
                pre_args.curriculum_anneal)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
