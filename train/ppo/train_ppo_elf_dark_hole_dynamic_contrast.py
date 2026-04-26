"""
PPO training: ELF dynamic dark-hole, log-mean contrast reward.

Twin of ``train_ppo_elf_dark_hole_dynamic.py`` with the linear
``-mean(I[hole]/I_max)`` term swapped out for the log-mean depth
formulation:

    depth_per_pixel = log10(I_max) - log10(max(I_pixel, floor))
    reward          = mean(depth_per_pixel)  over hole pixels

Higher depth = deeper hole = larger reward. Computed on the raw
pre-detector PSF for full intensity precision (the detector
quantization would floor anything below ~1 DN, hiding most of the
contrast regime we care about).

Same dynamic target-resampling envelope as the dynamic script, same
target-aware policy architecture (target_dim=4), same rail-baseline
randomisation. Only the reward weights differ.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole import (
    ELF_DARK_HOLE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# Mirror the dynamic env kwargs and swap the reward.
DYNAMIC_CONTRAST_ENV_KWARGS = dict(ELF_DARK_HOLE_ENV_KWARGS)
DYNAMIC_CONTRAST_ENV_KWARGS.update({
    "dark_hole": True,
    "dark_hole_randomize_on_reset": True,
    "dark_hole_angle_range_deg": (0.0, 360.0),
    "dark_hole_radius_range": (0.16, 0.32),
    "dark_hole_size_range": (0.08, 0.08),

    # Reward swap: turn off the linear mean-intensity dark-hole term,
    # turn on the log-mean depth term. Weight 0.1 keeps per-step values
    # in roughly the same magnitude band as the other terms (depth in
    # decades typically lives in [0, 8]).
    "reward_weight_dark_hole": 0.0,
    "reward_weight_log_mean_dark_hole": 0.1,

    # Holding bonus disabled because the log-mean signal is unbounded
    # above and the holding-bonus quality formula assumes the natural
    # reward sits in [-1, 0]. Easier to disable than to re-tune.
    "holding_bonus_weight": 0.0,
})


LOCAL_CONFIG = dict(BASE_LOCAL_CONFIG)
LOCAL_CONFIG["env_kwargs"] = DYNAMIC_CONTRAST_ENV_KWARGS
LOCAL_CONFIG["target_dim"] = 4

HPC_CONFIG = dict(BASE_HPC_CONFIG)
HPC_CONFIG["env_kwargs"] = DYNAMIC_CONTRAST_ENV_KWARGS
HPC_CONFIG["target_dim"] = 4


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--seed", type=int, default=None,
                            help="Override the PPO seed (default: from "
                                 "config = 1).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
