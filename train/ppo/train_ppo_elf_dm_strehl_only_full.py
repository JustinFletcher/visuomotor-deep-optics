"""Full-DM strehl-only sanity-check experiment.

Same control surface and aperture as ``train_ppo_elf_dm_strehl_only.py``
but with the bilateral wrapper DISABLED. The policy commands every
actuator of the 35x35 = 1225-element DM directly; no symmetry
constraint, no blind region, no dark hole, no curriculum.

Initial state is a small random DM perturbation (default 0.05 um per
actuator) -- only the DM is perturbed, segments and tip/tilt stay at
zero. The only reward is plain Strehl. This is the simplest possible
"can PPO close the loop on this hardware" test: 1225-dim action,
single-channel image obs, scalar Strehl reward.

If train/step_strehl rises monotonically toward 1.0 AND the
train/step_strehl_at_ep_step_NN trajectory shows actual within-episode
convergence (not just rollout-mean inflation), the architecture is
fine. Use this as the control for the bilateral-DM dark-hole gap.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole_bilateral_dm import (
    ENV_KWARGS as BILATERAL_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs
# ----------------------------------------------------------------------
ENV_KWARGS = dict(BILATERAL_ENV_KWARGS)

# No dark hole, no contrast term. The bilateral wrapper is NOT
# attached (see _patch below), so target_vec / blind_mask geometry is
# irrelevant -- the env just emits the raw focal-plane image.
ENV_KWARGS["dark_hole"] = False
ENV_KWARGS["reward_weight_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_log_mean_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_log_contrast"] = 0.0
ENV_KWARGS["reward_weight_log_contrast_strehl"] = 0.0
ENV_KWARGS["reward_weight_contrast_strehl"] = 0.0
ENV_KWARGS["reward_weight_centered_strehl"] = 0.0
ENV_KWARGS["reward_weight_centering"] = 0.0
ENV_KWARGS["reward_weight_flux"] = 0.0
ENV_KWARGS["reward_weight_peak"] = 0.0
ENV_KWARGS["reward_weight_shape"] = 0.0
ENV_KWARGS["reward_weight_image_quality"] = 0.0
ENV_KWARGS["reward_weight_strehl"] = 1.0
ENV_KWARGS["holding_bonus_weight"] = 0.0
ENV_KWARGS["action_penalty"] = False
ENV_KWARGS["action_penalty_weight"] = 0.0

# Small random DM perturbation per reset; NOT forced symmetric.
ENV_KWARGS["init_dm_micron_std"] = 0.05
ENV_KWARGS["init_dm_symmetric"] = False

# Absolute DM control. With incremental control + 1225-dim action the
# DM drifts off the strehl-good attractor over an episode (each step
# adds a small bias); absolute control lets every step write the full
# corrective shape directly so mean bias is a fixed offset, not a
# growing one.
ENV_KWARGS["dm_incremental_control"] = False
ENV_KWARGS["env_action_scale"] = 1.0


# ----------------------------------------------------------------------
# PPO config
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["target_dim"] = 0
    # Full DM -- no bilateral wrapper.
    cfg["bilateral_dm"] = False
    cfg.pop("bilateral_dm_mode", None)
    cfg.pop("bilateral_freeze_segments", None)
    # No curriculum on this run.
    cfg.pop("reward_weight_curriculum", None)
    cfg.pop("curriculum", None)
    cfg.pop("holding_bonus_anneal", None)
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--init-dm-micron-std", type=float, default=None,
        help="Override per-actuator std (microns) of the random init "
             "DM perturbation.")
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        if pre_args.init_dm_micron_std is not None:
            cfg["env_kwargs"]["init_dm_micron_std"] = float(
                pre_args.init_dm_micron_std)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
