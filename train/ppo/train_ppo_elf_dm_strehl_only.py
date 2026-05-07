"""Sanity-check experiment: can the bilateral-DM policy correct a
small bilaterally-symmetric DM error using Strehl alone?

This script intentionally drops everything dark-hole. The episode
starts with a small random DM perturbation that is exactly
bilaterally symmetric across the vertical axis (init_dm_symmetric
under v5's fixed-axis mirror partner map), and the only reward term
is centre-weighted Strehl. The bilateral wrapper is enabled in
fixed_vertical mode, so the policy can only emit bilaterally-
symmetric commands across the same axis the perturbation lives on
-- the perturbation is in principle perfectly correctable by the
policy's action subspace.

What this experiment tells us:

  * If centred Strehl rises monotonically toward 1.0, the policy
    can do 612-dim closed-loop wavefront control on this hardware
    at all -- a positive control on the architecture before we
    confront it with the dark-hole task again.

  * If it stalls or never improves, the failure is in the policy
    or PPO setup, not in the dark-hole reward shaping. That tells
    us where to dig next.

If this works, the planned follow-up is a curriculum: keep the
policy on Strehl until it converges, then ramp in dark-hole reward
weight (or contrast as a multiplicative factor) so the policy
inherits a good visuomotor prior before the dark-hole signal
introduces a degenerate "darken everything" attractor.
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

# No dark hole at all. The bilateral wrapper still constructs a blind
# mask each step but with target_vec all zeros it collapses to a
# 1-pixel no-op at the focal-plane centre, which is harmless for
# centred-Strehl reward.
ENV_KWARGS["dark_hole"] = False
ENV_KWARGS["reward_weight_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_log_mean_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_strehl"] = 0.0
ENV_KWARGS["reward_weight_centered_strehl"] = 1.0
ENV_KWARGS["holding_bonus_weight"] = 0.0

# Initial DM perturbation, bilaterally symmetric across the vertical
# axis using v5's exact actuator partner map. 50 nm per actuator is
# small (lambda / 20 at 1 micron) but produces a measurable Strehl
# drop when summed across the 1225-actuator influence basis -- enough
# headroom for the policy to demonstrate corrective behaviour without
# saturating against the stroke envelope.
ENV_KWARGS["init_dm_micron_std"] = 0.05
ENV_KWARGS["init_dm_symmetric"] = True


# ----------------------------------------------------------------------
# PPO config
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["target_dim"] = 0                       # no target geometry input
    cfg["bilateral_dm"] = True
    cfg["bilateral_dm_mode"] = "fixed_vertical"
    cfg["bilateral_freeze_segments"] = True
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
        help="Override per-actuator std (microns) of the symmetric "
             "init DM perturbation.")
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
