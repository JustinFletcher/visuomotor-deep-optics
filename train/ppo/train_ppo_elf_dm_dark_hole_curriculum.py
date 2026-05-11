"""PPO training: dark-hole shaping with a Strehl + log-contrast-Strehl
curriculum.

Builds on ``train_ppo_elf_dm_strehl_only.py`` by introducing a
dark-hole-shaping signal gradually once the wavefront-control prior
has been learned. The reward chosen is

    lcs = (log10(max(I[~hole])) - log10(mean(I[hole]))) * S_clamped

with I taken from the raw pre-detector PSF and S clamped to [0, 1].
The multiplicative coupling to Strehl is essential: log-contrast
alone is a scale-invariant ratio and the policy can saturate it
(~12 decades) by emptying the frame and leaving any small bright
residual, without actually digging next to a bright peak. Multiplying
by S closes the exploit by construction -- degenerate states have
either low log-contrast OR low Strehl, so the product vanishes.
This is what the bare log_contrast reward (used in the first
revision of this script) was missing; the policy converged to a
"darken everything" attractor exactly because the bare-log-contrast
formulation rewards flux removal.

Operating-range smoke test (35x35 DM, inner-ring hole):

  state                     lc      lcs    cs     S
  perfect (DM=0)           1.57   1.46  -0.08  0.93
  mild noise (sigma=0.05)  1.44   0.65  -0.55  0.45
  moderate noise           1.00   0.05  -0.96  0.04
  severe scattering        0.79   0.00  -1.00  ~0
  y-tilt exploit (mild)    1.79   1.54  -0.23  0.86
  y-tilt exploit (heavy)   3.25   2.70  -0.43  0.83

At end-of-curriculum weight 0.2, baseline reward is
0.2 * 1.46 + (-0.08) = +0.21, the heaviest y-tilt exploit is
0.2 * 2.7 + (-0.43) = +0.11, severe scattering is -1.00, and a
real deep dig (lcs ~ 12) is +2.4. Baseline beats every degenerate
state; deep dig dominates by 2.2 reward units.

Schedule
--------
  * 0          -> warmup_timesteps (default 10M)
        Pure centred-Strehl reward. The strehl-only sanity check
        found the wavefront-control prior in ~1M steps, but with
        the larger HPC training budget we hold longer so the
        policy stabilises on a low-entropy strehl-only basin
        before the dark-hole signal is introduced.

  * warmup     -> warmup + anneal (default 100M)
        Linear ramp of reward_weight_log_contrast_strehl from 0 to
        0.2. centred-Strehl weight stays at 1.0 throughout. The
        100M-step ramp is 10x slower than the previous attempt that
        produced a catastrophic KL explosion at the ramp midpoint;
        slower ramping gives the value function time to track the
        moving target.

  * warmup + anneal onward
        Final mix is centred-Strehl 1.0 + log_contrast_strehl 0.2.

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
# log-contrast-Strehl weight starts at 0 and is ramped to 0.2 by the
# curriculum. End-state weighted lcs at baseline is 0.2 * 1.46 = 0.29,
# at deep dig 0.2 * 12 = 2.4, at any degenerate (low-Strehl) state
# ~0. centered_strehl in [-1, 0] then provides the collapse penalty.
ENV_KWARGS["reward_weight_centered_strehl"] = 1.0
ENV_KWARGS["reward_weight_log_contrast_strehl"] = 0.0
ENV_KWARGS["reward_weight_log_contrast"] = 0.0
ENV_KWARGS["reward_weight_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_log_mean_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_strehl"] = 0.0
ENV_KWARGS["holding_bonus_weight"] = 0.0


# ----------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------
# Hold log-contrast-Strehl weight at zero for 10M env steps (10x the
# strehl-only-prior learning time, so the policy stabilises on a low-
# entropy strehl-only basin before the dark-hole signal arrives),
# then ramp linearly over 100M steps to a final value of 0.2. The
# 100M-step ramp is 10x slower than the previous attempt that produced
# a catastrophic KL explosion at the ramp midpoint -- slower ramping
# gives the value function time to track the moving target.
LOG_CONTRAST_CURRICULUM = {
    "attr": "_rw_log_contrast_strehl",
    "warmup_timesteps": 10_000_000,
    "anneal_timesteps": 100_000_000,
    "start_value": 0.0,
    "end_value": 0.2,
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
    cfg["reward_weight_curriculum"] = LOG_CONTRAST_CURRICULUM
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
        cfg["reward_weight_curriculum"] = dict(cfg["reward_weight_curriculum"])
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
            cfg["reward_weight_curriculum"]["warmup_timesteps"] = int(
                pre_args.curriculum_warmup)
        if pre_args.curriculum_anneal is not None:
            cfg["reward_weight_curriculum"]["anneal_timesteps"] = int(
                pre_args.curriculum_anneal)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
