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
# Strong L1 action penalty. Anchors the policy mean hard toward zero.
# At weight 1.0 the per-step penalty equals mean(|a|), so a real
# corrective action of ~0.05 per dim costs 0.05 reward -- comparable
# to (and partly trading against) the strehl improvement that
# correction would buy. Random-walk magnitudes of ~0.01 cost 0.01,
# enough to push the gradient toward zero whenever the policy isn't
# extracting clear strehl value. Earlier weight=0.01 was too weak --
# run dm_strehl_only_full_1779085555 still saw the policy mean drift
# to within-episode strehl collapse despite the tight sigma and the
# penalty.
ENV_KWARGS["action_penalty"] = True
ENV_KWARGS["action_penalty_weight"] = 1.0

# Small random DM perturbation per reset; NOT forced symmetric.
ENV_KWARGS["init_dm_micron_std"] = 0.05
ENV_KWARGS["init_dm_symmetric"] = False

# DM only. Segments and tip/tilt are NOT in the action space. The base
# bilateral config sets command_secondaries=True (the wrapper freezes
# those seg-piston dims at zero, so they don't matter there); with the
# wrapper OFF here we'd otherwise inherit 15 incremental seg-piston
# DOFs that accumulate ~5 µm of drift per episode under any sigma > 0,
# completely scrambling the on-axis flux even when the DM half works.
ENV_KWARGS["command_secondaries"] = False
ENV_KWARGS["command_tip_tilt"] = False

# Incremental DM control, matching the bilateral-DM recipe that does
# converge. Per-step DM change is bounded to env_action_scale *
# stroke_limit = 0.1 * 1.5 µm = 0.15 µm, so a single off-direction
# action only nudges the OPD instead of scrambling it. Absolute control
# was tried first and diverged in every seed (run dm_strehl_only_full
# _1778958135): the corrective signal is ~0.03-0.10 per actuator while
# the action bound is ±1.0, so the policy's exploration noise (sigma
# 0.082 at init) dominates the optimal mean (~0.033 for 1-σ init draw)
# and the LSTM-driven mean walks to saturation with no gradient to
# escape (strehl floors at zero in every direction at full deflection).
ENV_KWARGS["dm_incremental_control"] = True
# env_action_scale tightened from 0.1 -> 0.01. Max per-step DM change
# becomes 0.01 * 1.5 µm = 0.015 µm, comparable to (but below) the
# 1-sigma init perturbation (0.05 µm). The corrective signal needs
# 0.05/0.015 ~ 3 steps to fully cancel a 1-sigma init draw, easily
# inside the 64-step episode. Accumulated noise at sigma=0.018 is
# 0.018 * 0.01 * 1.5 * sqrt(64) = 0.0022 µm RMS -- effectively zero
# relative to init. Combined with action_penalty_weight=1.0 this should
# make 'do nothing' a stable attractor that the policy can incrementally
# improve from.
ENV_KWARGS["env_action_scale"] = 0.01


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
    # --- Entropy / log_std: ABSOLUTE VANILLA ----------------------------
    # Strip every entropy-related modification accumulated in the
    # bilateral patch chain. The result is stock PPO defaults:
    #   - ent_coef = 0.0          (no entropy bonus, CleanRL default)
    #   - init_log_std = -0.5     (RecurrentActorCritic class default,
    #                              sigma = exp(-0.5) ~= 0.61)
    #   - log_std_min = None      (no clamping)
    #   - log_std_max = None      (no clamping)
    # The bilateral patch had set ent_coef=1e-6, init_log_std=-2.5,
    # log_std_max=-2.0, log_std_min=-5.0; all of those are removed here.
    # If the run still misbehaves at these defaults, the failure is the
    # task / architecture itself, not an accumulated hack.
    cfg["ent_coef"] = 0.0
    cfg["init_log_std"] = -0.5
    cfg.pop("log_std_min", None)
    cfg.pop("log_std_max", None)
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
