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
# Small L1 action penalty. Anchors the policy mean toward zero, which
# is essential under incremental control: without it PPO's gradient on
# 1225 dims is dominated by per-step reward noise and the mean random-
# walks away from zero, compounding linearly into the DM over an
# episode (run dm_strehl_only_full_1779084109: eval/mean_final_strehl
# crashed from 0.72 -> 0.001 even though the policy starts with
# mean=0). 0.01 is tiny vs the strehl signal -- at a real correction
# magnitude of 0.3 per dim the penalty is 0.003, negligible vs a
# strehl improvement worth ~0.3 reward; at random-walk magnitudes
# (~0.01) the penalty is ~1e-4, just enough to anchor the mean.
ENV_KWARGS["action_penalty"] = True
ENV_KWARGS["action_penalty_weight"] = 0.01

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
ENV_KWARGS["env_action_scale"] = 0.1


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
    # --- Entropy / log_std envelope -------------------------------------
    # Walked back toward vanilla relative to the bilateral patch:
    #   - ent_coef removed (was 1e-6 "floor"; with log_std_min in place
    #     there's no underflow risk and the floor served no purpose --
    #     1e-6 * H_total of -1324 contributes 0.0013 to the loss, which
    #     is rounding noise next to a strehl signal of ~-0.3 per step).
    #   - init_log_std dropped from -2.5 to -4.0 (sigma 0.082 -> 0.018).
    #     End-of-episode accumulated DM noise under incremental control
    #     drops from ~0.10 um RMS to ~0.02 um RMS, comfortably below
    #     init_dm_micron_std=0.05. Stops the within-episode strehl
    #     crash observed in run dm_strehl_only_full_1779084109 (strehl
    #     went 0.70 at step 1 -> 0.001 at step 64).
    #   - log_std_max tightened from -2.0 to -3.0 (sigma cap 0.135 ->
    #     0.050). Cap matches the corrective-action range (~0.10) so
    #     exploration never overwhelms signal.
    #   - log_std_min kept at -5.0 as a numerical safety floor.
    cfg["ent_coef"] = 0.0
    cfg["init_log_std"] = -4.0
    cfg["log_std_max"] = -3.0
    cfg["log_std_min"] = -5.0
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
