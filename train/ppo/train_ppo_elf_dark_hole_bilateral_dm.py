"""PPO training: bilaterally-symmetric DM dark-hole shaping.

Differences from ``train_ppo_elf_dark_hole_static.py``:

  * Control surface is the deformable mirror, not segment piston. The
    35x35 = 1225-actuator DM is enabled via ``command_dm=True``; segment
    piston/tip/tilt are held at zero (the wrapper writes a zero seg
    slice on every step). The segment-piston handoff is treated as a
    separate stage and not modelled here.

  * The ``BilateralDMVectorEnv`` wrapper enforces strict bilateral
    symmetry of the DM command around the line through the origin
    perpendicular to the target's radial direction. The policy outputs
    only the n_dm // 2 = 612 actuators on the target side; the wrapper
    fills in the mirror half and pins on-axis actuators to zero. No
    additional symmetry loss or regularizer: the action space itself
    is reduced.

  * The wrapper also masks the focal-plane region exactly opposite the
    target dark hole (the bilateral mirror of the target). The policy
    never sees that region. At test time, light in the unmasked blind
    region is the honest verification signal -- the policy could not
    have been rewarded for what happens there during training.

  * Bandwidth dropped from 200 nm (20%) to 100 nm (10%) to bring the
    polychromatic contrast floor closer to deep coronagraph regimes
    (e.g. Roman CGI uses 10% bandpasses). With Delta_lambda / lambda
    halved, the chromatic floor on contrast scales by 1/4.

  * Reward is 1.0 * dark_hole + 0.25 * strehl. The 0.25 weight on
    Strehl is chosen so the two terms contribute roughly equally near
    the operating point: the target dark hole region's mean intensity
    fraction (the dark_hole reward, in [-1, 0]) and the on-axis Strehl
    (in [0, 1]) sit at order 0.1-1.0 magnitudes. Holding bonus is
    disabled here -- the dual-objective reward already shapes both the
    null and the on-axis peak directly.

  * DM repeatability noise (``actuator_noise``) is OFF for the first
    pass at this task. We can re-enable it when we want to study
    realism-vs-attainability tradeoffs.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole import (
    ELF_DARK_HOLE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs
# ----------------------------------------------------------------------
ENV_KWARGS = dict(ELF_DARK_HOLE_ENV_KWARGS)

# 10% bandpass at 1 micron.
ENV_KWARGS["bandwidth_nanometers"] = 100.0
# Keep the same number of wavelength samples; with a narrower band the
# samples cluster more tightly around line centre, so the chromatic
# floor measurement stays well-resolved.
ENV_KWARGS["bandwidth_sampling"] = 2

# Control surface: DM only. Segments still need to be constructed (v5
# requires command_secondaries=True), but the wrapper writes zero seg
# actions on every step so they stay pinned at the baseline (zero).
ENV_KWARGS["command_dm"] = True
ENV_KWARGS["command_secondaries"] = True
ENV_KWARGS["command_tip_tilt"] = False
ENV_KWARGS["dm_model_type"] = "gaussian_influence"
ENV_KWARGS["dm_num_actuators_across"] = 35           # 1225 actuators
# Skip v4's per-actuator interaction-matrix calibration. The matrix
# only feeds v4's AO closed-loop reconstructor, which we don't run --
# the policy drives the DM directly. On 1225 actuators the calibration
# takes many minutes (per-actuator HCIPy propagation sweep). Inert in
# v5 because v5 already forces this flag on its internal v4 build,
# but set here too in case the script is run against v4 directly.
ENV_KWARGS["dm_skip_calibration"] = True

# DM uses incremental control: each step's action is a delta from the
# current DM state, accumulated and clipped to +/- stroke_limit_m. This
# matches the segment-piston control mode and keeps each step's per-
# actuator OPD change to env_action_scale * stroke_limit_m. Segments
# still use incremental control but they're frozen at zero anyway, so
# the seg-side flag is irrelevant for this run.
ENV_KWARGS["dm_incremental_control"] = True
ENV_KWARGS["incremental_control"] = True
# Per-step DM delta cap: 10% of full stroke per step. With max_episode_steps
# = 64, the policy can sweep from one rail to the other in ~10 steps and
# still has plenty of fine-grained adjustment headroom near the operating
# point.
ENV_KWARGS["env_action_scale"] = 0.1

# No actuator noise on the first pass.
ENV_KWARGS["actuator_noise"] = False
ENV_KWARGS["actuator_noise_fraction"] = 0.0

# Initial state: every DOF starts exactly at zero. No rail randomisation
# (the segments don't move, so there's no pegging to defeat). Init wind
# perturbations also off.
ENV_KWARGS["init_differential_motion"] = False
ENV_KWARGS["init_differential_motion_configurable"] = True
ENV_KWARGS["init_piston_micron_mean"] = 0.0
ENV_KWARGS["init_piston_micron_std"] = 0.0
ENV_KWARGS["init_piston_clip_micron"] = 0.0
ENV_KWARGS["init_tip_arcsec_std"] = 0.0
ENV_KWARGS["init_tilt_arcsec_std"] = 0.0
ENV_KWARGS["rail_baseline_random"] = False
ENV_KWARGS["rail_baseline_piston_micron"] = 0.0
ENV_KWARGS["warmup_with_zero_action"] = True

# Reward: equal-magnitude mix of dark_hole and Strehl. Holding bonus
# disabled (the two-term reward already pulls toward both nulls and a
# bright on-axis core).
ENV_KWARGS["reward_function"] = "factored"
ENV_KWARGS["reward_weight_dark_hole"] = 1.0
ENV_KWARGS["reward_weight_strehl"] = 0.25
ENV_KWARGS["reward_weight_log_mean_dark_hole"] = 0.0
ENV_KWARGS["reward_weight_centered_strehl"] = 0.0
ENV_KWARGS["reward_weight_centering"] = 0.0
ENV_KWARGS["reward_weight_flux"] = 0.0
ENV_KWARGS["holding_bonus_weight"] = 0.0

# Inner-ring default geometry. Overridden by --dark-hole-* CLI flags
# (the launcher sets these per-target).
ENV_KWARGS["dark_hole"] = True
ENV_KWARGS["dark_hole_angular_location_degrees"] = 0.0
ENV_KWARGS["dark_hole_location_radius_fraction"] = 0.16
ENV_KWARGS["dark_hole_size_radius"] = 0.095


# ----------------------------------------------------------------------
# PPO config
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["target_dim"] = 4
    cfg["bilateral_dm"] = True
    cfg["bilateral_freeze_segments"] = True
    # ----- Hyperparameter rescaling for the much larger action dim ------
    # The piston script ran at action_dim = 15; this run is at 612 (DM
    # half of a 35x35 grid), a 40x increase. Two PPO terms scale with
    # action dim because the policy is per-dim independent Gaussian and
    # both entropy and log_prob are summed across dims:
    #
    #   1. Entropy bonus = ent_coef * sum_dims(H_per_dim). To preserve
    #      the per-dim regulation pressure that the piston script's
    #      ent_coef = 0.0015 produces at D = 15, scale by 15 / 612 ~ 0.025.
    #
    #   2. PPO ratio = exp(sum_dims(log_p_new - log_p_old)). With D
    #      independent dims, the log-ratio variance scales as D under
    #      a fixed per-dim policy drift, so its stdev grows ~sqrt(D).
    #      The clip threshold log(1.2) = 0.18 stays fixed; to avoid
    #      saturating the clip every update, drop the learning rate so
    #      per-dim gradient steps shrink. sqrt(40) ~ 6, so 3e-4 / 6
    #      ~ 5e-5. Round to 1e-4 as a less aggressive starting point.
    # Empirically: even at ent_coef = 4e-5, the entropy term dominates
    # past ~10M env steps -- the policy that learned a useful early
    # solution drifts toward a uniform (high-sigma) distribution and
    # the OPD scatter blacks out the entire frame. The per-dim
    # advantage signal in this task is too weak to defend against
    # entropy pressure summed over 612 dims for that long, so we drop
    # the entropy coefficient most of the way down and tighten the
    # log_std envelope. A tiny non-zero ent_coef stays in as a floor
    # so the policy never collapses entirely to a deterministic mean.
    cfg["ent_coef"] = 1e-7
    # Floor on per-dim policy log-sigma. With ent_coef ~ 0 there's no
    # entropy pressure pushing log_std up, and PPO updates can drift
    # log_std toward -inf. Once any per-dim sigma underflows to zero,
    # the next Normal log-prob computes 1/sigma -> inf, the backward
    # pass produces NaN gradients, and the policy head silently
    # poisons itself with NaN weights -- the run then crashes minutes
    # later when Normal(loc=NaN, scale=...) is constructed. Floor at
    # log_std = -5 keeps sigma >= ~6.7e-3 per dim, well clear of
    # float32 underflow and small enough that exploration stays
    # essentially deterministic at the operating point.
    cfg["log_std_min"] = -5.0
    cfg["learning_rate"] = 1e-4
    cfg["log_std_max"] = -2.0          # per-dim sigma cap = exp(-2) ~ 0.135
    cfg["init_log_std"] = -2.5         # per-dim sigma init = exp(-2.5) ~ 0.082
    # Slightly sharper eval figures than the piston runs default to.
    # Bumping local from 48 -> 64 and HPC from 72 -> 96 roughly doubles
    # the per-figure byte size but keeps it well under the TB event
    # file threshold and gives readable detail in the blind region.
    cfg["eval_figure_dpi"] = 96 if cfg.get("num_envs", 8) >= 32 else 64
    # Joint exploration radius: sqrt(612) * 0.135 ~ 3.3 at the cap and
    # sqrt(612) * 0.082 ~ 2.0 at init -- both bounded by the env's
    # [-1, 1] action clip, so the policy still has room to explore
    # without runaway scatter.
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
