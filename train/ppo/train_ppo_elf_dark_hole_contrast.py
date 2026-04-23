"""
PPO training: ELF piston-only dark-hole shaping (CONTRAST reward).

Variant of ``train_ppo_elf_dark_hole.py`` that swaps the normalised-
intensity-in-hole reward term for a *contrast-weighted Strehl* term:

    contrast = 1 - max(fpi[hole]) / max(fpi[~hole])
    term     = -(1 - contrast * strehl)     # in [-1, 0]

The multiplicative coupling prevents the policy from gaming the
reward by darkening the whole image (strehl = 0 kills the product),
and the term is directly comparable in magnitude to the existing
strehl / centered-strehl penalties.

Run-ID prefix ``dark_hole_contrast_`` makes these experiments easy to
tell apart from the plain-dark-hole runs in TensorBoard.

Usage:
    python train/ppo/train_ppo_elf_dark_hole_contrast.py \\
        --dark-hole-angle 45 \\
        --dark-hole-radius-frac 0.20 \\
        --dark-hole-size 0.04
    python train/ppo/train_ppo_elf_dark_hole_contrast.py --hpc ...
"""

import argparse
import sys

from train.ppo.train_ppo_optomech import run_main


# ============================================================================
# Environment kwargs
# ============================================================================

ELF_DARK_HOLE_ENV_KWARGS = {
    # --- Object / aperture -----------------------------------------------
    "object_type": "single",
    "aperture_type": "elf",
    "focal_plane_image_size_pixels": 256,
    "observation_mode": "image_only",
    "observation_window_size": 2,

    # --- Optics -----------------------------------------------------------
    "wavelength": 1e-6,
    "oversampling_factor": 8,
    "bandwidth_nanometers": 200.0,
    "bandwidth_sampling": 2,
    "object_plane_extent_meters": 1.0,
    "object_plane_distance_meters": 1.0,

    # --- Atmosphere -------------------------------------------------------
    "num_atmosphere_layers": 0,
    "seeing_arcsec": 0.5,
    "outer_scale_meters": 40.0,

    # --- Timing -----------------------------------------------------------
    "ao_interval_ms": 1.0,
    "control_interval_ms": 1.0,
    "frame_interval_ms": 1.0,
    "decision_interval_ms": 1.0,
    "max_episode_steps": 64,

    # --- Control ----------------------------------------------------------
    "command_secondaries": True,
    "command_tip_tilt": False,       # piston only
    "command_tensioners": False,
    "command_dm": False,
    "ao_loop_active": False,
    "incremental_control": True,
    "discrete_control": False,
    "discrete_control_steps": 128,
    "action_type": "none",
    "actuator_noise": True,
    "actuator_noise_fraction": 1e-4,
    "minimum_absolute_action": 0.0,
    "env_action_scale": 0.1,

    # --- Correction ranges ------------------------------------------------
    "max_piston_correction_micron": 10.0,
    "max_tip_correction_arcsec": 20.0,
    "max_tilt_correction_arcsec": 20.0,
    "get_disp_corr_max_piston_micron": 3.0,
    "get_disp_corr_max_tip_arcsec": 20.0,
    "get_disp_corr_max_tilt_arcsec": 20.0,

    # --- Initial / runtime disturbances -----------------------------------
    # Configurable init path: every DOF starts exactly at zero. The
    # dark-hole shaping task is operating on the residual wavefront
    # error the trained agent itself produces (plus the actuator
    # repeatability noise that fires on every commanded step), so
    # there is no init perturbation here.
    "init_differential_motion": False,
    "init_differential_motion_configurable": True,
    "init_piston_micron_mean": 0.0,
    "init_piston_micron_std": 0.0,
    "init_piston_clip_micron": 0.0,
    "init_tip_arcsec_std": 0.0,
    "init_tilt_arcsec_std": 0.0,
    "simulate_differential_motion": False,
    "model_wind_diff_motion": False,
    "model_gravity_diff_motion": False,
    "model_temp_diff_motion": False,
    "model_ao": False,
    "initial_ground_wind_speed_mps": 3.0,
    "ground_wind_speed_std_fraction": 0.08,
    "max_ground_wind_speed_mps": 20.0,
    "initial_ground_temp_degcel": 20.0,
    "ground_temp_ms_sampled_std": 0.0,
    "initial_gravity_normal_deg": 45.0,
    "gravity_normal_ms_sampled_std": 0.0,
    "init_wind_piston_micron_std": 3.0,
    "init_wind_piston_clip_m": 4e-6,
    "init_wind_tip_arcsec_std_tt": 0.05,
    "init_wind_tilt_arcsec_std_tt": 0.05,
    "runtime_wind_piston_micron_factor": 1.0 / 8.0,
    "runtime_wind_tip_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_tilt_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_incremental_factor": 0.01,
    "init_gravity_piston_micron_std": 300.0,
    "init_gravity_tip_arcsec_std_tt": 15.0,
    "init_gravity_tilt_arcsec_std_tt": 15.0,

    # --- Reward -----------------------------------------------------------
    # The dark-hole contribution is bounded in [-1, 0] and lives in the
    # ~[-0.05, -0.01] band for our regime, so we weight it ~20x to put
    # it on a comparable gradient footing with the centered-Strehl term.
    "reward_function": "factored",
    "reward_weight_strehl": 0.0,
    "reward_weight_centering": 0.0,
    "reward_weight_flux": 0.0,
    "reward_weight_convex_flux": 0.0,
    "convex_flux_power": 2.0,
    "reward_weight_dist": 0.0,
    "reward_weight_concentration": 0.0,
    "reward_weight_peak": 0.0,
    "reward_weight_centered_strehl": 1.0,
    "centering_mode": "circular",
    "centering_radius_fraction": 0.25,
    "centering_sigma_fraction": 0.25,
    # Contrast-weighted Strehl replaces the plain dark-hole flux term.
    # Both the contrast factor and the Strehl factor live in [0, 1],
    # so the multiplicative product is in [0, 1] and the resulting
    # penalty -(1 - contrast*strehl) is in [-1, 0]. Weight 1.0 keeps
    # it on parity with the centered-Strehl term.
    "reward_weight_contrast_strehl": 1.0,
    "reward_weight_dark_hole": 0.0,
    "reward_weight_image_quality": 0.0,
    "reward_weight_shape": 0.0,
    "reward_threshold": 25.0,
    "align_radius": 32,
    "align_radius_max_expand": 64,
    "align_mse_expand_threshold": -1.25,
    "ao_closed_inv_slope_threshold": 2e6,
    "dark_hole_alpha": 0.0,
    "action_penalty": True,
    "action_penalty_weight": 0.5,
    "oob_penalty": False,
    "oob_penalty_weight": 0.0,
    "holding_bonus_weight": 1.0,
    "holding_bonus_min_reward": -1.0,
    "holding_bonus_threshold": -0.7,

    # --- Dark hole (placeholders; overridden by CLI flags) ----------------
    "dark_hole": True,
    "dark_hole_angular_location_degrees": 45.0,
    "dark_hole_location_radius_fraction": 0.20,
    "dark_hole_size_radius": 0.04,

    # --- Detector model ---------------------------------------------------
    "detector_power_watts": 1e-12,
    "detector_wavelength_meters": 500e-9,
    "detector_quantum_efficiency": 0.8,
    "detector_gain_e_per_dn": 0.5,
    "detector_max_dn": 65535,

    # --- AO / DM ----------------------------------------------------------
    "randomize_dm": False,
    "dm_gain": 0.6,
    "dm_leakage": 0.01,
    "dm_model_type": "gaussian_influence",
    "dm_num_actuators_across": 35,
    "dm_num_modes": 500,
    "shwfs_f_number": 50,
    "shwfs_num_lenslets": 40,
    "shwfs_diameter_m": 5e-3,
    "dm_interaction_rcond": 1e-3,
    "dm_probe_amp_fraction": 0.01,
    "dm_flux_limit_fraction": 0.5,
    "dm_cache_dir": "./tmp/cache/",
    "microns_opd_per_actuator_bit": 0.00015,
    "stroke_count_limit": 20000,
    "num_tensioners": 0,

    # --- Misc -------------------------------------------------------------
    "render": False,
    "render_frequency": 1,
    "render_dpi": 100.0,
    "silence": False,
    "report_time": False,
    "record_env_state_info": False,
    "write_env_state_info": False,
    "state_info_save_dir": "./tmp/",
    "num_episodes": 1,
    "num_steps": 16,
    "tau0_seconds": 10.0,
}


# ============================================================================
# PPO hyperparameters
# ============================================================================

LOCAL_CONFIG = dict(
    total_timesteps=100_000_000,
    num_envs=8,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=False,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    lstm_hidden_dim=256,
    channel_scale=32,
    fc_scale=256,
    init_log_std=-2.0,
    action_scale=1.0,
    max_episode_steps=64,
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    model_save_interval=100,
    # Downsample per-step diagnostic scalars; with num_steps=128 this
    # emits 4 step-logs per rollout instead of 128.
    tb_step_log_interval=32,
    # Keep any embedded eval figures small (4x saving in bytes).
    eval_figure_dpi=48,
    env_kwargs=ELF_DARK_HOLE_ENV_KWARGS,
)

HPC_CONFIG = dict(
    total_timesteps=100_000_000,
    num_envs=64,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=False,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    lstm_hidden_dim=256,
    channel_scale=32,
    fc_scale=256,
    init_log_std=-2.0,
    action_scale=1.0,
    max_episode_steps=64,
    env_version="v5",
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    model_save_interval=100,
    # Aggressive scalar-downsampling and tiny embedded eval figures.
    # With num_steps=128 and interval=128 we emit exactly ONE step-log
    # per rollout — the bare minimum for training-curve visibility.
    tb_step_log_interval=128,
    eval_figure_dpi=24,
    env_kwargs=ELF_DARK_HOLE_ENV_KWARGS,
)


if __name__ == "__main__":
    # Pre-parse dark-hole geometry + seed before run_main sees the args.
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
