"""
PPO training: nanoelf piston-only alignment (2 DOF).

Trains a recurrent PPO agent to align piston offsets on the nanoelf
distributed-aperture telescope (2 segments, 1 piston DOF each).

All hyperparameters are embedded in this file so that a training run
is fully reproducible from this single artifact.

Usage:
    python train/ppo/train_ppo_nanoelf_piston.py                          # local run (v4, 8 envs)
    python train/ppo/train_ppo_nanoelf_piston.py --hpc                    # HPC run (v5, 64 envs, no eval)
    python train/ppo/train_ppo_nanoelf_piston.py --env-version v3         # use optomech-v3
    python train/ppo/train_ppo_nanoelf_piston.py --model-save-interval 50 # checkpoint every 50 updates
"""

from train.ppo.train_ppo_optomech import run_main

# ============================================================================
# Environment kwargs — every key the OptomechEnv requires
# ============================================================================

NANOELF_PISTON_ENV_KWARGS = {
    # --- Object / aperture -----------------------------------------------
    "object_type": "single",
    "aperture_type": "nanoelf",
    "focal_plane_image_size_pixels": 128,
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
    "ao_interval_ms": 100.0,
    "control_interval_ms": 100.0,
    "frame_interval_ms": 100.0,
    "decision_interval_ms": 100.0,
    "max_episode_steps": 100,

    # --- Control ----------------------------------------------------------
    "command_secondaries": True,
    "command_tip_tilt": False,       # <<< PISTON ONLY
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
    "init_differential_motion": True,
    "simulate_differential_motion": False,
    "model_wind_diff_motion": True,
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
    "reward_function": "factored",
    "reward_weight_strehl": 1.0,
    "reward_weight_centering": 0.0,  # <<< no centering for piston
    "centering_sigma_fraction": 0.25,
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

    # --- Dark hole --------------------------------------------------------
    "dark_hole": False,
    "dark_hole_angular_location_degrees": 60,
    "dark_hole_location_radius_fraction": 0.4,
    "dark_hole_size_radius": 0.1,

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
    # --- PPO algorithm ---
    total_timesteps=100_000_000,
    num_envs=8,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=True,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    # --- Model architecture ---
    lstm_hidden_dim=128,
    channel_scale=16,
    fc_scale=128,
    init_log_std=-0.5,
    action_scale=1.0,
    # --- Environment ---
    max_episode_steps=256,
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- Env kwargs ---
    env_kwargs=NANOELF_PISTON_ENV_KWARGS,
)

HPC_CONFIG = dict(
    # --- PPO algorithm (tuned for V5 batched GPU env on H100) ---
    total_timesteps=100_000_000,
    num_envs=64,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=True,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    # --- Model architecture ---
    lstm_hidden_dim=128,
    channel_scale=16,
    fc_scale=128,
    init_log_std=-0.5,
    action_scale=1.0,
    # --- Environment ---
    max_episode_steps=256,
    env_version="v5",
    # no_eval: use --no-eval CLI flag, not config
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- Env kwargs ---
    env_kwargs=NANOELF_PISTON_ENV_KWARGS,
)


if __name__ == "__main__":
    run_main(LOCAL_CONFIG, HPC_CONFIG)
