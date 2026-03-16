"""
Generate a dataset of focal-plane images for autoencoder pretraining.

Accretive: each run appends batches of images as separate .npz files
with unique IDs. Run multiple times to grow the dataset. The training
script (train_ppo_autoencoder.py) loads all .npz files from the folder.

Each batch file contains:
  - images: float32 array of normalized focal-plane images (N, H, W, C)
  - obs_ref_max: reference max used for normalization
  - tip_std, tilt_std, piston_std: env params used for this batch

Usage:
    python train/ppo/generate_autoencoder_dataset.py --env-version v3
    python train/ppo/generate_autoencoder_dataset.py --env-version v3 --num-batches 50
    python train/ppo/generate_autoencoder_dataset.py --env-version v3 --output-dir datasets/autoencoder
"""

import os
import sys
import time
import uuid
import argparse
import numpy as np
import gymnasium as gym

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.train_ppo_optomech import (
    register_optomech,
    make_optomech_env,
    normalize_obs_fixed,
)


# ============================================================================
# Base environment kwargs (nanoelf, PTT enabled, minimal reward overhead)
# ============================================================================

BASE_ENV_KWARGS = {
    # --- Object / aperture ---
    "object_type": "single",
    "aperture_type": "nanoelf",
    "focal_plane_image_size_pixels": 128,
    "observation_mode": "image_only",
    "observation_window_size": 2,
    # --- Optics ---
    "wavelength": 1e-6,
    "oversampling_factor": 8,
    "bandwidth_nanometers": 200.0,
    "bandwidth_sampling": 1,
    "object_plane_extent_meters": 1.0,
    "object_plane_distance_meters": 1.0,
    # --- Atmosphere ---
    "num_atmosphere_layers": 0,
    "seeing_arcsec": 0.5,
    "outer_scale_meters": 40.0,
    # --- Timing ---
    "ao_interval_ms": 100.0,
    "control_interval_ms": 100.0,
    "frame_interval_ms": 100.0,
    "decision_interval_ms": 100.0,
    "max_episode_steps": 20,
    # --- Control ---
    "command_secondaries": True,
    "command_tip_tilt": True,
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
    # --- Correction ranges ---
    "max_piston_correction_micron": 10.0,
    "max_tip_correction_arcsec": 10.0,
    "max_tilt_correction_arcsec": 10.0,
    "get_disp_corr_max_piston_micron": 3.0,
    "get_disp_corr_max_tip_arcsec": 20.0,
    "get_disp_corr_max_tilt_arcsec": 20.0,
    # --- Initial / runtime disturbances ---
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
    "init_wind_tip_arcsec_std_tt": 1.0,
    "init_wind_tilt_arcsec_std_tt": 1.0,
    "runtime_wind_piston_micron_factor": 1.0 / 8.0,
    "runtime_wind_tip_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_tilt_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_incremental_factor": 0.01,
    "init_gravity_piston_micron_std": 300.0,
    "init_gravity_tip_arcsec_std_tt": 5.0,
    "init_gravity_tilt_arcsec_std_tt": 5.0,
    # --- Reward (minimal) ---
    "reward_function": "factored",
    "reward_weight_strehl": 1.0,
    "reward_weight_centering": 0.0,
    "reward_weight_flux": 0.0,
    "reward_weight_convex_flux": 0.0,
    "convex_flux_power": 2.0,
    "reward_weight_dist": 0.0,
    "reward_weight_concentration": 0.0,
    "reward_weight_peak": 0.0,
    "centering_mode": "gaussian",
    "centering_radius_fraction": 0.25,
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
    "action_penalty": False,
    "action_penalty_weight": 0.0,
    "oob_penalty": False,
    "oob_penalty_weight": 0.0,
    "holding_bonus_weight": 0.0,
    # --- Dark hole ---
    "dark_hole": False,
    "dark_hole_angular_location_degrees": 60,
    "dark_hole_location_radius_fraction": 0.4,
    "dark_hole_size_radius": 0.1,
    # --- Detector model ---
    "detector_power_watts": 1e-12,
    "detector_wavelength_meters": 500e-9,
    "detector_quantum_efficiency": 0.8,
    "detector_gain_e_per_dn": 0.5,
    "detector_max_dn": 65535,
    # --- AO / DM ---
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
    # --- Misc ---
    "render": False,
    "render_frequency": 1,
    "render_dpi": 100.0,
    "silence": True,
    "report_time": False,
    "record_env_state_info": False,
    "write_env_state_info": False,
    "state_info_save_dir": "./tmp/",
    "num_episodes": 1,
    "num_steps": 16,
    "tau0_seconds": 10.0,
}


# ============================================================================
# Data collection helpers
# ============================================================================

def randomize_env_kwargs(base_kwargs, rng):
    """Return env kwargs with randomized perturbation parameters."""
    kwargs = dict(base_kwargs)
    kwargs["init_wind_tip_arcsec_std_tt"] = float(rng.uniform(0.0, 2.5))
    kwargs["init_wind_tilt_arcsec_std_tt"] = float(rng.uniform(0.0, 2.5))
    kwargs["init_wind_piston_micron_std"] = float(rng.uniform(0.1, 5.0))
    return kwargs


def collect_short_trajectory(env, obs_ref_max, num_steps, rng):
    """Collect a short trajectory with random actions."""
    images = []
    obs, _ = env.reset()
    obs = normalize_obs_fixed(obs[np.newaxis], obs_ref_max)[0]
    images.append(obs)

    action_dim = env.action_space.shape[0]
    for _ in range(num_steps - 1):
        action = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        obs = normalize_obs_fixed(obs[np.newaxis], obs_ref_max)[0]
        images.append(obs)
        if terminated or truncated:
            break

    return images


def collect_batch(base_kwargs, obs_ref_max, envs_per_batch, steps_per_env, rng):
    """Collect one batch: many short trajectories from diverse env configs.

    Returns (images, metadata) where metadata records the env params used.
    """
    all_images = []
    tip_stds = []
    tilt_stds = []
    piston_stds = []

    for _ in range(envs_per_batch):
        rand_kwargs = randomize_env_kwargs(base_kwargs, rng)
        tip_stds.append(rand_kwargs["init_wind_tip_arcsec_std_tt"])
        tilt_stds.append(rand_kwargs["init_wind_tilt_arcsec_std_tt"])
        piston_stds.append(rand_kwargs["init_wind_piston_micron_std"])

        env = make_optomech_env(rand_kwargs, max_episode_steps=20)()
        traj = collect_short_trajectory(env, obs_ref_max, steps_per_env, rng)
        all_images.extend(traj)
        env.close()

    images = np.array(all_images, dtype=np.float32)
    metadata = {
        "tip_stds": np.array(tip_stds, dtype=np.float32),
        "tilt_stds": np.array(tilt_stds, dtype=np.float32),
        "piston_stds": np.array(piston_stds, dtype=np.float32),
    }
    return images, metadata


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate autoencoder training dataset (accretive)")
    parser.add_argument("--env-version", type=str, default="v3",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dataset output directory "
                             "(default: datasets/autoencoder)")
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Number of batch files to generate "
                             "(default: run indefinitely until Ctrl-C)")
    parser.add_argument("--envs-per-batch", type=int, default=32,
                        help="Number of randomized envs per batch file")
    parser.add_argument("--steps-per-env", type=int, default=8,
                        help="Trajectory length per env")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: random)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "datasets", "autoencoder")

    os.makedirs(args.output_dir, exist_ok=True)

    # Count existing batch files
    import glob
    existing = glob.glob(os.path.join(args.output_dir, "batch_*.npz"))
    print(f"Output directory: {args.output_dir}")
    print(f"Existing batches: {len(existing)}")
    if args.num_batches is not None:
        print(f"Generating: {args.num_batches} new batches "
              f"({args.envs_per_batch} envs x {args.steps_per_env} steps = "
              f"~{args.envs_per_batch * args.steps_per_env} images each)")
    else:
        print(f"Generating: indefinitely (Ctrl-C to stop) "
              f"({args.envs_per_batch} envs x {args.steps_per_env} steps = "
              f"~{args.envs_per_batch * args.steps_per_env} images each)")

    # Seed: use provided or random
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    # Register env
    env_id = f"optomech-{args.env_version}"
    register_optomech(env_id, max_episode_steps=20)
    import train.ppo.train_ppo_optomech as _mod
    _mod._ENV_ID = env_id

    # Get obs_ref_max from a reference env
    init_env = make_optomech_env(BASE_ENV_KWARGS, max_episode_steps=20)()
    base_env = init_env.unwrapped
    if hasattr(base_env, "optical_system") and hasattr(
            base_env.optical_system, "_reference_fpi_max"):
        obs_ref_max = base_env.optical_system._reference_fpi_max
    else:
        obs_ref_max = 1.0
    obs_shape = init_env.observation_space.shape
    print(f"Obs shape: {obs_shape}, ref_max: {obs_ref_max:.1f}")
    init_env.close()

    total_images = 0
    batches_written = 0
    t_start = time.time()
    num_batches = args.num_batches  # None = infinite

    try:
        batch_idx = 0
        while num_batches is None or batch_idx < num_batches:
            t0 = time.time()

            images, metadata = collect_batch(
                BASE_ENV_KWARGS, obs_ref_max,
                envs_per_batch=args.envs_per_batch,
                steps_per_env=args.steps_per_env,
                rng=rng,
            )

            # Save with unique ID
            batch_id = uuid.uuid4().hex[:12]
            filename = f"batch_{batch_id}.npz"
            filepath = os.path.join(args.output_dir, filename)
            np.savez_compressed(
                filepath,
                images=images,
                obs_ref_max=np.float32(obs_ref_max),
                **metadata,
            )

            total_images += len(images)
            batches_written += 1
            elapsed = time.time() - t0

            label = (f"{batch_idx + 1}/{num_batches}"
                     if num_batches else f"{batch_idx + 1}")
            print(f"  [{label}] "
                  f"{filename}  "
                  f"{len(images)} images  "
                  f"{elapsed:.1f}s  "
                  f"({total_images:,} total this run)")

            batch_idx += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    total_elapsed = time.time() - t_start
    final_count = len(glob.glob(os.path.join(args.output_dir, "batch_*.npz")))
    print(f"\nDone. {batches_written} batches written in {total_elapsed:.0f}s")
    print(f"Total batches in {args.output_dir}: {final_count}")
    print(f"Total images this run: {total_images:,}")


if __name__ == "__main__":
    main()
