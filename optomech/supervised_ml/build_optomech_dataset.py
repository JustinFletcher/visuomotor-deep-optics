# Standard library imports
import os
import sys
import gc
import copy
import json
import uuid
import pickle
import random
import time
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path for local imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Third-party imports
import numpy as np
import torch
import gymnasium as gym
import tyro
import h5py
from matplotlib import pyplot as plt

# Local imports
try:
    from chunked_dataset_manager import ChunkedDatasetManager
    CHUNKED_AVAILABLE = True
except ImportError:
    from dataset_manager import DatasetManager
    CHUNKED_AVAILABLE = False

# Import job config loading function
try:
    from optomech_rollout import load_environment_config
except ImportError:
    def load_environment_config(config_path):
        """Fallback if optomech_rollout is not available"""
        with open(config_path, 'r') as f:
            return json.load(f)


def make_env(env_id, flags):
    """
    Create a thunk that instantiates the specified environment.

    Args:
        env_id (str): Gymnasium environment ID.
        flags (Namespace): Arguments for environment configuration.

    Returns:
        Callable: Thunk that creates and returns the environment.
    """
    if env_id == "optomech-v1":
        def thunk():
            env = gym.make(env_id, **vars(flags))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk
    else:
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk


def compute_correction_target(optical_system, applied_action):
    """
    Compute the perfect correction target given the current optical system state and applied action.
    This represents the perfect action that should have been applied to correct aberrations.
    
    Args:
        optical_system: The optical system object from the environment
        applied_action (np.ndarray): The random action that was applied to generate the observation
        
    Returns:
        np.ndarray: Perfect correction values as a flattened array
    """
    perfect_correction_list = []
    segment_baseline_dict = copy.deepcopy(optical_system.segment_baseline_dict)
    
    # Compute perfect corrections from baseline aberrations
    action_idx = 0
    for segment_id, baseline_dict in segment_baseline_dict.items():
        for key, baseline_value in baseline_dict.items():
            if key == "piston" and baseline_value != 0.0:
                # Perfect correction is negative of the normalized baseline
                piston_correction = -baseline_value / optical_system.max_piston_correction
                perfect_correction_list.append(piston_correction)
                action_idx += 1
                
            elif key == "tip" and baseline_value != 0.0:
                tip_correction = -baseline_value / optical_system.max_tip_correction
                perfect_correction_list.append(tip_correction)
                action_idx += 1
                
            elif key == "tilt" and baseline_value != 0.0:
                tilt_correction = -baseline_value / optical_system.max_tilt_correction
                perfect_correction_list.append(tilt_correction)
                action_idx += 1
    
    return np.array(perfect_correction_list, dtype=np.float32)


@dataclass
class Args:
    """
    Command-line arguments for building optomech direct SML dataset.
    Uses the same comprehensive configuration as SA script.
    """
    # 1. Dataset Settings
    num_samples: int = 10000
    dataset_save_path: str = "./datasets/"
    dataset_name: str = "sml_dataset"
    chunked_dataset: bool = False
    chunk_size: int = 1000
    write_frequency: int = 100  # Write samples to disk every N samples
    job_config_file: str = "sml_job_config.json"  # Job config file to use for environment settings

    # 2. Environment Configuration (same as SA)
    env_id: str = "optomech-v1"
    total_timesteps: int = 100_000_000
    action_type: str = "none"
    object_type: str = "single"
    aperture_type: str = "elf"
    max_episode_steps: int = 1000
    num_envs: int = 1
    observation_mode: str = "image_only"
    focal_plane_image_size_pixels: int = 256
    observation_window_size: int = 2**1
    num_tensioners: int = 0
    num_atmosphere_layers: int = 0
    optomech_version: str = "test"
    reward_function: str = "strehl"
    silence: bool = False

    # 3. Control Settings (same as SA)
    incremental_control: bool = False
    command_tensioners: bool = False
    command_secondaries: bool = False
    command_tip_tilt: bool = False
    command_dm: bool = False
    discrete_control: bool = False
    discrete_control_steps: int = 128

    # 4. Rendering and Logging (same as SA)
    render: bool = False
    render_frequency: int = 1
    render_dpi: float = 100.0
    record_env_state_info: bool = False
    write_env_state_info: bool = False
    write_state_interval: int = 1
    state_info_save_dir: str = "./tmp/"
    report_time: bool = False

    # 5. Simulation Timing Parameters (same as SA)
    ao_loop_active: bool = False
    ao_interval_ms: float = 1.0
    control_interval_ms: float = 2.0
    frame_interval_ms: float = 4.0
    decision_interval_ms: float = 8.0

    # 6. Optomech-Specific Modeling Toggles (same as SA)
    init_differential_motion: bool = False
    simulate_differential_motion: bool = False
    randomize_dm: bool = False
    model_wind_diff_motion: bool = False
    model_gravity_diff_motion: bool = False
    model_temp_diff_motion: bool = False

    # 7. Hardware/Infrastructure Settings (same as SA)
    gpu_list: str = "0"

    # 8. Extended Object Parameters (same as SA)
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    extended_object_distance: str = None
    extended_object_extent: str = None


def cli_main(args):
    """
    Command-line interface entry point for building optomech direct SML dataset.
    """
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    build_dataset(args)


def build_dataset(args):
    """
    Main function to build the direct supervised ML dataset.
    Generates IID samples with uniformly random actions and computes correction targets.
    """
    
    # Load job config file for environment settings
    if os.path.exists(args.job_config_file):
        print(f"📋 Loading environment config from: {args.job_config_file}")
        job_config = load_environment_config(args.job_config_file)
        config_vars = vars(job_config)
        print(f"✅ Loaded {len(config_vars)} environment flags")
        
        # Override args with job config values where they exist
        for key, value in config_vars.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"  Override: {key} = {value}")
    else:
        print(f"⚠️  Job config file not found: {args.job_config_file}")
        print("Using default environment settings from Args")
    
    # Register custom environments
    gym.envs.registration.register(
        id='optomech-v1',
        entry_point='optomech.optomech:OptomechEnv',
        max_episode_steps=args.max_episode_steps,
    )

    # Select device
    if torch.cuda.is_available():
        print("Running with CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running with MPS")
        device = torch.device("mps")
    else:
        print("Running with CPU")
        device = torch.device("cpu")

    # Create environment
    env = gym.make(args.env_id, **vars(args))
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Initialize dataset directory 
    dataset_path = Path(args.dataset_save_path)
    dataset_dir = dataset_path / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset will be saved to: {dataset_dir}")

    # Helper function to write a batch of samples to HDF5
    def write_batch(sample_pairs, batch_start_idx):
        """Write a batch of observation-action pairs to HDF5 file for efficient storage"""
        if not sample_pairs:
            return None
        
        # Convert to numpy arrays for efficient storage
        observations = np.array([pair['observation'] for pair in sample_pairs], dtype=np.uint16)
        perfect_actions = np.array([pair['perfect_action'] for pair in sample_pairs], dtype=np.float32)
        
        # Create HDF5 filename
        batch_filename = f"batch_{batch_start_idx:06d}_{batch_start_idx + len(sample_pairs) - 1:06d}.h5"
        batch_path = dataset_dir / batch_filename
        
        # Save to HDF5 with compression
        try:
            import h5py
            with h5py.File(batch_path, 'w') as f:
                # Store data with compression
                f.create_dataset('observations', data=observations, compression='gzip', compression_opts=6)
                f.create_dataset('perfect_actions', data=perfect_actions, compression='gzip', compression_opts=6)
                
                # Store metadata as attributes
                f.attrs['batch_size'] = len(sample_pairs)
                f.attrs['batch_start_idx'] = batch_start_idx
                f.attrs['dataset_type'] = 'direct_sml_pairs'
                f.attrs['env_id'] = args.env_id
                f.attrs['object_type'] = args.object_type
                f.attrs['aperture_type'] = args.aperture_type
                f.attrs['reward_function'] = args.reward_function
                f.attrs['total_samples_planned'] = args.num_samples
                f.attrs['observation_shape'] = observations.shape[1:]
                f.attrs['action_space_shape'] = perfect_actions.shape[1:]
                f.attrs['description'] = f'Direct SML pairs batch {batch_start_idx}-{batch_start_idx + len(sample_pairs) - 1}'
                
        except ImportError:
            print("⚠️  h5py not available, falling back to numpy compressed format")
            # Fallback to numpy compressed format
            batch_filename = f"batch_{batch_start_idx:06d}_{batch_start_idx + len(sample_pairs) - 1:06d}.npz"
            batch_path = dataset_dir / batch_filename
            
            np.savez_compressed(
                batch_path,
                observations=observations,
                perfect_actions=perfect_actions,
                batch_size=len(sample_pairs),
                batch_start_idx=batch_start_idx,
                dataset_type='direct_sml_pairs',
                env_id=args.env_id,
                object_type=args.object_type,
                aperture_type=args.aperture_type,
                reward_function=args.reward_function,
                total_samples_planned=args.num_samples
            )
        
        return str(batch_path)

    # Data collector for current batch
    sample_pairs = []  # List of {'observation': obs, 'perfect_action': action} pairs
    batch_ids = []  # Track all saved batch IDs

    print(f"Building direct SML dataset with {args.num_samples} samples...")
    print(f"Incremental write frequency: every {args.write_frequency} samples")
    print("=" * 60)

    # Track timing for progress reporting
    start_time = time.time()
    obs, info = env.reset()
    
    # Track consecutive perfect rewards for config validation
    consecutive_perfect_rewards = 0
    PERFECT_REWARD_THRESHOLD = 0.999  # Close to 1.0 to account for floating point precision
    MAX_CONSECUTIVE_PERFECT = 5      # If we see this many perfect rewards in a row, configs are wrong
        

    # Generate IID samples
    for sample_idx in range(args.num_samples):
        # Sample uniform random action
        action = env.action_space.sample()
        
        # Apply random action and get resulting observation
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Get optical system and compute perfect correction target
        optical_system = env.optical_system  # Use unwrapped to avoid deprecation warning
        perfect_action = compute_correction_target(optical_system, action)
        

        # Debug: Apply the correction and check reward improvement
        # Note: This is just for validation - we don't use the corrected reward in training
        corrected_action = action + perfect_action
        
        # Check if any action values are outside the valid range before clipping
        action_low = env.action_space.low
        action_high = env.action_space.high
        out_of_bounds = (corrected_action < action_low) | (corrected_action > action_high)
        if np.any(out_of_bounds):
            out_of_bounds_indices = np.where(out_of_bounds)[0]
            out_of_bounds_values = corrected_action[out_of_bounds_indices]
            low_bounds = action_low[out_of_bounds_indices]
            high_bounds = action_high[out_of_bounds_indices]
            print(f"⚠️  Warning: Corrected action has {np.sum(out_of_bounds)} values outside valid range:")
            for i, (idx, val, low, high) in enumerate(zip(out_of_bounds_indices, out_of_bounds_values, low_bounds, high_bounds)):
                print(f"    Index {idx}: {val:.4f} not in [{low:.4f}, {high:.4f}]")
                if i >= 5:  # Limit output to first 5 out-of-bounds values
                    print(f"    ... and {len(out_of_bounds_indices) - 5} more")
                    break
        
        # Clip the corrected action to valid action space bounds
        corrected_action = np.clip(corrected_action, action_low, action_high)
        _, corrected_reward, _, _, _ = env.step(corrected_action - action)
        print(f"Sample {sample_idx}: Reward before correction: {reward:.4f}, after perfect correction: {corrected_reward:.4f}")
        
        # Check for consecutive perfect rewards (indicates config issues)
        if corrected_reward >= PERFECT_REWARD_THRESHOLD:
            consecutive_perfect_rewards += 1
            if consecutive_perfect_rewards >= MAX_CONSECUTIVE_PERFECT:
                print(f"\n❌ ERROR: Detected {consecutive_perfect_rewards} consecutive perfect rewards (≥{PERFECT_REWARD_THRESHOLD:.3f})!")
                print("This is statistically impossible and indicates a configuration problem.")
                print("\n🔧 Please check your environment configuration:")
                print("   - Ensure optical aberrations are being properly introduced")
                print("   - Verify the reward function is configured correctly") 
                print("   - Check that the environment is not in a 'perfect' starting state")
                print("   - Review optomech-version, object-type, and aperture-type settings")
                print(f"\n💡 Perfect rewards should be rare in random sampling scenarios.")
                print("Terminating dataset generation early to prevent wasted computation.")
                env.close()
                return
        else:
            consecutive_perfect_rewards = 0  # Reset counter on non-perfect reward
        
        
        # Store as observation-action pair with uint16 conversion for storage efficiency
        obs_uint16 = np.clip(next_obs, 0, 65535).astype(np.uint16)  # Clip to valid uint16 range and convert
        sample_pair = {
            'observation': obs_uint16.tolist(),      # Observation after random action (model input)
            'perfect_action': corrected_action.tolist()  # Perfect correction (model target)
        }
        sample_pairs.append(sample_pair)
        
        # Write batch to disk if we've accumulated enough samples
        if len(sample_pairs) >= args.write_frequency:
            batch_start_idx = sample_idx + 1 - len(sample_pairs)
            batch_id = write_batch(sample_pairs, batch_start_idx)
            if batch_id:
                batch_ids.append(batch_id)
                print(f"💾 Wrote batch {len(batch_ids)} (samples {batch_start_idx}-{sample_idx}) to disk: {batch_id}")
            
            # Clear batch list for next batch
            sample_pairs = []
        
        # Progress reporting - more frequent and verbose for job manager parsing
        if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == args.num_samples:
            progress = (sample_idx + 1) / args.num_samples * 100
            elapsed = time.time() - start_time
            samples_per_sec = (sample_idx + 1) / elapsed if elapsed > 0 else 0
            eta_seconds = (args.num_samples - sample_idx - 1) / samples_per_sec if samples_per_sec > 0 else 0
            eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
            
            print(f"Progress: {sample_idx+1}/{args.num_samples} ({progress:.1f}%) | "
                  f"Rate: {samples_per_sec:.2f} samples/s | ETA: {eta_str}")
            
            # Flush output for real-time monitoring
            sys.stdout.flush()

    # Write any remaining samples as final batch
    if sample_pairs:
        batch_start_idx = args.num_samples - len(sample_pairs)
        batch_id = write_batch(sample_pairs, batch_start_idx)
        if batch_id:
            batch_ids.append(batch_id)
            print(f"💾 Wrote final batch {len(batch_ids)} (samples {batch_start_idx}-{args.num_samples-1}) to disk: {batch_id}")

    print(f"\n✅ Direct SML Dataset created successfully!")
    print(f"Total batches written: {len(batch_ids)}")
    print(f"Samples: {args.num_samples}")
    print(f"Write frequency: {args.write_frequency} samples per batch")
    print(f"Data format: HDF5 compressed observation-action pairs")
    print(f"Saved to: {dataset_dir}")
    if batch_ids:
        print(f"Sample batch files: {batch_ids[:3]}{'...' if len(batch_ids) > 3 else ''}")
    
    # Cleanup
    env.close()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = tyro.cli(Args)
    cli_main(args)
    sys.exit(0)
