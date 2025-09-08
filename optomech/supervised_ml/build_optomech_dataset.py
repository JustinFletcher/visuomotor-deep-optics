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
from matplotlib import pyplot as plt

# Local imports
try:
    from chunked_dataset_manager import ChunkedDatasetManager
    CHUNKED_AVAILABLE = True
except ImportError:
    from dataset_manager import DatasetManager
    CHUNKED_AVAILABLE = False


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
    render_dpi: float = 500.0
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

    # Initialize dataset manager
    dataset_path = args.dataset_save_path
    
    if args.chunked_dataset and CHUNKED_AVAILABLE and args.num_samples > 10000:
        print(f"Using chunked dataset manager (chunk_size={args.chunk_size})")
        dataset_manager = ChunkedDatasetManager(dataset_path, chunk_size=args.chunk_size)
        use_chunked = True
    else:
        print("Using standard dataset manager")
        from dataset_manager import DatasetManager
        dataset_manager = DatasetManager(dataset_path)
        use_chunked = False

    # Helper function to write a batch of samples
    def write_batch(sample_pairs, batch_start_idx):
        """Write a batch of observation-action pairs to disk"""
        if not sample_pairs:
            return None
            
        # Prepare dataset for SML - store as observation-action pairs for easy training iteration
        # Also maintain compatibility with dataset manager by providing expected keys
        dataset_dict = {
            'sample_pairs': sample_pairs,  # NEW: List of {'observation': obs, 'perfect_action': action} dicts
            'observations': [pair['observation'] for pair in sample_pairs],  # Compatibility 
            'next_observations': [pair['observation'] for pair in sample_pairs],  # Same as observations for SML
            'actions': [[0.0] for _ in range(len(sample_pairs))],  # Dummy actions (not used)
            'rewards': [0.0 for _ in range(len(sample_pairs))],    # Dummy rewards (not used)
            'dones': [False for _ in range(len(sample_pairs))],    # Dummy dones (not used)
            'perfect_actions': [pair['perfect_action'] for pair in sample_pairs],  # Compatibility
            'batch_size': len(sample_pairs),
            'batch_start_idx': batch_start_idx
        }
        
        # Prepare metadata for this batch (including the pairs in metadata for preservation)
        metadata = {
            'dataset_type': 'direct_sml_pairs',
            'env_id': args.env_id,
            'object_type': args.object_type,
            'aperture_type': args.aperture_type,
            'reward_function': args.reward_function,
            'batch_size': len(sample_pairs),
            'batch_start_idx': batch_start_idx,
            'total_samples_planned': args.num_samples,
            'observation_shape': len(sample_pairs[0]['observation']) if sample_pairs else 0,
            'action_space_shape': list(env.action_space.shape),
            'perfect_action_shape': len(sample_pairs[0]['perfect_action']) if sample_pairs else 0,
            'description': f'Direct SML pairs batch {batch_start_idx}-{batch_start_idx + len(sample_pairs) - 1}: observation-action pairs',
            'sample_pairs': sample_pairs  # Store the pairs in metadata to preserve them
        }
        
        # Save batch
        dataset_id = dataset_manager.save_episode(dataset_dict, metadata)
        return dataset_id

    # Data collector for current batch
    sample_pairs = []  # List of {'observation': obs, 'perfect_action': action} pairs
    batch_ids = []  # Track all saved batch IDs

    print(f"Building direct SML dataset with {args.num_samples} samples...")
    print(f"Incremental write frequency: every {args.write_frequency} samples")
    print("=" * 60)

    # Track timing for progress reporting
    start_time = time.time()
    obs, info = env.reset()
        

    # Generate IID samples
    for sample_idx in range(args.num_samples):
        # Sample uniform random action
        action = env.action_space.sample()
        
        # Apply random action and get resulting observation
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Get optical system and compute perfect correction target
        optical_system = env.optical_system
        perfect_action = compute_correction_target(optical_system, action)
        
        # Store as observation-action pair with uint16 conversion for storage efficiency
        obs_uint16 = np.clip(next_obs, 0, 65535).astype(np.uint16)  # Clip to valid uint16 range and convert
        sample_pair = {
            'observation': obs_uint16.tolist(),      # Observation after random action (model input)
            'perfect_action': perfect_action.tolist()  # Perfect correction (model target)
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

    # Create summary metadata for all batches
    summary_metadata = {
        'dataset_type': 'direct_sml_summary',
        'env_id': args.env_id,
        'object_type': args.object_type,
        'aperture_type': args.aperture_type,
        'reward_function': args.reward_function,
        'total_samples': args.num_samples,
        'write_frequency': args.write_frequency,
        'num_batches': len(batch_ids),
        'batch_ids': batch_ids,
        'observation_shape': list(next_obs.shape) if 'next_obs' in locals() else [],
        'action_space_shape': list(env.action_space.shape),
        'description': f'Direct SML dataset summary: {len(batch_ids)} batches totaling {args.num_samples} samples'
    }
    
    print(f"\n✅ Direct SML Dataset created successfully!")
    print(f"Total batches written: {len(batch_ids)}")
    print(f"Samples: {args.num_samples}")
    print(f"Write frequency: {args.write_frequency} samples per batch")
    print(f"Observation shape: {summary_metadata['observation_shape']}")
    print(f"Action space shape: {summary_metadata['action_space_shape']}")
    print(f"Data format: observation-action pairs for easy training iteration")
    print(f"Saved to: {dataset_path}")
    if batch_ids:
        print(f"Batch IDs: {batch_ids}")
    
    # Cleanup
    env.close()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = tyro.cli(Args)
    cli_main(args)
    sys.exit(0)
