# Standard library imports
import os
import sys
import gc
import copy
import json
import uuid
import pickle
import random
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
    Compute the correction target by adding the applied action to the segment baseline dict.
    This represents the 'correction' that should be applied given the current aberrations.
    
    Args:
        optical_system: The optical system object from the environment
        applied_action (np.ndarray): The action that was applied to generate the observation
        
    Returns:
        np.ndarray: Target correction values as a flattened array
    """
    target_correction_list = []
    segment_baseline_dict = copy.deepcopy(optical_system.segment_baseline_dict)
    
    # Convert the applied action back to segment-wise corrections
    action_idx = 0
    for segment_id, baseline_dict in segment_baseline_dict.items():
        for key, baseline_value in baseline_dict.items():
            if key == "piston" and baseline_value != 0.0:
                # The target is the correction needed: negative baseline plus applied action
                piston_correction = -baseline_value / optical_system.max_piston_correction
                # Add the applied action component
                if action_idx < len(applied_action):
                    piston_correction += applied_action[action_idx]
                    action_idx += 1
                target_correction_list.append(piston_correction)
                
            elif key == "tip" and baseline_value != 0.0:
                tip_correction = -baseline_value / optical_system.max_tip_correction
                if action_idx < len(applied_action):
                    tip_correction += applied_action[action_idx]
                    action_idx += 1
                target_correction_list.append(tip_correction)
                
            elif key == "tilt" and baseline_value != 0.0:
                tilt_correction = -baseline_value / optical_system.max_tilt_correction
                if action_idx < len(applied_action):
                    tilt_correction += applied_action[action_idx]
                    action_idx += 1
                target_correction_list.append(tilt_correction)
    
    return np.array(target_correction_list, dtype=np.float32)


@dataclass
class Args:
    """
    Command-line arguments for building optomech direct SML dataset.
    """
    # 1. Dataset Settings
    num_samples: int = 10000
    dataset_save_path: str = "./datasets/"
    dataset_name: str = "sml_dataset"
    chunked_dataset: bool = False
    chunk_size: int = 1000

    # 2. Environment Configuration
    env_id: str = "optomech-v1"
    action_type: str = "none"
    object_type: str = "single"
    aperture_type: str = "elf"
    max_episode_steps: int = 1000  # Large value since we reset after each sample
    num_envs: int = 1
    observation_mode: str = "image_only"
    focal_plane_image_size_pixels: int = 256
    observation_window_size: int = 2**1
    num_tensioners: int = 0
    num_atmosphere_layers: int = 0
    optomech_version: str = "test"
    reward_function: str = "strehl"
    silence: bool = False

    # 3. Control Settings
    incremental_control: bool = False
    command_tensioners: bool = False
    command_secondaries: bool = False
    command_tip_tilt: bool = False
    command_dm: bool = False
    discrete_control: bool = False
    discrete_control_steps: int = 128

    # 4. Rendering and Logging
    render: bool = False
    render_frequency: int = 1
    render_dpi: float = 500.0
    record_env_state_info: bool = False
    write_env_state_info: bool = False
    write_state_interval: int = 1
    state_info_save_dir: str = "./tmp/"
    report_time: bool = False

    # 5. Simulation Timing Parameters
    ao_loop_active: bool = False
    ao_interval_ms: float = 1.0
    control_interval_ms: float = 2.0
    frame_interval_ms: float = 4.0
    decision_interval_ms: float = 8.0

    # 6. Optomech-Specific Modeling Toggles
    init_differential_motion: bool = False
    simulate_differential_motion: bool = False
    randomize_dm: bool = False
    model_wind_diff_motion: bool = False
    model_gravity_diff_motion: bool = False
    model_temp_diff_motion: bool = False

    # 7. Hardware/Infrastructure Settings
    gpu_list: str = "0"

    # 8. Extended Object Parameters
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

    # Data collectors
    observations = []
    targets = []
    actions = []
    rewards = []

    print(f"Building direct SML dataset with {args.num_samples} samples...")
    print("=" * 60)

    # Generate IID samples
    for sample_idx in range(args.num_samples):
        # Reset environment for each sample to get fresh aberrations
        obs, info = env.reset()
        
        # Sample uniform random action
        action = env.action_space.sample()
        
        # Apply action and get resulting observation
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Get optical system and compute correction target
        optical_system = env.optical_system
        target = compute_correction_target(optical_system, action)
        
        # Store the data
        observations.append(next_obs.tolist())  # This is the observation after applying action
        targets.append(target.tolist())         # This is what the model should predict
        actions.append(action.tolist())         # This is the action that was applied
        rewards.append(float(reward))           # Reward after applying action
        
        # Progress reporting
        if (sample_idx + 1) % 100 == 0:
            progress = (sample_idx + 1) / args.num_samples * 100
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Progress: {sample_idx+1}/{args.num_samples} ({progress:.1f}%) | "
                  f"Avg Reward (last 100): {avg_reward:.4f}")

    # Prepare final dataset (IID format - each sample is independent, so all dones=True)
    dataset_dict = {
        'observations': observations,
        'next_observations': observations,  # For IID, next_obs = obs since no temporal structure
        'actions': actions,
        'rewards': rewards,
        'dones': [True] * len(observations),  # Each sample is independent (done=True)
        'targets': targets  # Our special SML targets
    }
    
    # Prepare metadata
    metadata = {
        'dataset_type': 'direct_sml',
        'env_id': args.env_id,
        'object_type': args.object_type,
        'aperture_type': args.aperture_type,
        'reward_function': args.reward_function,
        'num_samples': args.num_samples,
        'observation_shape': list(next_obs.shape),
        'action_space_shape': list(env.action_space.shape),
        'action_space_low': env.action_space.low.tolist(),
        'action_space_high': env.action_space.high.tolist(),
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'target_shape': list(targets[0]) if targets else []
    }
    
    # Save dataset using dataset manager
    dataset_id = dataset_manager.save_episode(dataset_dict, metadata)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"Dataset ID: {dataset_id}")
    print(f"Samples: {args.num_samples}")
    print(f"Average reward: {metadata['avg_reward']:.4f} ± {metadata['std_reward']:.4f}")
    print(f"Reward range: [{metadata['min_reward']:.4f}, {metadata['max_reward']:.4f}]")
    print(f"Observation shape: {metadata['observation_shape']}")
    print(f"Action space shape: {metadata['action_space_shape']}")
    print(f"Target shape: {metadata['target_shape']}")
    print(f"Saved to: {dataset_path}")
    
    # Cleanup
    env.close()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = tyro.cli(Args)
    cli_main(args)
    sys.exit(0)
