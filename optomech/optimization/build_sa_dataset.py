#!/usr/bin/env python3
"""
SA Dataset Builder - Simulated Annealing-based Dataset Generation

This script combines simulated annealing optimization with dataset generation to create
high-quality training datasets for optical control. It works exactly like sa.py but 
produces datasets in the same format as build_optomech_dataset.py.

Key Features:
- Runs simulated annealing optimization to find good optical control actions
- Saves ONLY accepted transitions (high-quality samples) to disk
- Records both the SA action taken AND the ideal correcting action for each state
- Supports dual learning objectives: behavior cloning (SA actions) + correction learning (perfect actions)
- Uses efficient HDF5 compression with UUID-based batch naming for parallel safety
- Maintains all SA hyperparameters and scheduling from the original sa.py

Dataset Structure:
Each saved sample contains:
- observation: State after SA action was applied (model input)
- sa_action: Action chosen by SA algorithm (behavior cloning target)  
- perfect_action: Ideal correcting action for this state (correction learning target)
- reward: Reward achieved by the SA action
- temperature: SA temperature when action was accepted
- acceptance_delta: Cost delta that was accepted
- episode_id: UUID identifying which optimization episode the sample belongs to
- episode_step: Sequential step number within the episode (starts at 0 for each episode)
- optimization_step: Total SA optimization step counter (includes both accepted and rejected steps)

Episode Grouping:
The episode_id field allows grouping transitions from the same optimization episode together.
Each episode starts with episode_step=0 and increments with each accepted transition.
Episodes reset when the environment terminates/truncates (max_episode_steps reached).

Example usage for grouping by episode:
    import h5py
    import numpy as np
    from collections import defaultdict
    
    # Load episode data
    with h5py.File('batch_file.h5', 'r') as f:
        episode_ids = f['episode_ids'][:]
        episode_steps = f['episode_steps'][:]
        sa_actions = f['sa_actions'][:]
        # ... load other arrays
    
    # Group by episode
    episodes = defaultdict(list)
    for i in range(len(episode_ids)):
        episode_id = episode_ids[i].decode('utf-8') if isinstance(episode_ids[i], bytes) else episode_ids[i]
        episodes[episode_id].append({
            'step': episode_steps[i], 
            'sa_action': sa_actions[i],
            'index': i  # Original index for accessing other arrays
        })
    
    # Sort each episode by step number
    for episode_id in episodes:
        episodes[episode_id].sort(key=lambda x: x['step'])

This enables training models for:
1. Behavior Cloning: Learn to mimic successful SA actions
2. Correction Learning: Learn to compute ideal corrections
3. Multi-objective Learning: Combine both objectives with appropriate loss weighting

Usage:
    python build_sa_dataset.py --num-samples 10000 --dataset-name my_sa_dataset
    python build_sa_dataset.py --job-config-file custom_config.json --num-samples 50000
    
The script will automatically look for job_config.json in the optimization folder if no 
absolute path is provided for --job-config-file.
"""

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
from argparse import Namespace

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

# Import job config loading function and config merger
try:
    from optomech.env_config import merge_config_with_flags
    MERGE_CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: Could not import merge_config_with_flags from optomech.env_config")
    MERGE_CONFIG_AVAILABLE = False

def load_environment_config(config_path):
    """Load environment configuration from JSON file with environment_flags support"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if this is a job config with environment_flags
    if 'environment_flags' in config and MERGE_CONFIG_AVAILABLE:
        environment_flags = config['environment_flags']
        print(f"✅ Found {len(environment_flags)} environment flags")
        
        # Use merge_config_with_flags to properly parse CLI-style flags
        env_args = merge_config_with_flags(
            config=None,  # Use default config
            flags_list=environment_flags,
            # Add any additional config from the JSON file
            **{k: v for k, v in config.items() if k != 'environment_flags'},
            # Ensure state recording is enabled for step file saving
            record_env_state_info=True,
            write_env_state_info=True,
        )
        return env_args
    else:
        # Simple dictionary conversion for direct parameter configs
        return Namespace(**config) if not isinstance(config, Namespace) else config


def sample_normal_action(action_space, std_dev=0.1):
    """
    Sample an action from a normal distribution, clipped to action space bounds.

    Args:
        action_space (gymnasium.Space): The action space of the environment.
        std_dev (float): Standard deviation for the normal distribution.

    Returns:
        np.ndarray: Sampled action within action space bounds.
    """
    mean = 0
    shape = action_space.shape
    epsilon = 1e-7
    action = np.clip(
        np.random.normal(loc=mean, scale=std_dev, size=shape),
        action_space.low + epsilon,
        action_space.high - epsilon,
    )
    return action.astype(action_space.dtype)


def sample_cauchy_action(action_space):
    """
    Sample an action from a standard Cauchy distribution, clipped to action space bounds.

    Args:
        action_space (gymnasium.Space): The action space of the environment.

    Returns:
        np.ndarray: Sampled action within action space bounds.
    """
    shape = action_space.shape
    epsilon = 1e-7
    action = np.clip(
        np.random.standard_cauchy(size=shape),
        action_space.low + epsilon,
        action_space.high - epsilon,
    )
    return action.astype(action_space.dtype)


def sample_q_gaussian_action(action_space, T, q, scale=1.0):
    """
    Sample a heavy-tailed perturbation based on an approximate q-Gaussian (Student's t) distribution.

    Args:
        action_space (gymnasium.Space): The action space of the environment.
        T (float): Temperature parameter.
        q (float): Tsallis q parameter (controls tail thickness).
        scale (float): Scaling factor for perturbation.

    Returns:
        np.ndarray: Sampled perturbation.
    """
    shape = action_space.shape
    # Student's t distribution with df related to q (q = (df+3)/(df+1))
    df = 2 / (q - 1) - 1 if 1 < q < 3 else 10
    delta = np.random.standard_t(df, size=shape) * scale * np.sqrt(T)
    return delta


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


def get_perfect_correction_action(optical_system):
    """
    Calculate the perfect action to correct optical system aberrations.
    
    Args:
        optical_system: The optical system object from the environment
        
    Returns:
        np.ndarray: Perfect action array to correct all aberrations
    """
    perfect_action_list = []
    segment_baseline_dict = copy.deepcopy(optical_system.segment_baseline_dict)
    
    for segment_id, baseline_dict in segment_baseline_dict.items():
        for key, value in baseline_dict.items():
            if key == "piston" and value != 0.0:
                piston_action_value = -1 * value / optical_system.max_piston_correction
                perfect_action_list.append(piston_action_value)
            elif key == "tip" and value != 0.0:
                tip_action_value = -1 * value / optical_system.max_tip_correction
                perfect_action_list.append(tip_action_value)
            elif key == "tilt" and value != 0.0:
                tilt_action_value = -1 * value / optical_system.max_tilt_correction
                perfect_action_list.append(tilt_action_value)
    
    return np.array(perfect_action_list, dtype=np.float32)


@dataclass
class Args:
    """
    Command-line arguments for the simulated annealing dataset builder.
    Combines SA optimization with dataset generation.
    """
    # 1. Dataset Settings
    num_samples: int = 10000
    dataset_save_path: str = "./datasets/"
    dataset_name: str = "sa_dataset"
    chunked_dataset: bool = False
    chunk_size: int = 1000
    write_frequency: int = 100  # Write samples to disk every N accepted samples
    job_config_file: str = "job_config.json"  # Job config file to use for environment settings

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

    # 9. Simulated Annealing Parameters (same as SA)
    init_temperature: float = 100.0
    std_dev_patience: int = 10
    sparsity_patience: int = 100000000000000
    temperature_patience: int = 1000000000000000
    init_std_dev: float = 1.0
    scale_patience: int = 100000000000000  # Steps before changing scale multiplier
    scale_stickiness: float = 1.0  # Default scale multiplier (sticky, persists until patience exceeded)


def cli_main(args):
    """
    Command-line interface entry point for the simulated annealing dataset builder.
    """
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    build_sa_dataset(args)


def build_sa_dataset(args):
    """
    Main function to build the SA dataset.
    Runs simulated annealing optimization and saves only accepted transitions with both
    SA actions and perfect correction actions for dual learning objectives.
    """
    
    # Load job config file for environment settings
    job_config_path = args.job_config_file
    if not os.path.isabs(job_config_path):
        # If relative path, look relative to current working directory first
        if os.path.exists(job_config_path):
            job_config_path = os.path.abspath(job_config_path)
        else:
            # If not found, try relative to the optimization folder
            optimization_dir = Path(__file__).parent
            potential_path = optimization_dir / args.job_config_file
            if os.path.exists(potential_path):
                job_config_path = potential_path
            else:
                # Keep original path for error reporting
                job_config_path = os.path.abspath(job_config_path)
    
    if os.path.exists(job_config_path):
        print(f"📋 Loading environment config from: {job_config_path}")
        with open(job_config_path, 'r') as f:
            job_config_raw = json.load(f)
        
        # Extract environment flags from job_command if available
        if 'job_command' in job_config_raw:
            job_command = job_config_raw['job_command']
            print(f"✅ Found job_command with {len(job_command)} elements")
            
            # Extract environment flags from the command line arguments
            # Handle both --flag=value and --flag value formats
            environment_flags = []
            for arg in job_command:
                if arg.startswith('--') and '=' in arg:
                    environment_flags.append(arg)
                elif arg.startswith('--') and not '=' in arg:
                    environment_flags.append(arg)
            
            print(f"✅ Extracted {len(environment_flags)} environment flags")
            if len(environment_flags) > 0:
                print(f"   First 5 flags: {environment_flags[:5]}")
                # Print timing-related flags specifically
                timing_flags = [f for f in environment_flags if 'interval_ms' in f or 'decision' in f]
                if timing_flags:
                    print(f"   Timing flags: {timing_flags}")
            
            # Create a config dict with environment_flags for compatibility
            config_for_loading = {
                'environment_flags': environment_flags,
                **{k: v for k, v in job_config_raw.items() if k != 'job_command'}
            }
            
            # Save temporary config and use existing loader
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(config_for_loading, tmp_file)
                tmp_config_path = tmp_file.name
            
            try:
                job_config = load_environment_config(tmp_config_path)
            finally:
                os.unlink(tmp_config_path)
                
        elif 'environment_flags' in job_config_raw:
            # Direct environment_flags format
            job_config = load_environment_config(job_config_path)
        else:
            # Simple dictionary format
            job_config = job_config_raw
            
        config_vars = vars(job_config) if hasattr(job_config, '__dict__') else job_config
        print(f"✅ Loaded {len(config_vars)} environment settings")
        
        # Override args with job config values where they exist
        for key, value in config_vars.items():
            if hasattr(args, key):
                # Handle special type conversions
                if key == 'max_episode_steps' and isinstance(value, str):
                    # Convert string like '10_000' to integer
                    value = int(value.replace('_', ''))
                elif key in ['ao_interval_ms', 'control_interval_ms', 'frame_interval_ms', 'decision_interval_ms'] and isinstance(value, str):
                    # Convert string intervals to float
                    value = float(value)
                
                setattr(args, key, value)
                if key in ['ao_interval_ms', 'control_interval_ms', 'frame_interval_ms', 'decision_interval_ms']:
                    print(f"  ⏱️  Override: {key} = {value}")
                else:
                    print(f"  Override: {key} = {value}")
    else:
        print(f"⚠️  Job config file not found: {job_config_path}")
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

    # Function to count existing samples in the dataset directory
    def count_existing_samples():
        """Count total samples across all existing batch files in the directory"""
        total_samples = 0
        
        # Look for HDF5 files
        try:
            h5_files = list(dataset_dir.glob("batch_*.h5"))
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if 'observations' in f:
                            total_samples += f['observations'].shape[0]
                except Exception as e:
                    print(f"⚠️  Could not read {h5_file}: {e}")
        except ImportError:
            pass
        
        # Look for NPZ files (fallback format)
        npz_files = list(dataset_dir.glob("batch_*.npz"))
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                if 'observations' in data:
                    total_samples += data['observations'].shape[0]
            except Exception as e:
                print(f"⚠️  Could not read {npz_file}: {e}")
        
        return total_samples

    # Helper function to write a batch of samples to HDF5
    def write_batch(sample_pairs, global_sample_count):
        """Write a batch of SA samples to HDF5 file with UUID naming for parallel safety"""
        if not sample_pairs:
            return None
        
        # Convert to numpy arrays for efficient storage
        observations = np.array([pair['observation'] for pair in sample_pairs], dtype=np.uint16)
        sa_actions = np.array([pair['sa_action'] for pair in sample_pairs], dtype=np.float32)
        perfect_actions = np.array([pair['perfect_action'] for pair in sample_pairs], dtype=np.float32)
        sa_incremental_actions = np.array([pair['sa_incremental_action'] for pair in sample_pairs], dtype=np.float32)
        perfect_incremental_actions = np.array([pair['perfect_incremental_action'] for pair in sample_pairs], dtype=np.float32)
        rewards = np.array([pair['reward'] for pair in sample_pairs], dtype=np.float32)
        temperatures = np.array([pair['temperature'] for pair in sample_pairs], dtype=np.float32)
        acceptance_deltas = np.array([pair['acceptance_delta'] for pair in sample_pairs], dtype=np.float32)
        episode_ids = [pair['episode_id'] for pair in sample_pairs]  # Keep as strings for UUID storage
        episode_steps = np.array([pair['episode_step'] for pair in sample_pairs], dtype=np.int32)
        optimization_steps = np.array([pair['optimization_step'] for pair in sample_pairs], dtype=np.int32)
        
        # Create UUID-based filename to prevent parallel job collisions
        batch_uuid = str(uuid.uuid4())
        batch_filename = f"batch_{batch_uuid}.h5"
        batch_path = dataset_dir / batch_filename
        
        # Save to HDF5 with compression
        try:
            import h5py
            with h5py.File(batch_path, 'w') as f:
                # Store data with compression
                f.create_dataset('observations', data=observations, compression='gzip', compression_opts=6)
                f.create_dataset('sa_actions', data=sa_actions, compression='gzip', compression_opts=6)
                f.create_dataset('perfect_actions', data=perfect_actions, compression='gzip', compression_opts=6)
                f.create_dataset('sa_incremental_actions', data=sa_incremental_actions, compression='gzip', compression_opts=6)
                f.create_dataset('perfect_incremental_actions', data=perfect_incremental_actions, compression='gzip', compression_opts=6)
                f.create_dataset('rewards', data=rewards, compression='gzip', compression_opts=6)
                f.create_dataset('temperatures', data=temperatures, compression='gzip', compression_opts=6)
                f.create_dataset('acceptance_deltas', data=acceptance_deltas, compression='gzip', compression_opts=6)
                f.create_dataset('episode_steps', data=episode_steps, compression='gzip', compression_opts=6)
                f.create_dataset('optimization_steps', data=optimization_steps, compression='gzip', compression_opts=6)
                
                # Store episode IDs as variable-length strings
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('episode_ids', data=episode_ids, dtype=dt, compression='gzip', compression_opts=6)
                
                # Store metadata as attributes
                f.attrs['batch_size'] = len(sample_pairs)
                f.attrs['batch_uuid'] = batch_uuid
                f.attrs['global_sample_count'] = global_sample_count
                f.attrs['dataset_type'] = 'sa_accepted_transitions'
                f.attrs['env_id'] = args.env_id
                f.attrs['object_type'] = args.object_type
                f.attrs['aperture_type'] = args.aperture_type
                f.attrs['reward_function'] = args.reward_function
                f.attrs['total_samples_planned'] = args.num_samples
                f.attrs['observation_shape'] = observations.shape[1:]
                f.attrs['action_space_shape'] = sa_actions.shape[1:]
                f.attrs['init_temperature'] = args.init_temperature
                f.attrs['init_std_dev'] = args.init_std_dev
                f.attrs['description'] = f'SA accepted transitions batch {batch_uuid} ({len(sample_pairs)} samples)'
                
        except ImportError:
            print("⚠️  h5py not available, falling back to numpy compressed format")
            # Fallback to numpy compressed format with UUID naming
            batch_filename = f"batch_{batch_uuid}.npz"
            batch_path = dataset_dir / batch_filename
            
            np.savez_compressed(
                batch_path,
                observations=observations,
                sa_actions=sa_actions,
                perfect_actions=perfect_actions,
                sa_incremental_actions=sa_incremental_actions,
                perfect_incremental_actions=perfect_incremental_actions,
                rewards=rewards,
                temperatures=temperatures,
                acceptance_deltas=acceptance_deltas,
                episode_ids=np.array(episode_ids, dtype='U36'),  # 36 chars for UUID strings
                episode_steps=episode_steps,
                optimization_steps=optimization_steps,
                batch_size=len(sample_pairs),
                batch_uuid=batch_uuid,
                global_sample_count=global_sample_count,
                dataset_type='sa_accepted_transitions',
                env_id=args.env_id,
                object_type=args.object_type,
                aperture_type=args.aperture_type,
                reward_function=args.reward_function,
                total_samples_planned=args.num_samples,
                init_temperature=args.init_temperature,
                init_std_dev=args.init_std_dev
            )
        
        return str(batch_path)

    # Data collector for current batch
    sample_pairs = []  # List of accepted SA transitions with all metadata
    batch_ids = []  # Track all saved batch IDs

    # Check existing samples in directory
    existing_samples = count_existing_samples()
    if existing_samples >= args.num_samples:
        print(f"✅ Target samples ({args.num_samples}) already reached! Found {existing_samples} existing samples.")
        env.close()
        return
    
    samples_needed = args.num_samples - existing_samples
    print(f"📊 Found {existing_samples} existing samples, need {samples_needed} more to reach {args.num_samples}")

    print(f"Building SA dataset with {samples_needed} additional accepted samples...")
    print(f"Incremental write frequency: every {args.write_frequency} accepted samples")
    print("=" * 60)

    # Initialize SA variables
    action_size = env.action_space.shape[0]
    
    accepted_samples = 0
    total_steps = 0
    previous_accepted_action = None  # Track previous accepted action for incremental computation
    
    # Track cost deltas for the last 100 steps for progress reporting
    recent_cost_deltas = []
    
    # Track timing for progress reporting
    start_time = time.time()
    last_print_time = start_time
    last_print_steps = 0

    print(f"Starting SA optimization to collect {samples_needed} accepted transitions...")
    
    # Outer loop: run multiple episodes until we have enough accepted samples
    while accepted_samples < samples_needed:
        # Check periodically if another job has completed the target
        if accepted_samples % 100 == 0 and accepted_samples > 0:
            current_total = count_existing_samples() + len(sample_pairs)
            if current_total >= args.num_samples:
                print(f"🎯 Target reached by other jobs! Current total: {current_total}")
                break
        
        # Start new episode
        obs, info = env.reset()
        current_episode_id = str(uuid.uuid4())  # Unique ID for this episode
        episode_step = 0  # Step counter within current episode
        
        # Start with random action for this episode
        actions = env.action_space.sample()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        current_cost = -rewards
        best_reward = rewards
        best_action = actions
        
        steps_since_acceptance = 0
        temperature_step = 0
        
        # Scale patience tracking (sticky scale multiplier)
        steps_since_scale_change = 0
        current_scale = args.scale_stickiness  # Start with default scale
        
        # Inner loop: run SA within this episode until max_episode_steps or episode ends
        for step_in_episode in range(args.max_episode_steps - 1):  # -1 because we already took one step above
            # Check if we have enough samples
            if accepted_samples >= samples_needed:
                break
                
            # Check if episode terminated or truncated
            if terminations or truncations:
                print(f"Episode {current_episode_id[:8]} ended at step {episode_step} (term: {terminations}, trunc: {truncations})")
                break
            # Simulated annealing temperature scheduling
            if steps_since_acceptance > args.temperature_patience:
                temperature_step = 1
            else:
                temperature_step += 1
            temperature = args.init_temperature / temperature_step

            # temperature = args.init_temperature / np.log(temperature_step+1)

            # Adaptive std_dev scheduling
            if steps_since_acceptance > args.std_dev_patience:
                std_dev = args.init_std_dev * np.array([
                    np.random.choice([
                        np.random.uniform(1e-0, 1e-1),
                        np.random.uniform(1e-1, 1e-2),
                        np.random.uniform(1e-2, 1e-3),
                        np.random.uniform(1e-3, 1e-4),
                        np.random.uniform(1e-4, 1e-5),
                        0.0
                    ], size=1, replace=True)
                    for _ in range(action_size)
                ], dtype=np.float32).flatten()
            else:
                std_dev = args.init_std_dev
            
            # Scale patience scheduling (sticky scale multiplier)
            if steps_since_acceptance > args.scale_patience + 1000:
                # Choose a new scale multiplier when patience is exceeded
                current_scale = np.random.choice([
                    1.0,
                    np.random.uniform(1e-0, 1e-1),
                    np.random.uniform(1e-1, 1e-2),
                    np.random.uniform(1e-2, 1e-3),
                ])
                steps_since_scale_change = 0  # Reset scale patience counter
            
            # Apply the sticky scale to std_dev (single multiplier for all dimensions)
            std_dev = std_dev * current_scale

            # Action perturbation strategy selection
            fsa = False  # Use Cauchy for fast simulated annealing
            gsa = False  # Set True to use q-Gaussian for generalized simulated annealing

            if gsa:
                q = 1.5
                T = 1.0
                candidate_actions = actions + sample_q_gaussian_action(env.action_space, T, q, scale=std_dev)
            elif fsa:
                candidate_actions = actions + std_dev * sample_cauchy_action(env.action_space)
            else:
                candidate_actions = actions + sample_normal_action(env.action_space, std_dev=std_dev)

            # Action sparsity scheduling
            if steps_since_acceptance > args.sparsity_patience:
                sparsity = np.random.uniform(0.0, 1.0)
            else:
                sparsity = 0.0
            sparsity_size = np.min([int(candidate_actions.shape[0] * sparsity), candidate_actions.shape[0]-1])
            if sparsity_size > 0:
                zero_out = np.random.choice(candidate_actions.shape[0], size=sparsity_size, replace=False)
                candidate_actions[zero_out] = 0.0

            # Clip actions to action space bounds
            candidate_actions = np.clip(candidate_actions, env.action_space.low, env.action_space.high)

            # Get perfect correction action for current state BEFORE stepping
            # This ensures the perfect action corresponds to the obs we'll save
            optical_system = env.optical_system
            perfect_action = get_perfect_correction_action(optical_system)

            # Step the environment with candidate action
            next_obs, candidate_rewards, terminations, truncations, step_infos = env.step(candidate_actions)

            # Update best reward and action if improved
            if candidate_rewards > best_reward:
                best_reward = candidate_rewards
                best_action = candidate_actions.copy()

            # Compute cost and cost delta
            candidate_cost = -candidate_rewards
            cost_delta = candidate_cost - current_cost
            
            # Track cost delta for progress reporting (keep last 100)
            recent_cost_deltas.append(cost_delta)
            if len(recent_cost_deltas) > 100:
                recent_cost_deltas.pop(0)

            # Acceptance criteria: accept if cost is reduced or with probability ~exp(-delta/T)
            accepted = False
            if (cost_delta <= 0.0) or (np.exp(-cost_delta / temperature) > random.uniform(0.0, 1.0)):
                # Accept the transition
                current_cost = candidate_cost
                steps_since_acceptance = 0
                accepted = True
                
                # Compute incremental actions (difference from previous accepted action)
                if previous_accepted_action is not None:
                    sa_incremental_action = candidate_actions - previous_accepted_action
                    perfect_incremental_action = perfect_action - previous_accepted_action
                else:
                    # For the first accepted action, incremental is the action itself (no previous reference)
                    sa_incremental_action = candidate_actions.copy()
                    perfect_incremental_action = perfect_action.copy()
                
                # Store accepted transition with both SA action and perfect action
                obs_uint16 = np.clip(obs, 0, 65535).astype(np.uint16)
                next_obs_uint16 = np.clip(next_obs, 0, 65535).astype(np.uint16)
                sample_pair = {
                    'observation': obs_uint16.tolist(),                        # Observation PRIOR TO SA action - this is the state the
                    'sa_action': candidate_actions.tolist(),                  # Action chosen by SA (behavior cloning target)
                    'sa_incremental_action': sa_incremental_action.tolist(),  # SA action increment from previous
                    'perfect_incremental_action': perfect_incremental_action.tolist(),  # Perfect action increment from previous
                    'perfect_action': perfect_action.tolist(),                # Perfect correcting action (correction learning target)
                    'reward': float(candidate_rewards),                       # Reward achieved
                    'next_observation': next_obs_uint16.tolist(),            # Observation AFTER SA action
                    'temperature': float(temperature),                        # SA temperature at acceptance
                    'acceptance_delta': float(cost_delta),                    # Cost delta that was accepted
                    'episode_id': current_episode_id,                         # Episode identifier for grouping
                    'episode_step': episode_step,                             # Step within episode
                    'optimization_step': total_steps                          # Total optimization step counter
                }
                sample_pairs.append(sample_pair)
                
                # Update state for next iteration
                actions = candidate_actions
                obs = next_obs
                accepted_samples += 1
                episode_step += 1  # Increment step counter within episode
                
                # Update previous accepted action for next incremental computation
                previous_accepted_action = candidate_actions.copy()
                
                # Write batch to disk if we've accumulated enough samples
                if len(sample_pairs) >= args.write_frequency:
                    current_total_samples = existing_samples + accepted_samples
                    batch_id = write_batch(sample_pairs, current_total_samples)
                    if batch_id:
                        batch_ids.append(batch_id)
                        print(f"💾 Wrote batch {len(batch_ids)} ({len(sample_pairs)} accepted samples) to disk: {batch_id}")
                    
                    # Clear batch list for next batch
                    sample_pairs = []
                
            else:
                # Reject the transition
                steps_since_acceptance += 1

            # Always increment scale patience counter (continues across acceptances)
            steps_since_scale_change += 1
            total_steps += 1
            
            # Progress reporting
            if total_steps % 100 == 0 or accepted_samples == samples_needed:
                current_total_samples = existing_samples + accepted_samples
                progress = current_total_samples / args.num_samples * 100
                current_time = time.time()
                elapsed = current_time - start_time
                acceptance_rate = accepted_samples / total_steps * 100 if total_steps > 0 else 0
                samples_per_sec = accepted_samples / elapsed if elapsed > 0 else 0
                remaining_needed = samples_needed - accepted_samples
                eta_seconds = remaining_needed / samples_per_sec if samples_per_sec > 0 else 0
                eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                
                # Calculate steps per second since last print
                time_since_last_print = current_time - last_print_time
                steps_since_last_print = total_steps - last_print_steps
                steps_per_sec_interval = steps_since_last_print / time_since_last_print if time_since_last_print > 0 else 0
                
                # Compute cost delta statistics for recent steps
                if recent_cost_deltas:
                    max_delta = max(recent_cost_deltas)
                    min_delta = min(recent_cost_deltas)
                    median_delta = np.median(recent_cost_deltas)
                    delta_stats = f"Δ[{min_delta:.3f},{median_delta:.3f},{max_delta:.3f}]"
                else:
                    delta_stats = "Δ[--,--,--]"
                
                print(f"Progress: {current_total_samples}/{args.num_samples} ({progress:.1f}%) | "
                      f"Accepted: {accepted_samples}/{samples_needed} | Steps: {total_steps} | "
                      f"Accept: {acceptance_rate:.1f}% | Rate: {samples_per_sec:.2f} accepted/s | "
                      f"Steps/s: {steps_per_sec_interval:.1f} | "
                      f"T: {temperature:.3f} | Best R: {best_reward:.4f} | {delta_stats} | ETA: {eta_str}")
                
                # Update tracking variables for next interval
                last_print_time = current_time
                last_print_steps = total_steps
                
                # Flush output for real-time monitoring
                sys.stdout.flush()

    # Write any remaining samples as final batch
    if sample_pairs:
        final_total_samples = existing_samples + accepted_samples
        batch_id = write_batch(sample_pairs, final_total_samples)
        if batch_id:
            batch_ids.append(batch_id)
            print(f"💾 Wrote final batch {len(batch_ids)} ({len(sample_pairs)} accepted samples) to disk: {batch_id}")

    # Get final count from directory
    final_sample_count = count_existing_samples()
    
    print(f"\n✅ SA Dataset generation completed!")
    print(f"Total accepted samples in directory: {final_sample_count}")
    print(f"Target samples: {args.num_samples}")
    print(f"Total SA steps executed: {total_steps}")
    print(f"Overall acceptance rate: {accepted_samples/total_steps*100:.2f}%")
    print(f"Best reward achieved: {best_reward:.6f}")
    print(f"Final temperature: {temperature:.6f}")
    print(f"Batches written by this job: {len(batch_ids)}")
    print(f"Write frequency: {args.write_frequency} accepted samples per batch")
    print(f"Data format: HDF5 compressed with SA actions + perfect corrections")
    print(f"Saved to: {dataset_dir}")
    if batch_ids:
        print(f"Sample batch files from this job: {[Path(b).name for b in batch_ids[:3]]}{'...' if len(batch_ids) > 3 else ''}")
    
    # Cleanup
    env.close()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = tyro.cli(Args)
    cli_main(args)
    sys.exit(0)
