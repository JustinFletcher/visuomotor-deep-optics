#!/usr/bin/env python3
"""
Rollout Instrumentation Tool for SML Models

This script performs comprehensive rollout evaluation of trained SML models,
providing statistical analysis and visualization of model performance across
multiple random seeds.
"""

import os
import sys
import time
import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import uuid

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from argparse import Namespace

# Add parent directory for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

# Import required modules for model loading
try:
    from optomech.eval.optomech_rollout import UniversalRolloutEngine, create_model_interface
except ImportError as e:
    print(f"Warning: Could not import rollout modules: {e}")
    UniversalRolloutEngine = None
    create_model_interface = None


def find_most_recent_model(runs_dir: str = "runs") -> Tuple[str, str]:
    """
    Find the most recently saved model in the runs directory.
    
    Args:
        runs_dir: Base directory containing training runs
        
    Returns:
        Tuple of (model_path, run_directory)
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Find all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    
    # Sort by modification time to get most recent
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Look for model files in the most recent runs
    for run_dir in run_dirs:
        model_path = run_dir / "sml_model.pth"
        if model_path.exists():
            print(f"🔍 Found most recent model: {model_path}")
            return str(model_path), str(run_dir)
    
    raise FileNotFoundError(f"No model files found in any run directory")


def load_dataset_config(dataset_path: str) -> Dict:
    """
    Load environment configuration from dataset job config.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary containing environment configuration
    """
    dataset_path = Path(dataset_path)
    
    # Look for job config file
    possible_configs = [
        dataset_path / f"{dataset_path.name}_job_config.json",
        dataset_path / "job_config.json",
        dataset_path / "dataset_job_config.json"
    ]
    
    for config_path in possible_configs:
        if config_path.exists():
            print(f"📋 Loading dataset config from: {config_path}")
            with open(config_path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"No job config found for dataset: {dataset_path}")


def perform_rollout_instrumentation(
    model_path: str = None,
    num_seeds: int = 32,
    rollout_steps: int = 250,
    runs_dir: str = "runs",
    save_results: bool = True,
    output_dir: str = None,
    env_config_path: str = None,
    model_type: str = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform rollout instrumentation using the most recent model.
    
    Args:
        model_path: Path to model (if None, finds most recent)
        num_seeds: Number of random seeds to evaluate
        rollout_steps: Number of steps per rollout
        runs_dir: Directory containing training runs
        save_results: Whether to save results to disk
        output_dir: Directory to save results (if None, uses model's run dir)
        env_config_path: Path to environment config JSON file (if None, uses default)
        model_type: Type of model ('sml', 'bc', or None for auto-detect)
        
    Returns:
        Tuple of (mean_rewards, std_rewards, metadata)
    """
    print("\n🎯 Starting Rollout Instrumentation")
    print("=" * 50)
    
    # Find model if not provided
    if model_path is None:
        model_path, run_dir = find_most_recent_model(runs_dir)
        if output_dir is None:
            output_dir = run_dir
    else:
        run_dir = str(Path(model_path).parent)
        if output_dir is None:
            output_dir = run_dir
    
    # Load environment configuration 
    if env_config_path and Path(env_config_path).exists():
        print(f"📋 Loading environment config from: {env_config_path}")
        with open(env_config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract environment configuration from provided config
        dataset_config = {
            "env_id": config_data.get("env_id", "optomech-v1"),
            "object_type": config_data.get("object_type", "single"),
            "aperture_type": config_data.get("aperture_type", "elf"),
            "reward_function": config_data.get("reward_function", "align"),
            "observation_mode": config_data.get("observation_mode", "image_only"),
            "focal_plane_image_size_pixels": config_data.get("focal_plane_image_size_pixels", 256),
            "environment_flags": config_data.get("environment_flags", []),
            # Extract interval settings (critical for matching dataset conditions)
            "ao_interval_ms": config_data.get("ao_interval_ms", 5.0),
            "control_interval_ms": config_data.get("control_interval_ms", 5.0),
            "frame_interval_ms": config_data.get("frame_interval_ms", 5.0),
            "decision_interval_ms": config_data.get("decision_interval_ms", 5.0),
            "num_atmosphere_layers": config_data.get("num_atmosphere_layers", 0),
            # Extract preprocessing settings
            "log_scale": config_data.get("log_scale", False),
            "input_crop_size": config_data.get("input_crop_size", 256),
            # Extract rollout episode length override
            "rollout_episode_steps": config_data.get("rollout_episode_steps")
        }
        print(f"🔍 Environment flags from config: {dataset_config['environment_flags']}")
        print(f"🔍 Preprocessing: log_scale={dataset_config['log_scale']}, crop_size={dataset_config['input_crop_size']}")
    else:
        # Load environment configuration from sml_job_config.json
        sml_config_path = Path("optomech/supervised_ml/sml_job_config.json")
        if sml_config_path.exists():
            print(f"📋 Loading environment config from: {sml_config_path}")
            with open(sml_config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract environment configuration
            dataset_config = {
                "env_id": config_data.get("env_id", "optomech-v1"),
                "object_type": config_data.get("object_type", "single"),
                "aperture_type": config_data.get("aperture_type", "elf"),
                "reward_function": config_data.get("reward_function", "align"),
                "observation_mode": config_data.get("observation_mode", "image_only"),
                "focal_plane_image_size_pixels": config_data.get("focal_plane_image_size_pixels", 256),
                "environment_flags": config_data.get("environment_flags", [])
            }
            print(f"🔍 Environment flags from config: {dataset_config['environment_flags']}")
        else:
            print("⚠️  sml_job_config.json not found, using default environment settings")
            dataset_config = {
                "env_id": "optomech-v1",
                "object_type": "single", 
                "aperture_type": "elf",
            "reward_function": "align",
            "observation_mode": "image_only",
            "focal_plane_image_size_pixels": 256,
            "environment_flags": []
        }
    
    # Create output directory for results
    output_path = Path(output_dir)
    rollout_results_dir = output_path / "rollout_results"
    rollout_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎲 Running {num_seeds} rollouts with {rollout_steps} steps each")
    print(f"📊 Results will be saved to: {rollout_results_dir}")
    
    # Store rewards for all seeds and steps
    all_episode_rewards = []  # Shape: [num_seeds, rollout_steps]
    successful_seeds = []
    
    # Get the optomech_rollout.py path
    optomech_rollout_path = Path(__file__).parent.parent / "optomech_rollout.py"
    if not optomech_rollout_path.exists():
        raise FileNotFoundError(f"optomech_rollout.py not found at: {optomech_rollout_path}")
    
    # Prepare base environment configuration from dataset config
    env_config_path = rollout_results_dir / "rollout_env_config.json"
    
    # Create environment config based on dataset config
    rollout_env_config = {
        "env_id": dataset_config.get("env_id", "optomech-v1"),
        "object_type": dataset_config.get("object_type", "single"),
        "aperture_type": dataset_config.get("aperture_type", "elf"),
        "reward_function": dataset_config.get("reward_function", "align"),
        "observation_mode": dataset_config.get("observation_mode", "image_only"),
        "focal_plane_image_size_pixels": dataset_config.get("focal_plane_image_size_pixels", 256),
        "environment_flags": dataset_config.get("environment_flags", [])
    }
    
    # Save environment config for rollout script
    with open(env_config_path, 'w') as f:
        json.dump(rollout_env_config, f, indent=2)
    
    print(f"💾 Saved rollout environment config to: {env_config_path}")
    
    # Create environment arguments from config with comprehensive defaults
    env_args = Namespace()
    
    # Set all default values based on the Args class from build_optomech_dataset.py
    # This ensures we have all required attributes with sensible defaults
    default_values = {
        # Core environment settings
        'env_id': 'optomech-v1',
        'total_timesteps': 100_000_000,
        'action_type': 'none',
        'object_type': 'single',
        'aperture_type': 'elf',
        'max_episode_steps': rollout_steps,
        'num_envs': 1,
        'observation_mode': 'image_only',
        'focal_plane_image_size_pixels': 256,
        'observation_window_size': 2,
        'num_tensioners': 0,
        'num_atmosphere_layers': 0,
        'optomech_version': 'test',
        'reward_function': 'strehl',
        'silence': True,  # Keep quiet during rollouts
        
        # Control settings
        'incremental_control': False,
        'command_tensioners': False,
        'command_secondaries': False,
        'command_tip_tilt': False,
        'command_dm': False,
        'discrete_control': False,
        'discrete_control_steps': 128,
        
        # Rendering and logging
        'render': False,
        'render_frequency': 1,
        'render_dpi': 100.0,
        'record_env_state_info': False,
        'write_env_state_info': False,
        'write_state_interval': 1,
        'state_info_save_dir': './tmp/',
        'report_time': False,
        
        # Simulation timing
        'ao_loop_active': False,
        'ao_interval_ms': 1.0,
        'control_interval_ms': 2.0,
        'frame_interval_ms': 4.0,
        'decision_interval_ms': 8.0,
        
        # Optomech modeling toggles
        'init_differential_motion': False,
        'simulate_differential_motion': False,
        'randomize_dm': False,
        'model_wind_diff_motion': False,
        'model_gravity_diff_motion': False,
        'model_temp_diff_motion': False,
        
        # Hardware
        'gpu_list': '0',
        
        # Extended object parameters
        'extended_object_image_file': '.\\resources\\sample_image.png',
        'extended_object_distance': None,
        'extended_object_extent': None,
    }
    
    # Apply all defaults first
    for key, value in default_values.items():
        setattr(env_args, key, value)
    
    # Override with values from config
    env_args.env_id = dataset_config["env_id"]
    env_args.object_type = dataset_config["object_type"]
    env_args.aperture_type = dataset_config["aperture_type"]
    env_args.reward_function = dataset_config["reward_function"]
    env_args.observation_mode = dataset_config["observation_mode"]
    env_args.focal_plane_image_size_pixels = dataset_config["focal_plane_image_size_pixels"]
    
    # CRITICAL: Set interval values from dataset config (must match dataset generation)
    if "ao_interval_ms" in dataset_config:
        env_args.ao_interval_ms = dataset_config["ao_interval_ms"]
    if "control_interval_ms" in dataset_config:
        env_args.control_interval_ms = dataset_config["control_interval_ms"]
    if "frame_interval_ms" in dataset_config:
        env_args.frame_interval_ms = dataset_config["frame_interval_ms"]
    if "decision_interval_ms" in dataset_config:
        env_args.decision_interval_ms = dataset_config["decision_interval_ms"]
    if "num_atmosphere_layers" in dataset_config:
        env_args.num_atmosphere_layers = dataset_config["num_atmosphere_layers"]
    
    # Parse environment_flags from config and apply them (these take highest precedence)f
    for flag in dataset_config.get("environment_flags", []):
        if "=" in flag:
            key, value = flag.split("=", 1)
            key = key.lstrip("-")  # Remove leading dashes
            # Convert value to appropriate type
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            setattr(env_args, key, value)
        else:
            # Boolean flag
            key = flag.lstrip("-")
            setattr(env_args, key, True)
    
    # Override max_episode_steps if rollout_episode_steps is provided in config
    if "rollout_episode_steps" in dataset_config and dataset_config["rollout_episode_steps"] is not None:
        env_args.rollout_episode_steps = dataset_config["rollout_episode_steps"]
        print(f"🔧 Setting rollout_episode_steps to {env_args.rollout_episode_steps} from config")
    
    # Debug: Print the critical control settings
    print(f"🔍 Final control settings:")
    print(f"  incremental_control: {getattr(env_args, 'incremental_control', 'NOT_SET')}")
    print(f"  ao_interval_ms: {getattr(env_args, 'ao_interval_ms', 'NOT_SET')}")
    print(f"  control_interval_ms: {getattr(env_args, 'control_interval_ms', 'NOT_SET')}")
    print(f"  frame_interval_ms: {getattr(env_args, 'frame_interval_ms', 'NOT_SET')}")
    print(f"  decision_interval_ms: {getattr(env_args, 'decision_interval_ms', 'NOT_SET')}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Auto-detect model type if not provided
    if model_type is None:
        print("🔍 Auto-detecting model type from checkpoint...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Check for ResNet18Actor (BC) architecture
            if any('resnet.encoder' in k or 'action_head' in k for k in state_dict.keys()):
                model_type = "bc"
                print("✅ Detected BC (ResNet18Actor) model")
            # Check for SML architecture
            elif any('features' in k or 'classifier' in k for k in state_dict.keys()):
                model_type = "sml"
                print("✅ Detected SML model")
            else:
                print("⚠️  Could not determine model type, defaulting to 'sml'")
                model_type = "sml"
        except Exception as e:
            print(f"⚠️  Error detecting model type: {e}, defaulting to 'sml'")
            model_type = "sml"
    
    # Create model interface
    print(f"📦 Loading model from: {model_path}")
    try:
        model_interface = create_model_interface(
            model_path=model_path,
            model_type=model_type,
            device=device
        )
        print(f"✅ Model loaded successfully: {model_interface.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Return empty results if model loading fails
        return np.array([]), np.array([]), {}
    
    # Create rollout engine with preprocessing settings
    rollout_engine = UniversalRolloutEngine(
        model_interface=model_interface,
        env_args=env_args,
        log_scale=dataset_config.get("log_scale", False),
        input_crop_size=dataset_config.get("input_crop_size", 256)
    )
    
    # Run rollouts for each seed
    for seed in range(num_seeds):
        print(f"\n🔄 Running rollout {seed + 1}/{num_seeds} (seed={seed})")
        
        try:
            # Set seed for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Run single rollout episode
            episodic_returns, step_wise_rewards = rollout_engine.run_rollout(
                num_episodes=1,
                exploration_noise=0.0,
                save_path=None,  # Don't save individual episodes
                save_episodes=False,
                random_policy=False,
                zero_policy=False
            )
            
            # Check if we have valid results
            if len(episodic_returns) > 0:
                # If we have step-wise rewards, use them (handle length mismatch gracefully)
                if len(step_wise_rewards) > 0 and len(step_wise_rewards[0]) > 0:
                    actual_step_rewards = step_wise_rewards[0]
                    # Handle case where episode ended early or has different length
                    if len(actual_step_rewards) != rollout_steps:
                        print(f"  ⚠️  Step count mismatch: got {len(actual_step_rewards)} rewards, expected {rollout_steps}")
                        # Pad with zeros if too short, or truncate if too long
                        if len(actual_step_rewards) < rollout_steps:
                            # Pad with the last reward value or zero
                            last_reward = actual_step_rewards[-1] if actual_step_rewards else 0.0
                            actual_step_rewards.extend([last_reward] * (rollout_steps - len(actual_step_rewards)))
                        else:
                            actual_step_rewards = actual_step_rewards[:rollout_steps]
                    all_episode_rewards.append(actual_step_rewards)
                else:
                    print(f"  ⚠️  No step-wise rewards available, using synthetic rewards")
                    # Otherwise, create synthetic step-wise rewards from the episode return
                    # This is a fallback for when detailed step rewards aren't available
                    episode_return = episodic_returns[0]
                    # Distribute the episode return evenly across steps
                    step_rewards = [episode_return / rollout_steps] * rollout_steps
                    all_episode_rewards.append(step_rewards)
                
                successful_seeds.append(seed)
                print(f"  ✅ Seed {seed}: Episode return = {episodic_returns[0]:.3f}")
            else:
                print(f"  ❌ Seed {seed}: Rollout failed or incomplete")
                
        except Exception as e:
            print(f"  ❌ Seed {seed} failed: {e}")
            continue
    
    if not all_episode_rewards:
        raise RuntimeError("No successful rollouts completed")
    
    # Convert to numpy array for analysis
    all_episode_rewards = np.array(all_episode_rewards)  # Shape: [num_successful, rollout_steps]
    
    print(f"\n📈 Analysis of {len(successful_seeds)} successful rollouts:")
    print(f"  Successful seeds: {successful_seeds}")
    
    # Calculate statistics across seeds for each timestep
    mean_rewards = np.mean(all_episode_rewards, axis=0)  # Shape: [rollout_steps]
    std_rewards = np.std(all_episode_rewards, axis=0)    # Shape: [rollout_steps]
    
    # Calculate cumulative rewards for additional analysis
    cumulative_rewards = np.cumsum(all_episode_rewards, axis=1)
    mean_cumulative = np.mean(cumulative_rewards, axis=0)
    std_cumulative = np.std(cumulative_rewards, axis=0)
    
    # Print summary statistics
    final_mean_reward = mean_rewards[-1]
    final_std_reward = std_rewards[-1]
    total_mean_return = mean_cumulative[-1]
    total_std_return = std_cumulative[-1]
    
    print(f"  Final step reward: {final_mean_reward:.4f} ± {final_std_reward:.4f}")
    print(f"  Total episode return: {total_mean_return:.4f} ± {total_std_return:.4f}")
    print(f"  Mean step reward: {np.mean(mean_rewards):.4f}")
    print(f"  Max step reward: {np.max(mean_rewards):.4f}")
    print(f"  Min step reward: {np.min(mean_rewards):.4f}")
    
    # Create comprehensive plots
    if save_results:
        print(f"\n📊 Creating rollout analysis plots...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        timesteps = np.arange(rollout_steps)
        
        # Plot 1: Step-wise rewards with error bars
        ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.fill_between(timesteps, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step-wise Reward Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        ax2.plot(timesteps, mean_cumulative, 'g-', linewidth=2, label='Mean Cumulative Return')
        ax2.fill_between(timesteps,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        alpha=0.3, color='green', label='±1 Std Dev')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative Return')
        ax2.set_title('Cumulative Return Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution heatmap (all seeds)
        im = ax3.imshow(all_episode_rewards, aspect='auto', cmap='viridis', interpolation='nearest')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Seed')
        ax3.set_title('Reward Heatmap Across All Seeds')
        plt.colorbar(im, ax=ax3, label='Reward')
        
        # Plot 4: Final return distribution
        final_returns = cumulative_rewards[:, -1]
        ax4.hist(final_returns, bins=min(20, len(final_returns)), alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(final_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_returns):.2f}')
        ax4.axvline(np.median(final_returns), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(final_returns):.2f}')
        ax4.set_xlabel('Total Episode Return')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Total Episode Returns')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = rollout_results_dir / "rollout_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  📈 Saved analysis plot: {plot_path}")
        
        # Save the raw data
        results_data = {
            'successful_seeds': successful_seeds,
            'all_episode_rewards': all_episode_rewards.tolist(),
            'mean_rewards': mean_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'mean_cumulative': mean_cumulative.tolist(),
            'std_cumulative': std_cumulative.tolist(),
            'metadata': {
                'model_path': model_path,
                'num_seeds': num_seeds,
                'rollout_steps': rollout_steps,
                'successful_rollouts': len(successful_seeds),
                'final_mean_reward': float(final_mean_reward),
                'final_std_reward': float(final_std_reward),
                'total_mean_return': float(total_mean_return),
                'total_std_return': float(total_std_return),
                'mean_step_reward': float(np.mean(mean_rewards)),
                'max_step_reward': float(np.max(mean_rewards)),
                'min_step_reward': float(np.min(mean_rewards))
            }
        }
        
        # Save results as JSON
        results_path = rollout_results_dir / "rollout_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  💾 Saved results data: {results_path}")
        
        # Save results as pickle for easy loading
        pickle_path = rollout_results_dir / "rollout_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"  💾 Saved results pickle: {pickle_path}")
        
        plt.close()
    
    # Prepare metadata for return
    metadata = {
        'model_path': model_path,
        'successful_seeds': successful_seeds,
        'num_successful_rollouts': len(successful_seeds),
        'rollout_steps': rollout_steps,
        'final_mean_reward': float(final_mean_reward),
        'final_std_reward': float(final_std_reward),
        'total_mean_return': float(total_mean_return),
        'total_std_return': float(total_std_return),
        'output_dir': str(rollout_results_dir)
    }
    
    print(f"\n🎉 Rollout instrumentation completed!")
    print(f"📁 Results saved to: {rollout_results_dir}")
    
    return mean_rewards, std_rewards, metadata


def main():
    """Main CLI entry point for rollout instrumentation"""
    parser = argparse.ArgumentParser(description="Rollout Instrumentation for SML Models")
    
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model file (default: find most recent)")
    parser.add_argument("--num_seeds", type=int, default=32,
                       help="Number of random seeds for evaluation (default: 32)")
    parser.add_argument("--rollout_steps", type=int, default=250,
                       help="Number of steps per rollout episode (default: 250)")
    parser.add_argument("--runs_dir", type=str, default="runs",
                       help="Directory containing training runs (default: runs)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (default: model's run directory)")
    parser.add_argument("--no_save", action="store_true",
                       help="Disable saving results to disk")
    parser.add_argument("--env_config", type=str, default=None,
                       help="Path to environment config JSON file to use")
    
    args = parser.parse_args()
    
    try:
        # Run rollout instrumentation
        mean_rewards, std_rewards, metadata = perform_rollout_instrumentation(
            model_path=args.model_path,
            num_seeds=args.num_seeds,
            rollout_steps=args.rollout_steps,
            runs_dir=args.runs_dir,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            env_config_path=args.env_config
        )
        
        print(f"\n✅ Rollout instrumentation completed successfully!")
        print(f"📊 Mean final reward: {metadata['final_mean_reward']:.4f} ± {metadata['final_std_reward']:.4f}")
        print(f"📈 Total episode return: {metadata['total_mean_return']:.4f} ± {metadata['total_std_return']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Rollout instrumentation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
