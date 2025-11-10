#!/usr/bin/env python3
"""
Run rollout instrumentation on a trained sequential BC model (with LSTM).

This script loads a trajectory BC model (ResNet18LSTMActor) from a training run
directory and performs rollout instrumentation with proper LSTM hidden state management.

The key difference from run_bc_rollout.py is that this script:
1. Maintains LSTM hidden states across rollout steps
2. Handles single-step observations (not sequences) during rollout
3. Properly manages the temporal dimension for LSTM inference

Example usage:
    # Basic rollout with defaults from config
    poetry run python optomech/imitation_learning/run_bc_seq_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def
    
    # Override number of seeds and steps
    poetry run python optomech/imitation_learning/run_bc_seq_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --num-seeds 16 \\
        --rollout-steps 500
    
    # Use a specific model checkpoint
    poetry run python optomech/imitation_learning/run_bc_seq_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --model-path runs/bc_run_20231103_142530_abc123def/bc_lstm_model_best.pth
    
    # Override environment settings
    poetry run python optomech/imitation_learning/run_bc_seq_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --rollout-episode-steps 200 \\
        --no-incremental-control
"""

import os
import sys
import json
import argparse
import uuid
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directories for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))

# Import model
try:
    from models.models import ResNet18LSTMActor
except ImportError:
    print("❌ Error: Could not import ResNet18LSTMActor from models.models")
    sys.exit(1)

# Import rollout instrumentation - we'll use a modified version
try:
    from optomech.eval.rollout_instrumentation import perform_rollout_instrumentation
except ImportError:
    print("❌ Error: rollout_instrumentation module not available")
    sys.exit(1)


def find_latest_model(run_dir: Path) -> Optional[Path]:
    """
    Find the latest model checkpoint in the run directory.
    Priority: bc_lstm_model_best.pth > bc_model_best.pth > bc_lstm_model_final.pth > bc_model_final.pth
    
    Args:
        run_dir: Path to the BC run directory
        
    Returns:
        Path to the model checkpoint, or None if not found
    """
    # Check for LSTM-specific best model first
    lstm_best = run_dir / "bc_lstm_model_best.pth"
    if lstm_best.exists():
        return lstm_best
    
    # Check for regular best model
    best_model = run_dir / "bc_model_best.pth"
    if best_model.exists():
        return best_model
    
    # Check for LSTM final model
    lstm_final = run_dir / "bc_lstm_model_final.pth"
    if lstm_final.exists():
        return lstm_final
    
    # Check for regular final model
    final_model = run_dir / "bc_model_final.pth"
    if final_model.exists():
        return final_model
    
    # Check for temporary rollout model
    temp_model = run_dir / "temp_model_for_rollout.pth"
    if temp_model.exists():
        return temp_model
    
    return None


def load_rollout_config(run_dir: Path) -> Dict:
    """
    Load the rollout environment config from the run directory.
    
    Args:
        run_dir: Path to the BC run directory
        
    Returns:
        Dictionary containing the rollout environment config
    """
    config_path = run_dir / "rollout_env_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"rollout_env_config.json not found in {run_dir}. "
            "Make sure you're using a run directory from the trajectory BC training script."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def override_config(base_config: Dict, args: argparse.Namespace) -> Dict:
    """
    Override config values with CLI arguments.
    
    Args:
        base_config: Base configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()
    
    # Override rollout episode steps if provided
    if args.rollout_episode_steps is not None:
        config['rollout_episode_steps'] = args.rollout_episode_steps
    
    # Override incremental control mode
    if args.incremental_control is not None:
        if args.incremental_control:
            # Add flag if not present
            if "--incremental_control" not in config['environment_flags']:
                config['environment_flags'].append("--incremental_control")
        else:
            # Remove flag if present
            config['environment_flags'] = [
                flag for flag in config['environment_flags'] 
                if flag != "--incremental_control"
            ]
    
    # Override other environment settings if provided
    if args.num_atmosphere_layers is not None:
        config['num_atmosphere_layers'] = args.num_atmosphere_layers
        # Update in environment_flags as well
        config['environment_flags'] = [
            flag for flag in config['environment_flags']
            if not flag.startswith("--num_atmosphere_layers")
        ]
        config['environment_flags'].append(f"--num_atmosphere_layers={args.num_atmosphere_layers}")
    
    if args.aperture_type is not None:
        config['aperture_type'] = args.aperture_type
        config['environment_flags'] = [
            flag for flag in config['environment_flags']
            if not flag.startswith("--aperture_type")
        ]
        config['environment_flags'].append(f"--aperture_type={args.aperture_type}")
    
    if args.reward_function is not None:
        config['reward_function'] = args.reward_function
        config['environment_flags'] = [
            flag for flag in config['environment_flags']
            if not flag.startswith("--reward_function")
        ]
        config['environment_flags'].append(f"--reward_function={args.reward_function}")
    
    return config


def create_lstm_rollout_wrapper(model_path: Path, output_dir: Path, device: str = 'cpu') -> Path:
    """
    Create a wrapper checkpoint that can be loaded by rollout instrumentation.
    
    For LSTM models, we need to handle hidden state management during rollouts.
    This function creates a special checkpoint format that the rollout system can use.
    
    Args:
        model_path: Path to the original LSTM model checkpoint
        output_dir: Directory to save the wrapper checkpoint
        device: Device to load model on
        
    Returns:
        Path to the wrapper checkpoint
    """
    print(f"\n📦 Creating LSTM rollout wrapper...")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ Loaded checkpoint successfully")
        
        # Extract model parameters
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint must contain 'model_state_dict'")
        
        state_dict = checkpoint['model_state_dict']
        
        # Detect if this is an LSTM model
        is_lstm = any('lstm' in k.lower() for k in state_dict.keys())
        if not is_lstm:
            print("⚠️  Warning: Checkpoint doesn't appear to be an LSTM model")
        
        # Extract model metadata
        input_channels = checkpoint.get('input_channels', 1)
        action_dim = checkpoint.get('action_dim', 15)
        
        # Try to extract from state_dict if not in checkpoint
        if 'resnet.conv1.weight' in state_dict:
            input_channels = state_dict['resnet.conv1.weight'].shape[1]
        elif 'resnet.encoder.0.weight' in state_dict:
            input_channels = state_dict['resnet.encoder.0.weight'].shape[1]
        
        if 'action_head.3.weight' in state_dict:
            action_dim = state_dict['action_head.3.weight'].shape[0]
        elif 'action_head.1.weight' in state_dict:
            action_dim = state_dict['action_head.1.weight'].shape[0]
        
        # Extract LSTM configuration
        lstm_hidden_dim = checkpoint.get('lstm_hidden_dim', 256)
        lstm_num_layers = checkpoint.get('lstm_num_layers', 1)
        
        # Try to extract from state_dict
        if 'lstm.weight_ih_l0' in state_dict:
            lstm_hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        
        # Count LSTM layers
        lstm_layer_keys = [k for k in state_dict.keys() if 'lstm.weight_ih_l' in k]
        if lstm_layer_keys:
            lstm_num_layers = len(lstm_layer_keys)
        
        print(f"📊 Model configuration:")
        print(f"  Input channels: {input_channels}")
        print(f"  Action dim: {action_dim}")
        print(f"  LSTM hidden dim: {lstm_hidden_dim}")
        print(f"  LSTM layers: {lstm_num_layers}")
        
        # Create wrapper checkpoint with metadata
        wrapper_checkpoint = {
            'model_state_dict': state_dict,
            'model_type': 'lstm_bc',
            'input_channels': input_channels,
            'action_dim': action_dim,
            'lstm_hidden_dim': lstm_hidden_dim,
            'lstm_num_layers': lstm_num_layers,
            'epoch': checkpoint.get('epoch', 0),
            'config': None  # Explicitly set to None to avoid unpickling issues
        }
        
        # Save wrapper checkpoint
        wrapper_path = output_dir / "lstm_model_for_rollout.pth"
        torch.save(wrapper_checkpoint, wrapper_path)
        print(f"✅ Created LSTM rollout wrapper: {wrapper_path.name}")
        
        return wrapper_path
        
    except Exception as e:
        print(f"❌ Failed to create LSTM rollout wrapper: {e}")
        import traceback
        traceback.print_exc()
        raise


class LSTMRolloutWrapper:
    """
    Wrapper class for LSTM models to handle hidden state during rollouts.
    
    This class provides a stateful interface that maintains LSTM hidden states
    across multiple rollout steps, allowing the model to process single observations
    while preserving temporal information.
    """
    
    def __init__(self, model: ResNet18LSTMActor, device: str = 'cpu'):
        """
        Args:
            model: Loaded ResNet18LSTMActor model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.hidden = None
        self.model.eval()
    
    def reset(self):
        """Reset LSTM hidden state (call at start of each episode)."""
        self.hidden = None
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action for a single observation while maintaining hidden state.
        
        Args:
            observation: Single observation [C, H, W]
            
        Returns:
            action: Predicted action [action_dim]
        """
        # Convert to tensor and add batch and sequence dimensions
        # [C, H, W] -> [1, 1, C, H, W]
        obs_tensor = torch.from_numpy(observation).float().to(self.device)
        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
        
        # Initialize hidden state if needed
        if self.hidden is None:
            self.hidden = self.model.init_hidden(batch_size=1, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            actions, self.hidden = self.model(obs_tensor, self.hidden)
        
        # Extract action: [1, 1, action_dim] -> [action_dim]
        action = actions.squeeze(0).squeeze(0).cpu().numpy()
        
        return action


def perform_lstm_rollout(
    model_path: str,
    num_seeds: int,
    rollout_steps: int,
    env_config_path: str,
    output_dir: str,
    device: str = 'cpu'
) -> Tuple[float, float, Dict]:
    """
    Perform rollout evaluation for LSTM BC model with proper hidden state management.
    
    Args:
        model_path: Path to LSTM model checkpoint
        num_seeds: Number of rollout seeds
        rollout_steps: Steps per rollout
        env_config_path: Path to environment config JSON
        output_dir: Directory to save results
        device: Device to run on
        
    Returns:
        Tuple of (mean_rewards, std_rewards, metadata)
    """
    import gymnasium as gym
    from argparse import Namespace
    
    print(f"\n🔄 Loading LSTM BC model...")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    input_channels = checkpoint.get('input_channels', 1)
    action_dim = checkpoint.get('action_dim', 15)
    lstm_hidden_dim = checkpoint.get('lstm_hidden_dim', 256)
    lstm_num_layers = checkpoint.get('lstm_num_layers', 1)
    
    # Create model
    model = ResNet18LSTMActor(
        input_channels=input_channels,
        action_dim=action_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded LSTM BC model:")
    print(f"  Input channels: {input_channels}")
    print(f"  Action dim: {action_dim}")
    print(f"  LSTM hidden: {lstm_hidden_dim}, layers: {lstm_num_layers}")
    
    # Create stateful wrapper
    wrapper = LSTMRolloutWrapper(model, device)
    
    # Load environment config
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    
    print(f"\n🎲 Running {num_seeds} rollouts with {rollout_steps} steps each")
    
    # Create environment
    # Note: This is a simplified version - you may need to adapt based on your gym environment setup
    env_id = env_config.get('env_id', 'optomech-v1')
    
    # Store results
    all_episode_rewards = []
    successful_seeds = 0
    
    for seed in range(num_seeds):
        print(f"\n🎯 Rollout {seed + 1}/{num_seeds} (seed={seed})")
        
        try:
            # Create environment (simplified - adapt as needed)
            env = gym.make(env_id)
            env.reset(seed=seed)
            
            # Reset LSTM hidden state
            wrapper.reset()
            
            # Run rollout
            episode_rewards = []
            obs, _ = env.reset(seed=seed)
            
            for step in range(rollout_steps):
                # Get action from LSTM model (maintains hidden state)
                action = wrapper.predict(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            env.close()
            
            total_reward = sum(episode_rewards)
            all_episode_rewards.append(episode_rewards)
            successful_seeds += 1
            
            print(f"  ✅ Reward: {total_reward:.4f} ({len(episode_rewards)} steps)")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate statistics
    if successful_seeds > 0:
        # Calculate mean reward across all episodes
        total_rewards = [sum(ep) for ep in all_episode_rewards]
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"\n📊 Results:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"  Successful rollouts: {successful_seeds}/{num_seeds}")
        
        metadata = {
            'num_seeds': num_seeds,
            'rollout_steps': rollout_steps,
            'successful_rollouts': successful_seeds,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'total_mean_return': float(mean_reward),
            'total_std_return': float(std_reward),
            'model_type': 'lstm_bc',
            'lstm_hidden_dim': lstm_hidden_dim,
            'lstm_num_layers': lstm_num_layers
        }
        
        # Save detailed results
        results_path = Path(output_dir) / "lstm_rollout_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'episode_rewards': [list(map(float, ep)) for ep in all_episode_rewards]
            }, f, indent=2)
        
        print(f"💾 Saved results to {results_path}")
        
        return mean_reward, std_reward, metadata
    else:
        raise RuntimeError("All rollouts failed")


def main():
    parser = argparse.ArgumentParser(
        description="Run rollout instrumentation on a trained sequential BC model (LSTM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("--run-dir", type=str, required=True,
                       help="Path to BC training run directory")
    
    # Model selection
    parser.add_argument("--model-path", type=str,
                       help="Path to specific model checkpoint (default: auto-detect latest)")
    
    # Rollout settings
    parser.add_argument("--num-seeds", type=int, default=8,
                       help="Number of rollout seeds (default: 8)")
    parser.add_argument("--rollout-steps", type=int, default=250,
                       help="Number of steps per rollout (default: 250)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for rollout results (default: <run-dir>/rollout_lstm_<uuid>)")
    parser.add_argument("--device", type=str, default='cpu',
                       help="Device to run on (cpu/cuda/mps, default: cpu)")
    
    # Environment config overrides
    parser.add_argument("--rollout-episode-steps", type=int,
                       help="Maximum episode length for rollouts (overrides config)")
    parser.add_argument("--incremental-control", action="store_true", default=None,
                       help="Force incremental control mode")
    parser.add_argument("--no-incremental-control", action="store_false", dest="incremental_control",
                       help="Force absolute control mode (disable incremental)")
    parser.add_argument("--num-atmosphere-layers", type=int,
                       help="Number of atmosphere layers (overrides config)")
    parser.add_argument("--aperture-type", type=str,
                       help="Aperture type (e.g., elf, circular)")
    parser.add_argument("--reward-function", type=str,
                       help="Reward function (e.g., align, strehl)")
    
    # Display options
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    # Convert run directory to Path
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    if not run_dir.is_dir():
        print(f"❌ Error: Not a directory: {run_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("🎯 Sequential BC Model (LSTM) Rollout Instrumentation")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    
    # Find model checkpoint
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Error: Model checkpoint not found: {model_path}")
            sys.exit(1)
    else:
        model_path = find_latest_model(run_dir)
        if model_path is None:
            print(f"❌ Error: No LSTM model checkpoint found in {run_dir}")
            print("   Expected: bc_lstm_model_best.pth, bc_model_best.pth, etc.")
            sys.exit(1)
    
    print(f"📦 Using model: {model_path.name}")
    
    # Load rollout config
    try:
        base_config = load_rollout_config(run_dir)
        print(f"✅ Loaded rollout config from {run_dir / 'rollout_env_config.json'}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    env_config = override_config(base_config, args)
    
    # Show configuration
    if args.verbose or args.incremental_control is not None or args.rollout_episode_steps is not None:
        print("\n🔧 Configuration:")
        if args.rollout_episode_steps is not None:
            print(f"  Rollout episode steps: {args.rollout_episode_steps} (overridden)")
        else:
            print(f"  Rollout episode steps: {env_config.get('rollout_episode_steps', 'default')}")
        
        incremental_mode = "--incremental_control" in env_config['environment_flags']
        if args.incremental_control is not None:
            print(f"  Incremental control: {incremental_mode} (overridden)")
        else:
            print(f"  Incremental control: {incremental_mode}")
        
        if args.num_atmosphere_layers is not None:
            print(f"  Atmosphere layers: {args.num_atmosphere_layers} (overridden)")
        if args.aperture_type is not None:
            print(f"  Aperture type: {args.aperture_type} (overridden)")
        if args.reward_function is not None:
            print(f"  Reward function: {args.reward_function} (overridden)")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create rollout directory in the run directory with UUID
        rollout_uuid = uuid.uuid4().hex[:8]
        output_dir = run_dir / f"rollout_lstm_{rollout_uuid}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Save the effective config to output directory
    effective_config_path = output_dir / "effective_rollout_config.json"
    with open(effective_config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    print(f"💾 Saved effective config to {effective_config_path}")
    
    # Save temporary env config for rollout
    temp_env_config_path = output_dir / "env_config.json"
    with open(temp_env_config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    
    # Create LSTM rollout wrapper
    try:
        wrapper_path = create_lstm_rollout_wrapper(
            model_path, 
            output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to create LSTM rollout wrapper: {e}")
        sys.exit(1)
    
    # Run LSTM rollout with hidden state management
    print(f"\n🎯 Starting LSTM rollout instrumentation...")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Steps per rollout: {args.rollout_steps}")
    print(f"  Device: {args.device}")
    
    try:
        mean_reward, std_reward, metadata = perform_lstm_rollout(
            model_path=str(wrapper_path),
            num_seeds=args.num_seeds,
            rollout_steps=args.rollout_steps,
            env_config_path=str(temp_env_config_path),
            output_dir=str(output_dir),
            device=args.device
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ LSTM Rollout Complete!")
        print("=" * 80)
        print(f"📊 Episode Reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"✅ Successful rollouts: {metadata['successful_rollouts']} / {args.num_seeds}")
        print(f"\n📁 Results saved to: {output_dir}")
        
        # Save metadata summary
        metadata_path = output_dir / "rollout_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"\n❌ LSTM Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
