#!/usr/bin/env python3
"""
Run rollout instrumentation on a trained BC model.

This script loads a BC model from a training run directory and performs
rollout instrumentation using the saved rollout environment config.
CLI arguments can override any config values.

Example usage:
    # Basic rollout with defaults from config
    poetry run python optomech/imitation_learning/run_bc_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def
    
    # Override number of seeds and steps
    poetry run python optomech/imitation_learning/run_bc_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --num-seeds 16 \\
        --rollout-steps 500
    
    # Use a specific model checkpoint
    poetry run python optomech/imitation_learning/run_bc_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --model-path runs/bc_run_20231103_142530_abc123def/bc_model_best.pth
    
    # Override environment settings
    poetry run python optomech/imitation_learning/run_bc_rollout.py \\
        --run-dir runs/bc_run_20231103_142530_abc123def \\
        --rollout-episode-steps 200 \\
        --no-incremental-control
"""

import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from typing import Dict, Optional

# Add parent directories for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))

# Import rollout instrumentation
try:
    from optomech.eval.rollout_instrumentation import perform_rollout_instrumentation
except ImportError:
    print("❌ Error: rollout_instrumentation module not available")
    sys.exit(1)


def find_latest_model(run_dir: Path) -> Optional[Path]:
    """
    Find the latest model checkpoint in the run directory.
    Priority: bc_model_best.pth > bc_model_final.pth > temp_model_for_rollout.pth
    
    Args:
        run_dir: Path to the BC run directory
        
    Returns:
        Path to the model checkpoint, or None if not found
    """
    # Check for best model first
    best_model = run_dir / "bc_model_best.pth"
    if best_model.exists():
        return best_model
    
    # Check for final model
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
            "Make sure you're using a run directory from the updated training script."
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


def main():
    parser = argparse.ArgumentParser(
        description="Run rollout instrumentation on a trained BC model",
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
                       help="Output directory for rollout results (default: <run-dir>/rollout_<uuid>)")
    
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
    print("🎯 BC Model Rollout Instrumentation")
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
            print(f"❌ Error: No model checkpoint found in {run_dir}")
            print("   Expected: bc_model_best.pth, bc_model_final.pth, or temp_model_for_rollout.pth")
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
    
    # Show any overrides
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
        output_dir = run_dir / f"rollout_{rollout_uuid}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Save the effective config to output directory
    effective_config_path = output_dir / "effective_rollout_config.json"
    with open(effective_config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    print(f"💾 Saved effective config to {effective_config_path}")
    
    # Save temporary env config for rollout instrumentation
    temp_env_config_path = output_dir / "env_config.json"
    with open(temp_env_config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    
    # Load the model checkpoint and create a temporary checkpoint in the format expected by rollout
    # The issue is that bc_model_best.pth contains a TrainingConfig object that can't be unpickled
    # in a different script context. We need to create a temp checkpoint like train_behavior_cloning.py does.
    print(f"\n📦 Loading model checkpoint to create rollout-compatible format...")
    try:
        import torch
        from dataclasses import dataclass
        import sys
        
        # Create a stub TrainingConfig class for unpickling
        # This allows us to load the checkpoint even though we don't have the real class
        @dataclass
        class TrainingConfig:
            """Stub class to allow unpickling of BC model checkpoints"""
            pass
        
        # Inject it into __main__ module so pickle can find it
        sys.modules['__main__'].TrainingConfig = TrainingConfig
        
        # Now load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded checkpoint successfully")
        
        # Create a new checkpoint with only state_dict (no TrainingConfig object)
        temp_model_path = output_dir / "temp_model_for_rollout.pth"
        temp_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
            'epoch': checkpoint.get('epoch', 0),
            'config': None  # Explicitly set to None to avoid unpickling issues
        }
        
        # Preserve metadata if available
        if 'input_channels' in checkpoint:
            temp_checkpoint['input_channels'] = checkpoint['input_channels']
        if 'action_dim' in checkpoint:
            temp_checkpoint['action_dim'] = checkpoint['action_dim']
        
        torch.save(temp_checkpoint, temp_model_path)
        print(f"✅ Created rollout-compatible checkpoint: {temp_model_path.name}")
        
        # Use the temp model for rollout
        rollout_model_path = str(temp_model_path)
        
    except Exception as e:
        print(f"❌ Failed to create temporary checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nThis script requires a BC model checkpoint that can be loaded.")
        print(f"The checkpoint at {model_path} may have been saved with incompatible format.")
        sys.exit(1)
    
    # Run rollout instrumentation
    print(f"\n🎯 Starting rollout instrumentation...")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Steps per rollout: {args.rollout_steps}")
    
    try:
        mean_rewards, std_rewards, rollout_metadata = perform_rollout_instrumentation(
            model_path=rollout_model_path,
            num_seeds=args.num_seeds,
            rollout_steps=args.rollout_steps,
            runs_dir=str(run_dir.parent),
            save_results=True,
            output_dir=str(output_dir),
            env_config_path=str(temp_env_config_path),
            model_type="bc"
        )
        
        # Clean up temporary model checkpoint
        if rollout_model_path != str(model_path):
            Path(rollout_model_path).unlink()
            print(f"🧹 Cleaned up temporary rollout checkpoint")
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ Rollout Complete!")
        print("=" * 80)
        
        if 'total_mean_return' in rollout_metadata and 'total_std_return' in rollout_metadata:
            print(f"📊 Episode Reward: {rollout_metadata['total_mean_return']:.4f} ± "
                  f"{rollout_metadata['total_std_return']:.4f}")
        
        if 'num_successful_rollouts' in rollout_metadata:
            print(f"✅ Successful rollouts: {rollout_metadata['num_successful_rollouts']} / {args.num_seeds}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        
        # Save metadata summary
        metadata_path = output_dir / "rollout_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(rollout_metadata, f, indent=2)
        print(f"💾 Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"\n❌ Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
