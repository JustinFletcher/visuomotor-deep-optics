#!/usr/bin/env python3
"""
Run rollout instrumentation on a trained sequential BC model (with LSTM).

This script loads a trajectory BC model (ResNet18LSTMActor) from a training run
directory and performs rollout instrumentation with proper LSTM hidden state management.

The key difference from run_bc_rollout.py is that this script automatically detects
and properly handles LSTM models through the unified rollout infrastructure.

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
        --model-name bc_model_epoch_50.pth
    
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
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))


def main():
    """Main entry point for BC LSTM rollout instrumentation."""
    parser = argparse.ArgumentParser(
        description="Run rollout instrumentation on a trained BC LSTM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the training run directory containing the model and config'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='bc_model_best.pth',
        help='Name of the model checkpoint file (default: bc_model_best.pth)'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=32,
        help='Number of random seeds to evaluate (default: 32)'
    )
    parser.add_argument(
        '--rollout-steps',
        type=int,
        default=250,
        help='Number of steps per rollout (default: 250)'
    )
    parser.add_argument(
        '--rollout-episode-steps',
        type=int,
        default=None,
        help='Override environment episode length'
    )
    parser.add_argument(
        '--no-incremental-control',
        action='store_true',
        help='Disable incremental control mode'
    )
    parser.add_argument(
        '--render-model-view',
        action='store_true',
        help='Render what the model sees (observation, action, hidden state, reward) and save as images'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    model_path = run_dir / args.model_name
    if not model_path.exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    env_config_path = run_dir / "rollout_env_config.json"
    if not env_config_path.exists():
        print(f"❌ Rollout config not found: {env_config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🎯 Sequential BC Model (LSTM) Rollout Instrumentation")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"📦 Using model: {args.model_name}")
    
    # Load rollout config
    with open(env_config_path, 'r') as f:
        rollout_config = json.load(f)
    print(f"✅ Loaded rollout config from {env_config_path}")
    
    # Apply command-line overrides to config
    print(f"\n🔧 Configuration:")
    if args.rollout_episode_steps is not None:
        rollout_config['rollout_episode_steps'] = args.rollout_episode_steps
        print(f"  Rollout episode steps: {args.rollout_episode_steps} (overridden)")
    
    # Handle incremental control override
    if args.no_incremental_control:
        # Remove incremental_control flag if present
        if 'environment_flags' in rollout_config:
            rollout_config['environment_flags'] = [
                flag for flag in rollout_config['environment_flags'] 
                if 'incremental_control' not in flag
            ]
        rollout_config['incremental_control'] = False
        print(f"  Incremental control: False (overridden)")
    
    # Create output directory for this rollout
    rollout_id = str(uuid.uuid4())[:8]
    output_dir = run_dir / f"rollout_lstm_{rollout_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Save effective config
    effective_config_path = output_dir / "effective_rollout_config.json"
    with open(effective_config_path, 'w') as f:
        json.dump(rollout_config, f, indent=2)
    print(f"💾 Saved effective config to {effective_config_path}")
    
    # Use the standard rollout instrumentation infrastructure
    print(f"\n🎯 Starting LSTM rollout instrumentation...")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Steps per rollout: {args.rollout_steps}")
    print(f"  Device: cpu")
    
    try:
        from optomech.eval.rollout_instrumentation import perform_rollout_instrumentation
        
        mean_rewards, std_rewards, rollout_metadata = perform_rollout_instrumentation(
            model_path=str(model_path),
            num_seeds=args.num_seeds,
            rollout_steps=args.rollout_steps,
            runs_dir=str(run_dir.parent),
            save_results=True,
            output_dir=str(output_dir),
            env_config_path=str(effective_config_path),
            model_type="bc",
            render_model_view=args.render_model_view
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ LSTM Rollout Complete!")
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
        print(f"\n❌ LSTM Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
