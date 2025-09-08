#!/usr/bin/env python3
"""
Dataset Checkout Script - Visualize SML Dataset Examples

This script loads a dataset and shows example observations and their corresponding
perfect action targets for verification and understanding.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any


def load_dataset_pairs(dataset_path: Path) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Load all observation-action pairs from a dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of (observation, perfect_action, metadata) tuples
    """
    dataset_path = Path(dataset_path)
    pairs = []
    
    # Find all episode files
    episode_files = list(dataset_path.glob("episode_*.json"))
    print(f"Found {len(episode_files)} episode files in {dataset_path}")
    
    for episode_file in sorted(episode_files):
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        episode_metadata = data.get('metadata', {})
        
        # Extract pairs from metadata (new format)
        if 'sample_pairs' in episode_metadata:
            episode_pairs = episode_metadata['sample_pairs']
            for pair in episode_pairs:
                obs = np.array(pair['observation'], dtype=np.uint16)
                action = np.array(pair['perfect_action'], dtype=np.float32)
                pairs.append((obs, action, episode_metadata))
            print(f"  Loaded {len(episode_pairs)} pairs from {episode_file.name} (new pairs format)")
        
        # Fallback to legacy format if needed
        elif 'observations' in data and 'perfect_actions' in data:
            observations = data['observations']
            perfect_actions = data['perfect_actions']
            for obs, action in zip(observations, perfect_actions):
                obs = np.array(obs, dtype=np.uint16)
                action = np.array(action, dtype=np.float32)
                pairs.append((obs, action, episode_metadata))
            print(f"  Loaded {len(observations)} pairs from {episode_file.name} (legacy format)")
    
    return pairs


def visualize_observation(obs: np.ndarray, ax: plt.Axes, title: str):
    """Visualize a single observation (image)"""
    if obs.ndim == 3 and obs.shape[0] == 1:
        # Single channel image, squeeze first dimension
        img = obs[0]
    elif obs.ndim == 3 and obs.shape[0] == 2:
        # Two channel image, show first channel
        img = obs[0]
    elif obs.ndim == 2:
        # Already 2D
        img = obs
    else:
        raise ValueError(f"Unexpected observation shape: {obs.shape}")
    
    # Display image
    im = ax.imshow(img, cmap='hot', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Intensity')
    
    return im


def format_action_text(action: np.ndarray) -> str:
    """Format action array for display"""
    if action.size == 0:
        return "No corrections needed (empty action)"
    elif action.size <= 6:
        return f"Action: [{', '.join(f'{x:.4f}' for x in action)}]"
    else:
        return f"Action ({action.size} dims): [{', '.join(f'{x:.4f}' for x in action[:3])} ... {', '.join(f'{x:.4f}' for x in action[-3:])}]"


def create_dataset_checkout(dataset_path: Path, num_examples: int = 6, save_path: Path = None):
    """
    Create a visualization of dataset examples
    
    Args:
        dataset_path: Path to dataset directory
        num_examples: Number of examples to show
        save_path: Optional path to save the visualization
    """
    # Load dataset
    print("Loading dataset...")
    pairs = load_dataset_pairs(dataset_path)
    
    if not pairs:
        print("❌ No dataset pairs found!")
        return
    
    print(f"✅ Loaded {len(pairs)} total observation-action pairs")
    
    # Sample examples
    if len(pairs) > num_examples:
        # Take evenly spaced examples
        indices = np.linspace(0, len(pairs) - 1, num_examples, dtype=int)
        sample_pairs = [pairs[i] for i in indices]
    else:
        sample_pairs = pairs
    
    # Create figure
    n_cols = min(3, len(sample_pairs))
    n_rows = (len(sample_pairs) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot examples
    for i, (obs, action, metadata) in enumerate(sample_pairs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Create title with action info
        action_text = format_action_text(action)
        title = f"Sample {i+1}\n{action_text}"
        
        # Visualize observation
        visualize_observation(obs, ax, title)
        
        # Add stats as text
        stats_text = f"Obs shape: {obs.shape}\nRange: [{obs.min()}, {obs.max()}]"
        if action.size > 0:
            stats_text += f"\nAction range: [{action.min():.4f}, {action.max():.4f}]"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(sample_pairs), len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    dataset_info = sample_pairs[0][2] if sample_pairs else {}
    env_id = dataset_info.get('env_id', 'unknown')
    dataset_type = dataset_info.get('dataset_type', 'unknown')
    
    fig.suptitle(f'SML Dataset Checkout: {dataset_path.name}\n'
                 f'Environment: {env_id} | Type: {dataset_type} | '
                 f'Showing {len(sample_pairs)} of {len(pairs)} samples', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n📊 Dataset Summary:")
    print("=" * 40)
    print(f"Total samples: {len(pairs)}")
    
    if pairs:
        obs_shapes = [obs.shape for obs, _, _ in pairs[:10]]  # Check first 10
        action_sizes = [action.size for _, action, _ in pairs[:10]]
        
        print(f"Observation shapes: {set(obs_shapes)}")
        print(f"Action sizes: {set(action_sizes)}")
        
        # Action statistics
        all_actions = [action for _, action, _ in pairs if action.size > 0]
        if all_actions:
            all_actions_flat = np.concatenate(all_actions)
            print(f"Action statistics:")
            print(f"  Total non-empty actions: {len(all_actions)}")
            print(f"  Action value range: [{all_actions_flat.min():.4f}, {all_actions_flat.max():.4f}]")
            print(f"  Action mean: {all_actions_flat.mean():.4f}")
            print(f"  Action std: {all_actions_flat.std():.4f}")
        else:
            print("No non-empty actions found (all corrections empty)")


def main():
    parser = argparse.ArgumentParser(description="Visualize SML dataset examples")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--num_examples", type=int, default=6, 
                        help="Number of examples to show")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save visualization (if not provided, will display)")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    save_path = Path(args.save) if args.save else None
    
    create_dataset_checkout(dataset_path, args.num_examples, save_path)


if __name__ == "__main__":
    main()
