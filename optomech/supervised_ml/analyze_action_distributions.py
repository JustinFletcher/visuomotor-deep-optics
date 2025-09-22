#!/usr/bin/env python3
"""
Script to analyze action value distributions in SML datasets.
Plots histograms of all action values across all examples in the dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("⚠️  h5py not available, HDF5 files cannot be loaded")

# Add parent directory for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


def load_single_episode(episode_file: Path) -> List[np.ndarray]:
    """
    Load a single episode file and return action values
    
    Args:
        episode_file: Path to episode JSON file
        
    Returns:
        List of action arrays
    """
    actions = []
    
    try:
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        # Extract actions from metadata (new format)
        if 'sample_pairs' in data['metadata']:
            episode_pairs = data['metadata']['sample_pairs']
            for pair in episode_pairs:
                action = np.array(pair['perfect_action'], dtype=np.float32)
                
                # Only include pairs with non-empty actions
                if action.size > 0:
                    actions.append(action)
        
        # Fallback to legacy format if needed
        elif 'perfect_actions' in data:
            perfect_actions = data['perfect_actions']
            for action in perfect_actions:
                action = np.array(action, dtype=np.float32)
                
                # Only include pairs with non-empty actions
                if action.size > 0:
                    actions.append(action)
                    
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Failed to load {episode_file}: {e}")
        return []
    
    return actions


def load_dataset_actions(dataset_path: str, max_examples: int = None) -> np.ndarray:
    """
    Load all action values from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        max_examples: Maximum number of examples to load
        
    Returns:
        Flattened array of all action values
    """
    dataset_path = Path(dataset_path)
    
    print(f"📂 Loading actions from {dataset_path}")
    all_actions = []
    total_examples = 0
    
    # Find all dataset files (prioritize HDF5, then NPZ, then JSON)
    h5_files = list(dataset_path.glob("*.h5"))
    npz_files = list(dataset_path.glob("*.npz"))
    json_files = list(dataset_path.glob("episode_*.json")) + list(dataset_path.glob("batch_*.json"))
    
    print(f"Found {len(h5_files)} H5, {len(npz_files)} NPZ, {len(json_files)} JSON files")
    
    # Load HDF5 files (preferred format)
    if h5_files and HDF5_AVAILABLE:
        print("Loading from HDF5 files...")
        for h5_file in sorted(h5_files):
            if max_examples and total_examples >= max_examples:
                break
                
            try:
                with h5py.File(h5_file, 'r') as f:
                    perfect_actions = f['perfect_actions'][:]
                    
                    # Add actions
                    for action in perfect_actions:
                        if max_examples and total_examples >= max_examples:
                            break
                        all_actions.extend(action.flatten())
                        total_examples += 1
                    
                print(f"  Loaded {len(perfect_actions)} actions from {h5_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {h5_file.name}: {e}")
    
    # Load NPZ files (fallback format)
    elif npz_files:
        print("Loading from NPZ files...")
        for npz_file in sorted(npz_files):
            if max_examples and total_examples >= max_examples:
                break
                
            try:
                data = np.load(npz_file)
                perfect_actions = data['perfect_actions']
                
                # Add actions
                for action in perfect_actions:
                    if max_examples and total_examples >= max_examples:
                        break
                    all_actions.extend(action.flatten())
                    total_examples += 1
                
                print(f"  Loaded {len(perfect_actions)} actions from {npz_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {npz_file.name}: {e}")
    
    # Load JSON files (legacy format)
    elif json_files:
        print("Loading from JSON files (legacy format)...")
        for episode_file in sorted(json_files):
            if max_examples and total_examples >= max_examples:
                break
                
            episode_actions = load_single_episode(episode_file)
            
            # Add actions, respecting max_examples limit
            for action in episode_actions:
                if max_examples and total_examples >= max_examples:
                    break
                all_actions.extend(action.flatten())
                total_examples += 1
            
            if len(episode_actions) > 0:
                print(f"  Loaded {len(episode_actions)} actions from {episode_file.name}")
    else:
        print("❌ No dataset files found!")
        return np.array([])
    
    all_actions = np.array(all_actions)
    print(f"✅ Total action values loaded: {len(all_actions):,}")
    print(f"📊 Examples processed: {total_examples:,}")
    
    return all_actions


def plot_action_distribution(actions: np.ndarray, dataset_name: str, save_path: str = None):
    """
    Plot histogram of action value distribution
    
    Args:
        actions: Flattened array of all action values
        dataset_name: Name of dataset for plot title
        save_path: Optional path to save plot
    """
    if len(actions) == 0:
        print("❌ No actions to plot!")
        return
    
    # Calculate statistics
    mean_val = np.mean(actions)
    std_val = np.std(actions)
    min_val = np.min(actions)
    max_val = np.max(actions)
    median_val = np.median(actions)
    
    print(f"\n📊 Action Value Statistics:")
    print(f"  Count: {len(actions):,}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std:  {std_val:.6f}")
    print(f"  Min:  {min_val:.6f}")
    print(f"  Max:  {max_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    n_bins = min(100, int(np.sqrt(len(actions))))
    counts, bins, patches = plt.hist(actions, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.xlabel('Action Value')
    plt.ylabel('Count')
    plt.title(f'Action Value Distribution - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log-scale histogram
    plt.subplot(2, 2, 2)
    plt.hist(actions, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.xlabel('Action Value')
    plt.ylabel('Count (log scale)')
    plt.title(f'Action Value Distribution (Log Scale) - {dataset_name}')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 3)
    plt.boxplot(actions, vert=True)
    plt.ylabel('Action Value')
    plt.title('Box Plot of Action Values')
    plt.grid(True, alpha=0.3)
    
    # Statistics text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Dataset: {dataset_name}
    
    Total Values: {len(actions):,}
    
    Mean: {mean_val:.6f}
    Median: {median_val:.6f}
    Std Dev: {std_val:.6f}
    
    Min: {min_val:.6f}
    Max: {max_val:.6f}
    Range: {max_val - min_val:.6f}
    
    Percentiles:
    5%:  {np.percentile(actions, 5):.6f}
    25%: {np.percentile(actions, 25):.6f}
    75%: {np.percentile(actions, 75):.6f}
    95%: {np.percentile(actions, 95):.6f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze action value distributions in SML datasets")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to analyze")
    parser.add_argument("--save_plot", type=str, default=None,
                       help="Path to save the plot (optional)")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for the dataset (for plot title)")
    
    args = parser.parse_args()
    
    # Use dataset path as name if not provided
    if args.dataset_name is None:
        args.dataset_name = Path(args.dataset_path).name
    
    print("🔍 SML Dataset Action Distribution Analyzer")
    print("=" * 50)
    
    # Load actions
    actions = load_dataset_actions(args.dataset_path, args.max_examples)
    
    if len(actions) == 0:
        print("❌ No actions found in dataset!")
        return
    
    # Plot distribution
    plot_action_distribution(actions, args.dataset_name, args.save_plot)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
