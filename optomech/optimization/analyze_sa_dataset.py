#!/usr/bin/env python3
"""
SA Dataset Analyzer

Simple script to load and analyze SA datasets.
Plots histograms of action values to understand their scale and distribution.

Usage:
    python analyze_sa_dataset.py --dataset-path datasets/sa_dataset_100k
    python analyze_sa_dataset.py --dataset-path datasets/sa_dataset_100k --max-files 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("⚠️  h5py not available, will only load .npz files")


def load_dataset_stats(dataset_path, max_files=None):
    """
    Load SA dataset and compute statistics on action values.
    
    Args:
        dataset_path: Path to dataset directory
        max_files: Maximum number of batch files to load (None = all)
        
    Returns:
        Dictionary with action statistics
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all batch files
    h5_files = list(dataset_dir.glob("batch_*.h5")) if HDF5_AVAILABLE else []
    npz_files = list(dataset_dir.glob("batch_*.npz"))
    all_files = h5_files + npz_files
    
    if not all_files:
        raise ValueError(f"No batch files found in {dataset_dir}")
    
    print(f"Found {len(all_files)} batch files ({len(h5_files)} HDF5, {len(npz_files)} NPZ)")
    
    if max_files:
        all_files = all_files[:max_files]
        print(f"Loading first {len(all_files)} files only")
    
    # Collect all actions
    sa_actions_all = []
    perfect_actions_all = []
    sa_incremental_actions_all = []
    perfect_incremental_actions_all = []
    rewards_all = []
    temperatures_all = []
    cost_deltas_all = []
    
    total_samples = 0
    
    print("Loading dataset...")
    for file_path in tqdm(all_files):
        try:
            if file_path.suffix == '.h5' and HDF5_AVAILABLE:
                with h5py.File(file_path, 'r') as f:
                    sa_actions_all.append(f['sa_actions'][:])
                    perfect_actions_all.append(f['perfect_actions'][:])
                    sa_incremental_actions_all.append(f['sa_incremental_actions'][:])
                    perfect_incremental_actions_all.append(f['perfect_incremental_actions'][:])
                    rewards_all.append(f['rewards'][:])
                    temperatures_all.append(f['temperatures'][:])
                    # Handle both old and new field names
                    if 'cost_deltas' in f:
                        cost_deltas_all.append(f['cost_deltas'][:])
                    else:
                        cost_deltas_all.append(f['acceptance_deltas'][:])
                    total_samples += f['sa_actions'].shape[0]
            elif file_path.suffix == '.npz':
                data = np.load(file_path)
                sa_actions_all.append(data['sa_actions'])
                perfect_actions_all.append(data['perfect_actions'])
                sa_incremental_actions_all.append(data['sa_incremental_actions'])
                perfect_incremental_actions_all.append(data['perfect_incremental_actions'])
                rewards_all.append(data['rewards'])
                temperatures_all.append(data['temperatures'])
                # Handle both old and new field names
                if 'cost_deltas' in data:
                    cost_deltas_all.append(data['cost_deltas'])
                else:
                    cost_deltas_all.append(data['acceptance_deltas'])
                total_samples += data['sa_actions'].shape[0]
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue
    
    if not sa_actions_all:
        raise ValueError("No data could be loaded from any batch files")
    
    # Concatenate all data
    sa_actions = np.concatenate(sa_actions_all, axis=0)
    perfect_actions = np.concatenate(perfect_actions_all, axis=0)
    sa_incremental_actions = np.concatenate(sa_incremental_actions_all, axis=0)
    perfect_incremental_actions = np.concatenate(perfect_incremental_actions_all, axis=0)
    rewards = np.concatenate(rewards_all, axis=0)
    temperatures = np.concatenate(temperatures_all, axis=0)
    cost_deltas = np.concatenate(cost_deltas_all, axis=0)
    
    print(f"✅ Loaded {total_samples} samples from {len(all_files)} files")
    print(f"   Action dimension: {sa_actions.shape[1]}")
    
    # Compute statistics
    stats = {
        'total_samples': total_samples,
        'action_dim': sa_actions.shape[1],
        'sa_actions': sa_actions,
        'perfect_actions': perfect_actions,
        'sa_incremental_actions': sa_incremental_actions,
        'perfect_incremental_actions': perfect_incremental_actions,
        'rewards': rewards,
        'temperatures': temperatures,
        'cost_deltas': cost_deltas,
        # SA action statistics
        'sa_actions_mean': np.mean(sa_actions, axis=0),
        'sa_actions_std': np.std(sa_actions, axis=0),
        'sa_actions_min': np.min(sa_actions, axis=0),
        'sa_actions_max': np.max(sa_actions, axis=0),
        'sa_actions_median': np.median(sa_actions, axis=0),
        # SA incremental action statistics
        'sa_inc_mean': np.mean(sa_incremental_actions, axis=0),
        'sa_inc_std': np.std(sa_incremental_actions, axis=0),
        'sa_inc_min': np.min(sa_incremental_actions, axis=0),
        'sa_inc_max': np.max(sa_incremental_actions, axis=0),
        'sa_inc_median': np.median(sa_incremental_actions, axis=0),
        # Perfect action statistics
        'perfect_actions_mean': np.mean(perfect_actions, axis=0),
        'perfect_actions_std': np.std(perfect_actions, axis=0),
        'perfect_actions_min': np.min(perfect_actions, axis=0),
        'perfect_actions_max': np.max(perfect_actions, axis=0),
        # Perfect incremental action statistics
        'perfect_inc_mean': np.mean(perfect_incremental_actions, axis=0),
        'perfect_inc_std': np.std(perfect_incremental_actions, axis=0),
        'perfect_inc_min': np.min(perfect_incremental_actions, axis=0),
        'perfect_inc_max': np.max(perfect_incremental_actions, axis=0),
        # Reward statistics
        'rewards_mean': np.mean(rewards),
        'rewards_std': np.std(rewards),
        'rewards_min': np.min(rewards),
        'rewards_max': np.max(rewards),
        'rewards_median': np.median(rewards),
    }
    
    return stats


def plot_action_histograms(stats, save_path=None):
    """
    Plot histograms of action values.
    
    Args:
        stats: Dictionary with action statistics
        save_path: Optional path to save figure
    """
    action_dim = stats['action_dim']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SA Dataset Action Analysis', fontsize=16, fontweight='bold')
    
    # 1. SA Incremental Actions - All dimensions flattened (no filtering)
    ax = axes[0, 0]
    sa_inc_flat = stats['sa_incremental_actions'].flatten()
    counts, bins, patches = ax.hist(sa_inc_flat, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Action Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'SA Incremental Actions (all {len(sa_inc_flat):,} values)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(sa_inc_flat):.7f}\n'
                  f'Std: {np.std(sa_inc_flat):.7f}\n'
                  f'Median: {np.median(sa_inc_flat):.7f}\n'
                  f'Min: {np.min(sa_inc_flat):.7f}\n'
                  f'Max: {np.max(sa_inc_flat):.7f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Perfect Incremental Actions - All dimensions flattened (no filtering)
    ax = axes[0, 1]
    perfect_inc_flat = stats['perfect_incremental_actions'].flatten()
    counts, bins, patches = ax.hist(perfect_inc_flat, bins=100, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Action Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Perfect Incremental Actions (all {len(perfect_inc_flat):,} values)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(perfect_inc_flat):.4f}\n'
                  f'Std: {np.std(perfect_inc_flat):.4f}\n'
                  f'Median: {np.median(perfect_inc_flat):.4f}\n'
                  f'Min: {np.min(perfect_inc_flat):.4f}\n'
                  f'Max: {np.max(perfect_inc_flat):.4f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. SA Actions (non-incremental, absolute) - All dimensions flattened (no filtering)
    ax = axes[1, 0]
    sa_actions_flat = stats['sa_actions'].flatten()
    counts, bins, patches = ax.hist(sa_actions_flat, bins=100, alpha=0.7, color='darkblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Action Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'SA Actions (non-incremental, all {len(sa_actions_flat):,} values)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(sa_actions_flat):.7f}\n'
                  f'Std: {np.std(sa_actions_flat):.7f}\n'
                  f'Median: {np.median(sa_actions_flat):.7f}\n'
                  f'Min: {np.min(sa_actions_flat):.7f}\n'
                  f'Max: {np.max(sa_actions_flat):.7f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Perfect Actions (non-incremental, absolute) - All dimensions flattened (no filtering)
    ax = axes[1, 1]
    perfect_actions_flat = stats['perfect_actions'].flatten()
    counts, bins, patches = ax.hist(perfect_actions_flat, bins=100, alpha=0.7, color='darkgreen', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Action Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Perfect Actions (non-incremental, all {len(perfect_actions_flat):,} values)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(perfect_actions_flat):.7f}\n'
                  f'Std: {np.std(perfect_actions_flat):.7f}\n'
                  f'Median: {np.median(perfect_actions_flat):.7f}\n'
                  f'Min: {np.min(perfect_actions_flat):.7f}\n'
                  f'Max: {np.max(perfect_actions_flat):.7f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    plt.show()


def print_statistics(stats):
    """Print detailed statistics about the dataset."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    print(f"\nDataset Size:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Action dimension: {stats['action_dim']}")
    
    print(f"\nSA Incremental Actions:")
    print(f"  Mean (per dim):   [{', '.join([f'{x:.4f}' for x in stats['sa_inc_mean'][:5]])}...]")
    print(f"  Std (per dim):    [{', '.join([f'{x:.4f}' for x in stats['sa_inc_std'][:5]])}...]")
    print(f"  Overall mean:     {np.mean(stats['sa_incremental_actions']):.6f}")
    print(f"  Overall std:      {np.std(stats['sa_incremental_actions']):.6f}")
    print(f"  Overall median:   {np.median(stats['sa_incremental_actions']):.6f}")
    print(f"  Overall min:      {np.min(stats['sa_incremental_actions']):.6f}")
    print(f"  Overall max:      {np.max(stats['sa_incremental_actions']):.6f}")
    print(f"  95th percentile:  {np.percentile(stats['sa_incremental_actions'], 95):.6f}")
    print(f"  99th percentile:  {np.percentile(stats['sa_incremental_actions'], 99):.6f}")
    
    print(f"\nPerfect Incremental Actions:")
    print(f"  Mean (per dim):   [{', '.join([f'{x:.4f}' for x in stats['perfect_inc_mean'][:5]])}...]")
    print(f"  Std (per dim):    [{', '.join([f'{x:.4f}' for x in stats['perfect_inc_std'][:5]])}...]")
    print(f"  Overall mean:     {np.mean(stats['perfect_incremental_actions']):.6f}")
    print(f"  Overall std:      {np.std(stats['perfect_incremental_actions']):.6f}")
    print(f"  Overall median:   {np.median(stats['perfect_incremental_actions']):.6f}")
    print(f"  Overall min:      {np.min(stats['perfect_incremental_actions']):.6f}")
    print(f"  Overall max:      {np.max(stats['perfect_incremental_actions']):.6f}")
    
    print(f"\nSA Actions (absolute):")
    print(f"  Overall mean:     {np.mean(stats['sa_actions']):.6f}")
    print(f"  Overall std:      {np.std(stats['sa_actions']):.6f}")
    print(f"  Overall min:      {np.min(stats['sa_actions']):.6f}")
    print(f"  Overall max:      {np.max(stats['sa_actions']):.6f}")
    
    print(f"\nRewards:")
    print(f"  Mean:             {stats['rewards_mean']:.6f}")
    print(f"  Std:              {stats['rewards_std']:.6f}")
    print(f"  Median:           {stats['rewards_median']:.6f}")
    print(f"  Min:              {stats['rewards_min']:.6f}")
    print(f"  Max:              {stats['rewards_max']:.6f}")
    
    print(f"\nTemperatures:")
    print(f"  Mean:             {np.mean(stats['temperatures']):.6f}")
    print(f"  Min:              {np.min(stats['temperatures']):.6f}")
    print(f"  Max:              {np.max(stats['temperatures']):.6f}")
    
    print(f"\nCost Deltas:")
    print(f"  Mean:             {np.mean(stats['cost_deltas']):.6f}")
    print(f"  Std:              {np.std(stats['cost_deltas']):.6f}")
    print(f"  Min:              {np.min(stats['cost_deltas']):.6f}")
    print(f"  Max:              {np.max(stats['cost_deltas']):.6f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SA dataset and plot action value distributions"
    )
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to SA dataset directory")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of batch files to load (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save analysis outputs (default: dataset directory)")
    parser.add_argument("--save-plot", type=str, default=None,
                       help="Path to save the plot (default: <output-dir>/action_analysis.png)")
    
    args = parser.parse_args()
    
    print("SA Dataset Analyzer")
    print("=" * 80)
    print(f"Dataset path: {args.dataset_path}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    
    # Load dataset and compute statistics
    try:
        stats = load_dataset_stats(args.dataset_path, max_files=args.max_files)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Print statistics
    print_statistics(stats)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.dataset_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot histograms
    if args.save_plot:
        save_path = args.save_plot
    else:
        # Auto-generate save path in output directory
        save_path = output_dir / "action_analysis.png"
    
    plot_action_histograms(stats, save_path=save_path)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
