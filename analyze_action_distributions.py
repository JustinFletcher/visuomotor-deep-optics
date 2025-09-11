#!/usr/bin/env python3
"""
Analyze the distribution of action values in our debug dataset.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_action_distributions(dataset_path):
    """Analyze the distribution of perfect action values."""
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"Dataset directory {dataset_dir} does not exist!")
        return
    
    # Find all episode files
    episode_files = list(dataset_dir.glob("episode_*.json"))
    print(f"Found {len(episode_files)} episode files")
    
    all_perfect_actions = []
    
    # Load all perfect actions
    for episode_file in episode_files:
        print(f"Loading {episode_file.name}...")
        
        with open(episode_file, 'r') as f:
            episode_data = json.load(f)
        
        # Extract perfect actions
        for transition in episode_data['transitions']:
            perfect_action = transition['perfect_action']
            all_perfect_actions.append(perfect_action)
    
    # Convert to numpy array
    all_perfect_actions = np.array(all_perfect_actions)
    print(f"Loaded {len(all_perfect_actions)} perfect action vectors")
    print(f"Each vector has {all_perfect_actions.shape[1]} dimensions")
    
    # Analyze distributions per dimension
    print("\n=== Action Distribution Analysis ===")
    print("Dimension | Min      | Max      | Mean     | Std      | Range")
    print("-" * 60)
    
    for dim in range(all_perfect_actions.shape[1]):
        values = all_perfect_actions[:, dim]
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        range_val = max_val - min_val
        
        print(f"{dim:9d} | {min_val:8.4f} | {max_val:8.4f} | {mean_val:8.4f} | {std_val:8.4f} | {range_val:8.4f}")
    
    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Global min: {np.min(all_perfect_actions):.4f}")
    print(f"Global max: {np.max(all_perfect_actions):.4f}")
    print(f"Global mean: {np.mean(all_perfect_actions):.4f}")
    print(f"Global std: {np.std(all_perfect_actions):.4f}")
    
    # Check for clipping evidence
    clipped_low = np.sum(all_perfect_actions <= -1.0)
    clipped_high = np.sum(all_perfect_actions >= 1.0) 
    total_values = all_perfect_actions.size
    
    print(f"\n=== Clipping Analysis ===")
    print(f"Values clipped to -1.0: {clipped_low} ({100*clipped_low/total_values:.2f}%)")
    print(f"Values clipped to 1.0: {clipped_high} ({100*clipped_high/total_values:.2f}%)")
    print(f"Values in valid range (-1, 1): {total_values - clipped_low - clipped_high} ({100*(total_values - clipped_low - clipped_high)/total_values:.2f}%)")
    
    # Show some sample action vectors
    print(f"\n=== Sample Perfect Action Vectors ===")
    for i in range(min(5, len(all_perfect_actions))):
        action_str = ", ".join([f"{val:7.4f}" for val in all_perfect_actions[i]])
        print(f"Sample {i+1}: [{action_str}]")
    
    # Create histograms for a few dimensions
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot histograms for first 6 dimensions
        for dim in range(min(6, all_perfect_actions.shape[1])):
            plt.subplot(2, 3, dim + 1)
            values = all_perfect_actions[:, dim]
            plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Dimension {dim}')
            plt.xlabel('Action Value')
            plt.ylabel('Frequency')
            plt.axvline(-1.0, color='red', linestyle='--', alpha=0.7, label='Clip bounds')
            plt.axvline(1.0, color='red', linestyle='--', alpha=0.7)
            if dim == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('action_distributions.png', dpi=150, bbox_inches='tight')
        print(f"\n=== Histogram saved as action_distributions.png ===")
        
    except ImportError:
        print("Matplotlib not available, skipping histogram plots")
    
    return all_perfect_actions

if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/final_debug_test"
    analyze_action_distributions(dataset_path)
