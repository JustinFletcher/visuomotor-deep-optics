#!/usr/bin/env python3
"""
Simple script to plot histogram of observation pixel values from an SML episode file
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_observation_histogram(episode_file: Path, bins: int = 100):
    """
    Read an episode file and plot histogram of all observation pixel values
    
    Args:
        episode_file: Path to the episode JSON file
        bins: Number of histogram bins
    """
    print(f"Reading episode file: {episode_file}")
    
    # Load the episode data
    with open(episode_file, 'r') as f:
        data = json.load(f)
    
    # Extract observations
    if 'episode_data' in data:
        episode_data = data['episode_data']
    else:
        episode_data = data
    
    if 'observations' not in episode_data:
        print("❌ No 'observations' found in episode data")
        print(f"Available keys: {list(episode_data.keys())}")
        return
    
    observations = episode_data['observations']
    print(f"Found {len(observations)} observations")
    
    # Convert to numpy array and flatten all pixel values
    obs_array = np.array(observations)
    print(f"Observation shape: {obs_array.shape}")
    
    # Flatten to get all pixel values
    all_pixels = obs_array.flatten()
    print(f"Total pixel values: {len(all_pixels):,}")
    
    # Print basic statistics
    print(f"\nPixel value statistics:")
    print(f"  Min: {all_pixels.min():.6f}")
    print(f"  Max: {all_pixels.max():.6f}")
    print(f"  Mean: {all_pixels.mean():.6f}")
    print(f"  Std: {all_pixels.std():.6f}")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Main histogram
    plt.subplot(1, 2, 1)
    plt.hist(all_pixels, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of All Observation Pixel Values\n({len(all_pixels):,} total pixels)')
    plt.grid(True, alpha=0.3)
    
    # Log scale histogram for better visualization if there are outliers
    plt.subplot(1, 2, 2)
    plt.hist(all_pixels, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency (log scale)')
    plt.title('Histogram (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = episode_file.parent / f"observation_histogram_{episode_file.stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Histogram saved to: {output_file}")
    
    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of observation pixel values from SML episode file")
    parser.add_argument("episode_file", type=str, help="Path to episode JSON file")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins (default: 100)")
    
    args = parser.parse_args()
    
    episode_file = Path(args.episode_file)
    if not episode_file.exists():
        print(f"❌ File not found: {episode_file}")
        return 1
    
    try:
        plot_observation_histogram(episode_file, args.bins)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
