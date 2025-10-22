#!/usr/bin/env python3
"""
Quick utility to check SA dataset size and statistics.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import os


def format_bytes(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def analyze_sa_dataset(dataset_path: str):
    """Analyze SA dataset size and contents"""
    dataset_path = Path(dataset_path)
    
    print(f"📊 SA Dataset Size Analysis")
    print(f"Dataset path: {dataset_path}")
    print("=" * 60)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Find all HDF5 batch files
    h5_files = list(dataset_path.glob("*.h5"))
    
    if not h5_files:
        print("❌ No HDF5 batch files found!")
        return
    
    print(f"📁 Found {len(h5_files)} batch files")
    
    total_size = 0
    total_samples = 0
    total_episodes = set()
    
    # Analyze each file
    for h5_file in h5_files:
        file_size = h5_file.stat().st_size
        total_size += file_size
        
        with h5py.File(h5_file, 'r') as f:
            num_samples = f['observations'].shape[0]
            total_samples += num_samples
            
            # Count unique episodes
            episode_ids = f['episode_ids'][:]
            for ep_id in episode_ids:
                if isinstance(ep_id, bytes):
                    ep_id = ep_id.decode('utf-8')
                total_episodes.add(ep_id)
        
        print(f"  📄 {h5_file.name}: {format_bytes(file_size)} ({num_samples} samples)")
    
    print(f"\n📈 Dataset Summary:")
    print(f"  Total size on disk: {format_bytes(total_size)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total episodes: {len(total_episodes)}")
    print(f"  Average samples per episode: {total_samples / len(total_episodes):.1f}")
    print(f"  Average file size: {format_bytes(total_size / len(h5_files))}")
    print(f"  Bytes per sample: {total_size / total_samples:.1f}")
    
    # Analyze first file for data structure
    if h5_files:
        print(f"\n🔍 Data Structure (from {h5_files[0].name}):")
        with h5py.File(h5_files[0], 'r') as f:
            print(f"  Available datasets: {list(f.keys())}")
            
            # Check data shapes and types
            obs_shape = f['observations'].shape
            sa_actions_shape = f['sa_actions'].shape
            perfect_actions_shape = f['perfect_actions'].shape
            
            print(f"  Observations: {obs_shape} ({f['observations'].dtype})")
            print(f"  SA actions: {sa_actions_shape} ({f['sa_actions'].dtype})")
            print(f"  Perfect actions: {perfect_actions_shape} ({f['perfect_actions'].dtype})")
            
            # Memory usage estimates
            obs_bytes = np.prod(obs_shape) * np.dtype(f['observations'].dtype).itemsize
            sa_bytes = np.prod(sa_actions_shape) * np.dtype(f['sa_actions'].dtype).itemsize
            perfect_bytes = np.prod(perfect_actions_shape) * np.dtype(f['perfect_actions'].dtype).itemsize
            
            total_sample_bytes = obs_bytes + sa_bytes + perfect_bytes
            
            print(f"\n💾 Memory Usage per sample:")
            print(f"  Observations: {format_bytes(obs_bytes / obs_shape[0])}")
            print(f"  SA actions: {format_bytes(sa_bytes / sa_actions_shape[0])}")
            print(f"  Perfect actions: {format_bytes(perfect_bytes / perfect_actions_shape[0])}")
            print(f"  Total per sample: {format_bytes(total_sample_bytes / obs_shape[0])}")
            
            # Estimate uncompressed size
            uncompressed_estimate = total_sample_bytes * (total_samples / obs_shape[0])
            compression_ratio = uncompressed_estimate / total_size
            print(f"\n🗜️  Compression Analysis:")
            print(f"  Estimated uncompressed: {format_bytes(uncompressed_estimate)}")
            print(f"  Compression ratio: {compression_ratio:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Check SA dataset size and statistics")
    parser.add_argument("dataset_path", help="Path to SA dataset directory")
    
    args = parser.parse_args()
    analyze_sa_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
