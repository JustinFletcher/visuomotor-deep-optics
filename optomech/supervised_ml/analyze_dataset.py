#!/usr/bin/env python3
"""
Quick utility to analyze dataset statistics and verify data quality.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_dataset(dataset_path: str):
    """Analyze dataset and print statistics"""
    dataset_path = Path(dataset_path)
    
    print(f"📊 Analyzing dataset: {dataset_path}")
    print("=" * 50)
    
    episode_files = list(dataset_path.glob("episode_*.json"))
    print(f"Episode files found: {len(episode_files)}")
    
    if not episode_files:
        print("❌ No episode files found!")
        return
    
    # Sample first file to check structure
    with open(episode_files[0], 'r') as f:
        sample_data = json.load(f)
    
    print(f"\nDataset structure:")
    print(f"  Keys: {list(sample_data.keys())}")
    print(f"  Metadata keys: {list(sample_data['metadata'].keys())}")
    
    # Check if using new pairs format
    if 'sample_pairs' in sample_data['metadata']:
        print(f"  ✅ Using new pairs format")
        sample_pairs = sample_data['metadata']['sample_pairs']
        
        if sample_pairs:
            sample_pair = sample_pairs[0]
            obs = np.array(sample_pair['observation'], dtype=np.uint16)
            action = np.array(sample_pair['perfect_action'], dtype=np.float32)
            
            print(f"\nSample data:")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Observation dtype: {obs.dtype}")
            print(f"  Observation range: [{obs.min()}, {obs.max()}]")
            print(f"  Action shape: {action.shape}")
            print(f"  Action dtype: {action.dtype}")
            if action.size > 0:
                print(f"  Action range: [{action.min():.4f}, {action.max():.4f}]")
                print(f"  Action sample: {action[:min(5, len(action))]}...")
            else:
                print(f"  Action: [empty - no corrections needed]")
    else:
        print(f"  ⚠️  Using legacy format")
    
    # Quick statistics
    total_pairs = 0
    valid_pairs = 0
    action_dims = []
    
    print(f"\n🔍 Scanning all files...")
    for i, episode_file in enumerate(episode_files[:10]):  # Check first 10 files
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        if 'sample_pairs' in data['metadata']:
            pairs = data['metadata']['sample_pairs']
            total_pairs += len(pairs)
            
            for pair in pairs:
                action = np.array(pair['perfect_action'], dtype=np.float32)
                if action.size > 0:
                    valid_pairs += 1
                    action_dims.append(len(action))
    
    print(f"\nStatistics (first 10 files):")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Valid pairs (non-empty actions): {valid_pairs}")
    print(f"  Valid pair ratio: {valid_pairs/total_pairs:.1%}")
    
    if action_dims:
        unique_dims = set(action_dims)
        print(f"  Action dimensions found: {unique_dims}")
        print(f"  Most common action dim: {max(set(action_dims), key=action_dims.count)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_dataset.py <dataset_path>")
        sys.exit(1)
    
    analyze_dataset(sys.argv[1])
