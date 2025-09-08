#!/usr/bin/env python3
"""
Demo script showing how to easily iterate through observation-action pairs for training.
This demonstrates the benefit of the new pairs format vs separate arrays.
"""

import json
import numpy as np
from pathlib import Path

def load_pairs_dataset(dataset_path):
    """
    Load all observation-action pairs from a dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    dataset_path = Path(dataset_path)
    pairs = []
    
    # Find all episode files
    episode_files = list(dataset_path.glob("episode_*.json"))
    print(f"Found {len(episode_files)} episode files")
    
    for episode_file in episode_files:
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        # Extract pairs from metadata (new format)
        if 'sample_pairs' in data['metadata']:
            episode_pairs = data['metadata']['sample_pairs']
            for pair in episode_pairs:
                obs = np.array(pair['observation'], dtype=np.uint16)
                action = np.array(pair['perfect_action'], dtype=np.float32)
                pairs.append((obs, action))
            print(f"  Loaded {len(episode_pairs)} pairs from {episode_file.name}")
        
        # Fallback to legacy format if needed
        elif 'observations' in data and 'perfect_actions' in data:
            observations = data['observations']
            perfect_actions = data['perfect_actions']
            for obs, action in zip(observations, perfect_actions):
                obs = np.array(obs, dtype=np.uint16)
                action = np.array(action, dtype=np.float32)
                pairs.append((obs, action))
            print(f"  Loaded {len(observations)} pairs from {episode_file.name} (legacy format)")
    
    return pairs

def demo_training_iteration(pairs):
    """
    Demonstrate how easy it is to iterate through pairs for training.
    """
    print(f"\n🚀 Training Demo with {len(pairs)} observation-action pairs")
    print("=" * 50)
    
    for i, (observation, perfect_action) in enumerate(pairs):
        print(f"Sample {i+1}:")
        print(f"  Observation shape: {observation.shape}")
        print(f"  Observation dtype: {observation.dtype}")
        print(f"  Observation range: [{observation.min()}, {observation.max()}]")
        print(f"  Perfect action shape: {perfect_action.shape}")
        print(f"  Perfect action dtype: {perfect_action.dtype}")
        if perfect_action.size > 0:
            print(f"  Perfect action range: [{perfect_action.min():.4f}, {perfect_action.max():.4f}]")
            print(f"  Perfect action sample: {perfect_action[:min(3, len(perfect_action))]}...")
        else:
            print(f"  Perfect action: [empty - no corrections needed]")
        print()
        
        # In real training, you would:
        # 1. Convert observation to tensor: torch.from_numpy(observation).float()
        # 2. Convert action to tensor: torch.from_numpy(perfect_action).float() 
        # 3. Forward pass: prediction = model(observation_tensor)
        # 4. Compute loss: loss = criterion(prediction, perfect_action_tensor)
        # 5. Backward pass: loss.backward()
        
        if i >= 2:  # Just show first 3 for demo
            print(f"... and {len(pairs) - 3} more pairs")
            break

if __name__ == "__main__":
    # Demo with the test dataset we just created
    dataset_path = "./datasets/"
    
    print("📂 Loading dataset pairs...")
    pairs = load_pairs_dataset(dataset_path)
    
    if pairs:
        demo_training_iteration(pairs)
        print("\n✅ This shows how easy it is to iterate through observation-action pairs!")
        print("   No need to manually zip separate observation and action arrays.")
        print("   Each iteration gives you exactly what you need: (obs, target_action)")
    else:
        print("❌ No pairs found in dataset")
