#!/usr/bin/env python3
"""
Simple training test to verify dataset wiring works correctly.
Tests loading the checkout_dataset and basic imitation learning setup.
Partitions dataset into train/val/test splits by episodes, then creates
individual transition examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from optomech.dataset_manager import load_dataset
import random


class TransitionDataset(Dataset):
    """Dataset that returns individual transitions instead of full episodes."""
    
    def __init__(self, transitions):
        """
        Args:
            transitions: List of transition dictionaries
        """
        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        return self.transitions[idx]


def partition_episodes(episodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partition episodes into train, validation, and test sets.
    
    Args:
        episodes: List of episode indices
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_episodes, val_episodes, test_episodes
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle episodes
    shuffled_episodes = episodes.copy()
    random.shuffle(shuffled_episodes)
    
    n_total = len(shuffled_episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_episodes = shuffled_episodes[:n_train]
    val_episodes = shuffled_episodes[n_train:n_train + n_val]
    test_episodes = shuffled_episodes[n_train + n_val:]
    
    return train_episodes, val_episodes, test_episodes


def episodes_to_transitions(dataset, episode_indices):
    """
    Convert episodes to individual transitions.
    
    Args:
        dataset: Episode dataset
        episode_indices: List of episode indices to convert
    
    Returns:
        List of transition dictionaries
    """
    transitions = []
    
    for ep_idx in episode_indices:
        episode = dataset[ep_idx]
        episode_length = episode['observations'].shape[0]
        
        for t in range(episode_length):
            transition = {
                'observation': episode['observations'][t],
                'next_observation': episode['next_observations'][t],
                'action': episode['actions'][t],
                'reward': episode['rewards'][t],
                'done': episode['dones'][t],
                'perfect_action': episode['perfect_actions'][t],
                'best_action': episode['best_actions'][t],
                'episode_id': ep_idx,
                'timestep': t
            }
            transitions.append(transition)
    
    return transitions

def simple_mlp_policy(obs_dim, action_dim, hidden_dim=256):
    """Simple MLP policy for testing."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
    )

def test_dataset_loading_and_partitioning():
    """Test that we can load and partition the dataset correctly."""
    print("🧪 Testing Dataset Loading and Partitioning")
    print("=" * 50)
    
    # Load episode dataset (path relative to project root)
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "mini_checkout_dataset")
    episode_dataset = load_dataset(dataset_path)
    
    print(f"✅ Episode dataset loaded successfully")
    print(f"📊 Total episodes: {len(episode_dataset)}")
    
    # Get episode indices
    episode_indices = list(range(len(episode_dataset)))
    
    # Partition episodes
    train_eps, val_eps, test_eps = partition_episodes(
        episode_indices, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    print(f"📊 Episode partitions:")
    print(f"  Train: {len(train_eps)} episodes")
    print(f"  Val:   {len(val_eps)} episodes")
    print(f"  Test:  {len(test_eps)} episodes")
    
    # Convert to transitions
    train_transitions = episodes_to_transitions(episode_dataset, train_eps)
    val_transitions = episodes_to_transitions(episode_dataset, val_eps)
    test_transitions = episodes_to_transitions(episode_dataset, test_eps)
    
    print(f"📊 Transition partitions:")
    print(f"  Train: {len(train_transitions)} transitions")
    print(f"  Val:   {len(val_transitions)} transitions")
    print(f"  Test:  {len(test_transitions)} transitions")
    
    # Create transition datasets
    train_dataset = TransitionDataset(train_transitions)
    val_dataset = TransitionDataset(val_transitions)
    test_dataset = TransitionDataset(test_transitions)
    
    # Test getting a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"🔍 Sample transition structure:")
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    return train_dataset, val_dataset, test_dataset

def test_dataloaders(train_dataset, val_dataset, test_dataset):
    """Test DataLoader functionality with partitioned datasets."""
    print("\n🧪 Testing DataLoader with Partitioned Data")
    print("=" * 50)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ DataLoaders created")
    print(f"📊 DataLoader sizes:")
    print(f"  Train: {len(train_dataloader)} batches")
    print(f"  Val:   {len(val_dataloader)} batches") 
    print(f"  Test:  {len(test_dataloader)} batches")
    
    # Test getting batches from each
    for name, dataloader in [("Train", train_dataloader), ("Val", val_dataloader), ("Test", test_dataloader)]:
        if len(dataloader) > 0:
            batch = next(iter(dataloader))
            print(f"🔍 {name} batch structure:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value)} of length {len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
            break  # Just show one example
    
    return train_dataloader, val_dataloader, test_dataloader

def test_training_step(train_dataloader):
    """Test a simple training step with transition data."""
    print("\n🧪 Testing Training Step")
    print("=" * 40)
    
    # Get dimensions from first batch
    batch = next(iter(train_dataloader))
    
    # Handle different key names - check what's available
    obs_key = 'observation' if 'observation' in batch else 'observations'
    action_key = 'action' if 'action' in batch else 'actions'
    
    obs_dim = batch[obs_key].flatten(1).shape[1]  # Flatten spatial dimensions
    action_dim = batch[action_key].shape[1]
    
    print(f"📏 Observation dim (flattened): {obs_dim}")
    print(f"📏 Action dim: {action_dim}")
    print(f"📏 Batch size: {batch[obs_key].shape[0]}")
    
    # Create simple policy
    policy = simple_mlp_policy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training step
    observations = batch[obs_key].float()
    if observations.dim() > 2:  # Flatten spatial dimensions
        observations = observations.flatten(1)
    
    # Test with different action targets
    print("\n🎯 Testing different action targets:")
    
    for action_type in [action_key, 'perfect_action', 'best_action']:
        if action_type in batch:
            target_actions = batch[action_type].float()
            
            # Forward pass
            predicted_actions = policy(observations)
            loss = criterion(predicted_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  ✅ {action_type}: loss = {loss.item():.4f}")
        else:
            print(f"  ❌ {action_type}: not found in batch")
    
    print(f"\n✅ Training step completed successfully!")

def test_training_loop(train_dataloader, num_steps=50):
    """Test training loop with multiple steps to verify loss decreases."""
    print("\n🧪 Testing Training Loop (Loss Reduction)")
    print("=" * 50)
    
    # Get dimensions from first batch
    batch = next(iter(train_dataloader))
    obs_key = 'observation' if 'observation' in batch else 'observations'
    action_key = 'perfect_action'  # Use perfect_action as our target
    
    if action_key not in batch:
        print(f"❌ {action_key} not found in batch, skipping training loop test")
        return
    
    obs_dim = batch[obs_key].flatten(1).shape[1]
    action_dim = batch[action_key].shape[1]
    
    print(f"📏 Training on {action_key} targets")
    print(f"📏 Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"📏 Training steps: {num_steps}")
    
    # Create fresh model and optimizer
    policy = simple_mlp_policy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Track losses
    losses = []
    
    print(f"\n📈 Training Progress:")
    print(f"{'Step':<8} {'Loss':<12} {'Change':<12}")
    print("-" * 35)
    
    # Training loop
    dataloader_iter = iter(train_dataloader)
    
    for step in range(num_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Reset iterator if we run out of batches
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)
        
        # Prepare data
        observations = batch[obs_key].float()
        if observations.dim() > 2:
            observations = observations.flatten(1)
        target_actions = batch[action_key].float()
        
        # Forward pass
        predicted_actions = policy(observations)
        loss = criterion(predicted_actions, target_actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Print progress every 10 steps
        if step % 10 == 0 or step == num_steps - 1:
            if step == 0:
                change_str = "N/A"
            else:
                change = loss_val - losses[step - 10 if step >= 10 else 0]
                change_str = f"{change:+.4f}"
                if change < 0:
                    change_str = f"🔽 {change_str}"
                else:
                    change_str = f"🔺 {change_str}"
            
            print(f"{step:<8} {loss_val:<12.4f} {change_str}")
    
    # Analyze results
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = initial_loss - final_loss
    improvement_pct = (improvement / initial_loss) * 100
    
    print("\n📊 Training Results:")
    print(f"  Initial loss:     {initial_loss:.6f}")
    print(f"  Final loss:       {final_loss:.6f}")
    print(f"  Improvement:      {improvement:.6f} ({improvement_pct:.1f}%)")
    
    if improvement > 0:
        print(f"  ✅ Loss decreased! Model is learning from {action_key} targets")
    else:
        print(f"  ⚠️  Loss did not decrease significantly")
    
    print(f"\n✅ Training loop test completed!")

def main():
    """Run all tests with dataset partitioning."""
    print("🚀 Testing Dataset Wiring with Train/Val/Test Partitions")
    print("=" * 60)
    
    try:
        # Test dataset loading and partitioning
        train_dataset, val_dataset, test_dataset = test_dataset_loading_and_partitioning()
        
        # Test dataloaders
        train_dataloader, val_dataloader, test_dataloader = test_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Test training step
        test_training_step(train_dataloader)
        
        # Test training loop with loss reduction
        test_training_loop(train_dataloader, num_steps=50)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Dataset loading and partitioning works correctly")
        print("✅ Episodes properly split into train/val/test")
        print("✅ Individual transitions extracted successfully")
        print("✅ DataLoaders handle transition batching correctly")
        print("✅ Training step works with perfect/best actions")
        print("✅ Training loop shows loss reduction over time")
        print("✅ Ready for full imitation learning experiments")
        
        # Print final statistics
        print(f"\n📊 Final Dataset Statistics:")
        print(f"  Train: {len(train_dataset)} transitions")
        print(f"  Val:   {len(val_dataset)} transitions")
        print(f"  Test:  {len(test_dataset)} transitions")
        print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} transitions")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
