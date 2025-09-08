"""
Example script showing how to load and use the RL dataset with PyTorch.
This demonstrates the seamless data loading for training models.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset_manager import load_dataset, get_dataset_stats
from pathlib import Path


def collate_episodes(batch):
    """Custom collate function to handle variable-length episodes."""
    # For now, just return the first episode in the batch
    # In practice, you might want to pad sequences or use other strategies
    return batch[0]


def demonstrate_dataset_loading(dataset_dir):
    """Demonstrate how to load and use the dataset."""
    
    # Check if dataset exists
    if not Path(dataset_dir).exists():
        print(f"Dataset directory {dataset_dir} does not exist!")
        return
    
    # Get dataset statistics
    print("=== Dataset Statistics ===")
    stats = get_dataset_stats(dataset_dir)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    # Load the dataset (limit to 3 episodes for faster testing)
    print("=== Loading Dataset ===")
    print("Loading first 3 episodes for demonstration (use max_episodes=None for all episodes)")
    dataset = load_dataset(dataset_dir, max_episodes=3)
    print(f"Loaded episodes: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No episodes found in dataset!")
        return
    
    # Examine a single episode
    print("\n=== Episode Structure ===")
    episode = dataset[0]
    print(f"Episode keys: {list(episode.keys())}")
    print(f"Observations shape: {episode['observations'].shape}")
    print(f"Actions shape: {episode['actions'].shape}")
    print(f"Rewards shape: {episode['rewards'].shape}")
    
    # Check for perfect_action and best_action
    if 'perfect_actions' in episode:
        print(f"Perfect actions shape: {episode['perfect_actions'].shape}")
    else:
        print("Perfect actions: Not available")
        
    if 'best_actions' in episode:
        print(f"Best actions shape: {episode['best_actions'].shape}")
    else:
        print("Best actions: Not available")
    
    print(f"Episode length: {len(episode['observations'])}")
    print(f"Total reward: {episode['rewards'].sum().item():.4f}")
    print(f"Mean reward: {episode['rewards'].mean().item():.4f}")
    print(f"Metadata: {episode['metadata']}")
    
    # Create a DataLoader for batch processing
    print("\n=== Creating DataLoader ===")
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Start with batch_size=1 for episode data
        shuffle=True,
        collate_fn=collate_episodes
    )
    
    # Iterate through a few episodes
    print("\n=== Sample Episodes ===")
    for i, episode in enumerate(dataloader):
        if i >= 3:  # Just show first 3 episodes
            break
            
        print(f"Episode {i+1}:")
        print(f"  Shape: obs={episode['observations'].shape}, actions={episode['actions'].shape}")
        print(f"  Reward: total={episode['rewards'].sum():.4f}, mean={episode['rewards'].mean():.4f}")
        print(f"  Reward function: {episode['metadata'].get('reward_function', 'unknown')}")
        print()
    
    # Demonstrate converting to training data
    print("=== Converting to Training Data ===")
    all_observations = []
    all_actions = []
    all_rewards = []
    all_perfect_actions = []
    
    for episode in dataset:
        all_observations.append(episode['observations'])
        all_actions.append(episode['actions'])
        all_rewards.append(episode['rewards'])
        
        # Collect perfect actions if available
        if 'perfect_actions' in episode:
            all_perfect_actions.append(episode['perfect_actions'])
    
    # Concatenate all episodes
    if all_observations:
        combined_obs = torch.cat(all_observations, dim=0)
        combined_actions = torch.cat(all_actions, dim=0)
        combined_rewards = torch.cat(all_rewards, dim=0)
        
        print(f"Combined dataset shapes:")
        print(f"  Observations: {combined_obs.shape}")
        print(f"  Actions: {combined_actions.shape}")
        print(f"  Rewards: {combined_rewards.shape}")
        print(f"  Total transitions: {len(combined_obs)}")
        
        # Perfect actions for regression
        if all_perfect_actions:
            combined_perfect_actions = torch.cat(all_perfect_actions, dim=0)
            print(f"  Perfect actions: {combined_perfect_actions.shape}")
            print(f"  -> Perfect for supervised learning: observations -> perfect_actions")
        else:
            print(f"  Perfect actions: Not available")
        
        # Example: Create input-target pairs for supervised learning
        if len(combined_obs) > 1:
            inputs = combined_obs[:-1]  # All but last observation
            targets = combined_actions[1:]  # All but first action
            print(f"  Sequential training pairs: {len(inputs)} (obs_t -> action_t+1)")
            
            if all_perfect_actions:
                perfect_targets = combined_perfect_actions[:-1]  # All but last perfect action
                print(f"  Perfect action pairs: {len(inputs)} (obs_t -> perfect_action_t)")
    
    print("\n=== Ready for Training! ===")
    print("You can now use this data to train your models:")
    print("1. Use individual episodes for sequence modeling")
    print("2. Combine all transitions for supervised learning")
    print("3. Use episodes for offline RL algorithms") 
    if all_perfect_actions:
        print("4. Use observations -> perfect_actions for imitation learning")


def create_simple_model_example(dataset_dir):
    """Example of how you might create a simple model using this data."""
    
    dataset = load_dataset(dataset_dir)
    if len(dataset) == 0:
        print("No data available for model creation example")
        return
    
    print("\n=== Simple Model Example ===")
    
    # Get data dimensions from first episode
    sample_episode = dataset[0]
    obs_shape = sample_episode['observations'].shape[1:]  # Remove time dimension
    action_dim = sample_episode['actions'].shape[1]
    has_perfect_actions = 'perfect_actions' in sample_episode
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Perfect actions available: {has_perfect_actions}")
    
    # Simple neural network example
    class SimplePolicy(torch.nn.Module):
        def __init__(self, obs_shape, action_dim):
            super().__init__()
            
            # For image observations, use CNN
            if len(obs_shape) == 3:  # (C, H, W)
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(obs_shape[0], 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(8),
                    torch.nn.Flatten(),
                    torch.nn.Linear(32 * 8 * 8, 256),
                    torch.nn.ReLU()
                )
                self.decoder = torch.nn.Linear(256, action_dim)
            else:
                # For vector observations, use MLP
                obs_dim = np.prod(obs_shape)
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU()
                )
                self.decoder = torch.nn.Linear(256, action_dim)
        
        def forward(self, obs):
            features = self.encoder(obs.flatten(1))
            return self.decoder(features)
    
    # Create model
    model = SimplePolicy(obs_shape, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    if has_perfect_actions:
        print("\n=== Perfect Action Regression Setup ===")
        print("This dataset contains perfect_actions - ideal for imitation learning!")
        print("Training setup:")
        print("  Input: observations")
        print("  Target: perfect_actions") 
        print("  Loss: MSE between model(obs) and perfect_actions")
        print("  This trains the model to directly imitate the perfect policy")
        
        # Show how to set up training loop
        print("\nExample training loop:")
        print("""
        for episode in dataset:
            observations = episode['observations']
            perfect_actions = episode['perfect_actions']
            
            predicted_actions = model(observations)
            loss = torch.nn.functional.mse_loss(predicted_actions, perfect_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        """)
    else:
        print("Perfect actions not available - using regular action data")
    
    print("Ready to train with your dataset!")


if __name__ == "__main__":
    import sys
    
    # Default to ./rollouts if no argument provided
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./rollouts"
    
    print(f"Loading dataset from: {dataset_dir}")
    print("=" * 50)
    
    demonstrate_dataset_loading(dataset_dir)
    create_simple_model_example(dataset_dir)
    
    print("\n" + "=" * 50)
    print("Dataset loading example complete!")
    print("\nTo use with your own training script:")
    print(f"  from dataset_manager import load_dataset")
    print(f"  dataset = load_dataset('{dataset_dir}')")
    print(f"  # Then use dataset with PyTorch DataLoader")
