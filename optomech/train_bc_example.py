"""
Simple training script example using the RL dataset.
Shows how to train a behavior cloning policy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from dataset_manager import load_dataset


class BehaviorCloningPolicy(nn.Module):
    """Simple behavior cloning policy network."""
    
    def __init__(self, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        
        if len(obs_shape) == 3:  # Image observations (C, H, W)
            self.encoder = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU()
            )
        else:  # Vector observations
            obs_dim = np.prod(obs_shape)
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        if obs.dim() > 2:  # Image observations
            features = self.encoder(obs)
        else:  # Vector observations
            features = self.encoder(obs.flatten(1))
        return self.policy_head(features)


def collate_transitions(batch):
    """Collate function to convert episodes to individual transitions."""
    all_obs = []
    all_actions = []
    all_rewards = []
    
    for episode in batch:
        all_obs.append(episode['observations'])
        all_actions.append(episode['actions'])
        all_rewards.append(episode['rewards'])
    
    # Concatenate all transitions
    obs = torch.cat(all_obs, dim=0)
    actions = torch.cat(all_actions, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    
    return {
        'observations': obs,
        'actions': actions,
        'rewards': rewards
    }


def train_behavior_cloning(dataset_dir, num_epochs=10, batch_size=32, lr=1e-3, 
                          device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train a behavior cloning policy on the dataset."""
    
    print(f"Training on device: {device}")
    
    # Load dataset
    dataset = load_dataset(dataset_dir)
    if len(dataset) == 0:
        print("No episodes found in dataset!")
        return None
    
    print(f"Loaded {len(dataset)} episodes")
    
    # Get data dimensions
    sample_episode = dataset[0]
    obs_shape = sample_episode['observations'].shape[1:]
    action_dim = sample_episode['actions'].shape[1]
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    
    # Create model
    model = BehaviorCloningPolicy(obs_shape, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            
            # Forward pass
            predicted_actions = model(observations)
            loss = criterion(predicted_actions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
    
    return model


def evaluate_policy(model, dataset, device='cpu', num_episodes=5):
    """Evaluate the trained policy on some episodes."""
    
    model.eval()
    total_loss = 0.0
    num_transitions = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for i in range(min(num_episodes, len(dataset))):
            episode = dataset[i]
            observations = episode['observations'].to(device)
            actions = episode['actions'].to(device)
            
            predicted_actions = model(observations)
            loss = criterion(predicted_actions, actions)
            
            total_loss += loss.item() * len(observations)
            num_transitions += len(observations)
            
            print(f"Episode {i+1}: Loss = {loss.item():.6f}, Length = {len(observations)}")
    
    avg_loss = total_loss / num_transitions
    print(f"\nAverage evaluation loss: {avg_loss:.6f}")
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train behavior cloning policy')
    parser.add_argument('--dataset_dir', type=str, default='./rollouts',
                       help='Directory containing the dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--save_model', type=str, default='bc_policy.pth',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset_dir).exists():
        print(f"Dataset directory {args.dataset_dir} does not exist!")
        return
    
    print("=" * 60)
    print("Training Behavior Cloning Policy")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Train model
    model = train_behavior_cloning(
        dataset_dir=args.dataset_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    if model is None:
        return
    
    print("\n" + "=" * 60)
    print("Evaluating trained policy")
    print("=" * 60)
    
    # Evaluate model
    dataset = load_dataset(args.dataset_dir)
    evaluate_policy(model, dataset)
    
    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"\nModel saved to {args.save_model}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"To use the trained policy:")
    print(f"  model = BehaviorCloningPolicy(obs_shape, action_dim)")
    print(f"  model.load_state_dict(torch.load('{args.save_model}'))")
    print(f"  action = model(observation)")


if __name__ == "__main__":
    main()
