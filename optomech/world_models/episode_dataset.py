#!/usr/bin/env python3
"""
Episode-based dataset for proper temporal learning in world models.

Instead of extracting fixed-length sequences, this dataset yields full episodes
and handles sequence chunking during training to maintain temporal continuity.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import h5py

class WorldModelEpisodeDataset(Dataset):
    """
    Dataset that yields full episodes for proper temporal learning.
    
    Each item is a complete episode, allowing the training loop to:
    1. Reset hidden state at episode start
    2. Process episode in chunks with carried hidden state
    3. Apply BPTT across longer temporal sequences
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        dataset_type: str,
        transforms=None,
        obs_key: str = 'observations',
        action_key: str = 'actions',
        min_episode_length: int = 10
    ):
        """
        Args:
            file_paths: List of paths to episode files (each file = one episode)
            dataset_type: Type of dataset ('hdf5' or 'npz')
            transforms: Optional transforms to apply to observations
            obs_key: Key for observations in files
            action_key: Key for actions in files
            min_episode_length: Minimum episode length to include
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.obs_key = obs_key
        self.action_key = action_key
        self.min_episode_length = min_episode_length
        
        # Filter episodes by minimum length
        self.valid_episodes = []
        self._filter_episodes()
        
        print(f"✅ Episode dataset: {len(self.valid_episodes)}/{len(file_paths)} episodes "
              f"(min_length={min_episode_length})")
    
    def _filter_episodes(self):
        """Filter episodes by minimum length."""
        print(f"📋 Filtering episodes by minimum length ({self.min_episode_length})...")
        
        for file_path in self.file_paths:
            try:
                # Get episode length
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        episode_length = f[self.obs_key].shape[0]
                elif self.dataset_type == 'npz':
                    data = np.load(file_path, mmap_mode='r')
                    episode_length = data[self.obs_key].shape[0]
                else:
                    continue
                
                # Only include episodes with sufficient length
                if episode_length >= self.min_episode_length:
                    self.valid_episodes.append((file_path, episode_length))
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {file_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.valid_episodes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Load a complete episode.
        
        Returns:
            Tuple of (obs, actions, next_obs, episode_length):
                - obs: [episode_length, C, H, W]
                - actions: [episode_length, action_dim]
                - next_obs: [episode_length, C, H, W]
                - episode_length: int
        """
        file_path, episode_length = self.valid_episodes[idx]
        
        try:
            if self.dataset_type == 'hdf5':
                with h5py.File(file_path, 'r') as f:
                    if 'next_observations' in f:
                        # New format: use explicit next_observations
                        obs = f[self.obs_key][:]
                        actions = f[self.action_key][:]
                        next_obs = f['next_observations'][:]
                    else:
                        # Old format: compute next_obs from shifted observations
                        obs = f[self.obs_key][:-1]  # Remove last observation
                        next_obs = f[self.obs_key][1:]  # Shifted observations
                        actions = f[self.action_key][:-1]  # Remove last action
            
            elif self.dataset_type == 'npz':
                data = np.load(file_path)
                if 'next_observations' in data:
                    # New format: use explicit next_observations
                    obs = data[self.obs_key]
                    actions = data[self.action_key]
                    next_obs = data['next_observations']
                else:
                    # Old format: compute next_obs from shifted observations
                    obs = data[self.obs_key][:-1]  # Remove last observation
                    next_obs = data[self.obs_key][1:]  # Shifted observations
                    actions = data[self.action_key][:-1]  # Remove last action
            
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
            # Apply transforms if provided
            if self.transforms is not None:
                obs_list = []
                next_obs_list = []
                for i in range(len(obs)):
                    obs_i = self.transforms(obs[i])
                    next_obs_i = self.transforms(next_obs[i])
                    obs_list.append(obs_i)
                    next_obs_list.append(next_obs_i)
                obs = torch.stack(obs_list)
                next_obs = torch.stack(next_obs_list)
                actions = torch.from_numpy(actions).float()
            else:
                # Convert to tensors
                obs = torch.from_numpy(obs).float()
                next_obs = torch.from_numpy(next_obs).float()
                actions = torch.from_numpy(actions).float()
            
            # Update episode length after potential truncation in old format
            actual_length = obs.shape[0]
            
            return obs, actions, next_obs, actual_length
            
        except Exception as e:
            print(f"⚠️  Error loading episode {file_path}: {e}")
            # Return empty episode on error
            return (torch.empty(0, 3, 64, 64), torch.empty(0, 1), 
                   torch.empty(0, 3, 64, 64), 0)


def collate_episodes(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """
    Collate function for episodes. Since episodes have different lengths,
    we return them as a list rather than trying to stack them.
    
    Args:
        batch: List of (obs, actions, next_obs, episode_length) tuples
    
    Returns:
        List of episodes: [(obs, actions, next_obs, episode_length), ...]
    """
    # Filter out empty episodes
    valid_episodes = [(obs, actions, next_obs, length) 
                     for obs, actions, next_obs, length in batch 
                     if length > 0]
    
    return valid_episodes