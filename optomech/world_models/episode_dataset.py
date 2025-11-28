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
        """Filter episodes by minimum length. For SA datasets, reconstruct episodes from transitions."""
        print(f"📋 Reconstructing episodes from SA dataset format (min_length={self.min_episode_length})...")
        
        # Dictionary to group transitions by episode_id
        episode_groups = {}
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # Load episode metadata from file
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        if 'episode_ids' in f and 'episode_steps' in f:
                            # SA dataset format: multiple episodes per file
                            episode_ids = f['episode_ids'][:]
                            episode_steps = f['episode_steps'][:]
                            
                            # Convert episode IDs to strings if they're bytes
                            if len(episode_ids) > 0:
                                if isinstance(episode_ids[0], bytes):
                                    episode_ids = [eid.decode('utf-8') for eid in episode_ids]
                            
                            # Group by episode_id
                            for i, (episode_id, step) in enumerate(zip(episode_ids, episode_steps)):
                                if episode_id not in episode_groups:
                                    episode_groups[episode_id] = []
                                episode_groups[episode_id].append({
                                    'file_path': file_path,
                                    'file_idx': file_idx,
                                    'transition_idx': i,
                                    'step': step
                                })
                        else:
                            # Regular dataset format: assume entire file is one episode
                            episode_length = f[self.obs_key].shape[0]
                            if episode_length >= self.min_episode_length:
                                # Use file path as episode ID for regular datasets
                                episode_id = str(file_path)
                                episode_groups[episode_id] = [{
                                    'file_path': file_path,
                                    'file_idx': file_idx,
                                    'transition_idx': None,  # Load entire file
                                    'step': None,
                                    'episode_length': episode_length
                                }]
                
                elif self.dataset_type == 'npz':
                    data = np.load(file_path)
                    if 'episode_ids' in data and 'episode_steps' in data:
                        # SA dataset format: multiple episodes per file
                        episode_ids = data['episode_ids']
                        episode_steps = data['episode_steps']
                        
                        # Convert to strings if needed
                        if len(episode_ids) > 0:
                            if isinstance(episode_ids[0], bytes):
                                episode_ids = [eid.decode('utf-8') for eid in episode_ids]
                        
                        # Group by episode_id
                        for i, (episode_id, step) in enumerate(zip(episode_ids, episode_steps)):
                            if episode_id not in episode_groups:
                                episode_groups[episode_id] = []
                            episode_groups[episode_id].append({
                                'file_path': file_path,
                                'file_idx': file_idx,
                                'transition_idx': i,
                                'step': step
                            })
                    else:
                        # Regular dataset format: assume entire file is one episode
                        episode_length = data[self.obs_key].shape[0]
                        if episode_length >= self.min_episode_length:
                            episode_id = str(file_path)
                            episode_groups[episode_id] = [{
                                'file_path': file_path,
                                'file_idx': file_idx,
                                'transition_idx': None,
                                'step': None,
                                'episode_length': episode_length
                            }]
                
            except Exception as e:
                print(f"  ⚠️  Error processing {file_path}: {e}")
        
        # Sort transitions within each episode by step number and filter by length
        valid_episode_count = 0
        rejected_episode_count = 0
        for episode_id, transitions in episode_groups.items():
            if transitions[0]['step'] is not None:
                # SA dataset: sort by step number
                transitions.sort(key=lambda x: x['step'])
                episode_length = len(transitions)
            else:
                # Regular dataset: length already computed
                episode_length = transitions[0]['episode_length']
            
            # Only include episodes that meet minimum length requirement
            if episode_length >= self.min_episode_length:
                self.valid_episodes.append((episode_id, transitions, episode_length))
                valid_episode_count += 1
            else:
                rejected_episode_count += 1
        
        print(f"  📊 Reconstructed {len(episode_groups)} episodes from {len(self.file_paths)} files")
        
        # Always show episode length distribution for debugging
        lengths = []
        for episode_id, transitions in episode_groups.items():
            if transitions[0]['step'] is not None:
                lengths.append(len(transitions))
            else:
                lengths.append(transitions[0]['episode_length'])
        
        if lengths:
            print(f"  📊 Episode length stats: min={min(lengths)}, max={max(lengths)}, "
                  f"median={np.median(lengths):.0f}, mean={np.mean(lengths):.1f}")
            print(f"  📊 Filtering with min_episode_length={self.min_episode_length}")
            print(f"  ✅ {valid_episode_count} episodes accepted, {rejected_episode_count} rejected")
            
            # Show some example episode lengths for debugging
            if rejected_episode_count > 0:
                sorted_lengths = sorted(lengths)
                print(f"  📊 First 10 episode lengths: {sorted_lengths[:10]}")
                print(f"  📊 Last 10 episode lengths: {sorted_lengths[-10:]}")
        else:
            print(f"  ✅ {valid_episode_count} episodes meet min_length requirement ({self.min_episode_length})")
        
        if len(episode_groups) > 0 and valid_episode_count == 0:
            print(f"  💡 Consider lowering min_episode_length (currently {self.min_episode_length})")
        
        # Store valid episodes for __getitem__
        self.episode_data = []
        for episode_id, transitions, episode_length in self.valid_episodes:
            self.episode_data.append((episode_id, transitions, episode_length))
        
        print(f"✅ Episode dataset: {len(self.episode_data)} episodes ready")
    
    def __len__(self) -> int:
        return len(self.episode_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Load a complete episode (reconstructed from SA transitions or regular file).
        
        Returns:
            Tuple of (obs, actions, next_obs, episode_length):
                - obs: [episode_length, C, H, W]
                - actions: [episode_length, action_dim]
                - next_obs: [episode_length, C, H, W]
                - episode_length: int
        """
        episode_id, transitions, episode_length = self.episode_data[idx]
        
        try:
            if transitions[0]['transition_idx'] is not None:
                # SA dataset format: reconstruct episode from transitions
                obs_list = []
                actions_list = []
                next_obs_list = []
                
                for transition_info in transitions:
                    file_path = transition_info['file_path']
                    trans_idx = transition_info['transition_idx']
                    
                    # Load this transition
                    if self.dataset_type == 'hdf5':
                        with h5py.File(file_path, 'r') as f:
                            obs_list.append(f[self.obs_key][trans_idx])
                            actions_list.append(f[self.action_key][trans_idx])
                            
                            if 'next_observations' in f:
                                next_obs_list.append(f['next_observations'][trans_idx])
                            else:
                                # For SA datasets, the next_obs is the obs of the next transition
                                # For the last transition, we'll use the same obs (episode ends)
                                if len(next_obs_list) < len(obs_list) - 1:
                                    next_obs_list.append(f[self.obs_key][trans_idx])
                    
                    elif self.dataset_type == 'npz':
                        data = np.load(file_path)
                        obs_list.append(data[self.obs_key][trans_idx])
                        actions_list.append(data[self.action_key][trans_idx])
                        
                        if 'next_observations' in data:
                            next_obs_list.append(data['next_observations'][trans_idx])
                        else:
                            if len(next_obs_list) < len(obs_list) - 1:
                                next_obs_list.append(data[self.obs_key][trans_idx])
                
                # Handle last next_obs for SA datasets without explicit next_observations
                if len(next_obs_list) < len(obs_list):
                    # Use the last obs as the final next_obs (episode termination)
                    next_obs_list.append(obs_list[-1])
                
                # Convert to numpy arrays
                obs = np.array(obs_list)
                actions = np.array(actions_list)
                next_obs = np.array(next_obs_list)
                
            else:
                # Regular dataset format: load entire file
                file_path = transitions[0]['file_path']
                
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        if 'next_observations' in f:
                            obs = f[self.obs_key][:]
                            actions = f[self.action_key][:]
                            next_obs = f['next_observations'][:]
                        else:
                            obs = f[self.obs_key][:-1]
                            next_obs = f[self.obs_key][1:]
                            actions = f[self.action_key][:-1]
                
                elif self.dataset_type == 'npz':
                    data = np.load(file_path)
                    if 'next_observations' in data:
                        obs = data[self.obs_key]
                        actions = data[self.action_key]
                        next_obs = data['next_observations']
                    else:
                        obs = data[self.obs_key][:-1]
                        next_obs = data[self.obs_key][1:]
                        actions = data[self.action_key][:-1]
            
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
            
            # Update episode length after potential truncation
            actual_length = obs.shape[0]
            
            return obs, actions, next_obs, actual_length
            
        except Exception as e:
            print(f"⚠️  Error loading episode {episode_id}: {e}")
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