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
        min_episode_length: int = 10,
        load_in_memory: bool = False
    ):
        """
        Args:
            file_paths: List of paths to episode files (each file = one episode)
            dataset_type: Type of dataset ('hdf5' or 'npz')
            transforms: Optional transforms to apply to observations
            obs_key: Key for observations in files
            action_key: Key for actions in files
            min_episode_length: Minimum episode length to include
            load_in_memory: If True, preload all episodes into memory for faster access
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.obs_key = obs_key
        self.action_key = action_key
        self.min_episode_length = min_episode_length
        self.load_in_memory = load_in_memory
        
        # Filter episodes by minimum length
        self.valid_episodes = []
        self.total_episodes = 0  # Will be set in _filter_episodes
        self.preloaded_episodes = None  # Will store preloaded data if load_in_memory=True
        self._filter_episodes()
        
        # Preload episodes into memory if requested
        if self.load_in_memory:
            self._preload_episodes()
        
        print(f"✅ Episode dataset: {len(self.valid_episodes)}/{self.total_episodes} episodes "
              f"(min_length={min_episode_length}) from {len(file_paths)} files")
    
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
        
        # Store total episodes count for summary message
        self.total_episodes = len(episode_groups)
        
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
    
    def _preload_episodes(self):
        """Preload all episodes into memory for faster access."""
        import time
        print(f"\n💾 Preloading {len(self.episode_data)} episodes into memory...")
        start_time = time.time()
        
        self.preloaded_episodes = []
        for idx in range(len(self.episode_data)):
            episode_id, transitions, episode_length = self.episode_data[idx]
            
            # Load episode data using existing logic
            obs, actions, next_obs = self._load_episode_from_disk(episode_id, transitions, episode_length)
            
            # Apply transforms if provided (before converting to tensors, as transforms expect numpy)
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
                # Convert to tensors and normalize
                obs = torch.from_numpy(obs).float() / 65535.0
                actions = torch.from_numpy(actions).float()
                next_obs = torch.from_numpy(next_obs).float() / 65535.0
            
            self.preloaded_episodes.append((obs, actions, next_obs, episode_length))
            
            # Progress indicator
            if (idx + 1) % 100 == 0 or idx == len(self.episode_data) - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(self.episode_data) - idx - 1) / rate if rate > 0 else 0
                print(f"  💾 Loaded {idx + 1}/{len(self.episode_data)} episodes "
                      f"({rate:.1f} eps/s, ETA: {eta:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"✅ Preloading complete in {total_time:.1f}s ({len(self.episode_data)/total_time:.1f} eps/s)")
    
    def _load_episode_from_disk(self, episode_id, transitions, episode_length):
        """Load episode data from disk (used by both __getitem__ and preloading)."""
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
        
        return obs, actions, next_obs
    
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
        # If data is preloaded, return directly from memory
        if self.preloaded_episodes is not None:
            return self.preloaded_episodes[idx]
        
        # Otherwise load from disk
        episode_id, transitions, episode_length = self.episode_data[idx]
        
        try:
            obs, actions, next_obs = self._load_episode_from_disk(episode_id, transitions, episode_length)
            
            # Apply transforms if provided (before converting to tensors, as transforms expect numpy)
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
                # Convert to tensors and normalize
                obs = torch.from_numpy(obs).float() / 65535.0
                actions = torch.from_numpy(actions).float()
                next_obs = torch.from_numpy(next_obs).float() / 65535.0
            
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


def collate_episodes_padded(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """
    Collate function that pads variable-length episodes to the max length in the batch.
    This enables proper batched processing for TBPTT (Truncated Backpropagation Through Time).
    
    Args:
        batch: List of (obs, actions, next_obs, episode_length) tuples
    
    Returns:
        Tuple of:
            - obs_padded: Padded observations [batch_size, max_len, C, H, W]
            - actions_padded: Padded actions [batch_size, max_len, action_dim]
            - next_obs_padded: Padded next observations [batch_size, max_len, C, H, W]
            - lengths: Actual episode lengths [batch_size]
            - mask: Binary mask [batch_size, max_len] where 1=valid, 0=padding
    """
    # Filter out empty episodes
    valid_episodes = [(obs, actions, next_obs, length) 
                     for obs, actions, next_obs, length in batch 
                     if length > 0]
    
    if len(valid_episodes) == 0:
        # Return empty batch
        return (torch.empty(0, 0, 3, 64, 64), torch.empty(0, 0, 1), 
                torch.empty(0, 0, 3, 64, 64), torch.tensor([]), torch.empty(0, 0))
    
    # Find max episode length in this batch
    max_length = max(length for _, _, _, length in valid_episodes)
    batch_size = len(valid_episodes)
    
    # Get tensor shapes from first episode
    obs_shape = valid_episodes[0][0].shape[1:]  # (C, H, W)
    action_dim = valid_episodes[0][1].shape[1]
    
    # Initialize padded tensors with zeros
    obs_padded = torch.zeros(batch_size, max_length, *obs_shape)
    actions_padded = torch.zeros(batch_size, max_length, action_dim)
    next_obs_padded = torch.zeros(batch_size, max_length, *obs_shape)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    # Fill in actual episode data
    for i, (obs, actions, next_obs, length) in enumerate(valid_episodes):
        obs_padded[i, :length] = obs
        actions_padded[i, :length] = actions
        next_obs_padded[i, :length] = next_obs
        lengths[i] = length
        mask[i, :length] = True  # Mark valid timesteps
    
    return obs_padded, actions_padded, next_obs_padded, lengths, mask