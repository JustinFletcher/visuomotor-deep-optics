#!/usr/bin/env python3
"""
World Model Dataset

Dataset for loading sequences of (observation, action, next_observation) tuples
for training recurrent world models with BPTT.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import h5py


class WorldModelSequenceDataset(Dataset):
    """
    Dataset that loads sequences for world model training with BPTT.
    
    Each sample is a sequence of (obs_t, action_t, obs_t+1) tuples.
    Supports HDF5 and NPZ file formats.
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        dataset_type: str,
        sequence_length: int = 10,
        transforms=None,
        obs_key: str = 'observations',
        action_key: str = 'actions',
        load_in_memory: bool = False,
        max_examples: int = None
    ):
        """
        Args:
            file_paths: List of paths to dataset files
            dataset_type: Type of dataset ('hdf5' or 'npz')
            sequence_length: Length of sequences to extract
            transforms: Optional transforms to apply to observations
            obs_key: Key for observations in files
            action_key: Key for actions in files
            load_in_memory: If True, preload all data into memory
            max_examples: Maximum number of sequences to use (for debugging/testing)
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.obs_key = obs_key
        self.action_key = action_key
        self.load_in_memory = load_in_memory
        self.max_examples = max_examples
        
        # Build index of valid sequence start positions
        self.sequence_indices = []  # List of (file_idx, start_idx, end_idx)
        self._build_sequence_index()
        
        # Apply max_examples limit if specified
        if self.max_examples is not None and len(self.sequence_indices) > self.max_examples:
            print(f"⚠️  Limiting dataset to {self.max_examples} sequences (from {len(self.sequence_indices)})")
            self.sequence_indices = self.sequence_indices[:self.max_examples]
        
        # Optionally load all data into memory
        self.data_cache = {}  # {file_idx: (obs, actions)}
        if self.load_in_memory:
            self._load_all_data()
    
    def __getstate__(self):
        """Custom pickle method for multiprocessing compatibility."""
        # Return state without the data_cache to avoid pickling large arrays
        state = self.__dict__.copy()
        # Store whether we had a cache
        state['_had_cache'] = bool(state['data_cache'])
        # Don't pickle the actual cache
        state['data_cache'] = {}
        return state
    
    def __setstate__(self, state):
        """Custom unpickle method for multiprocessing compatibility."""
        # Restore state
        self.__dict__.update(state)
        # If we had a cache, reload it in the worker process
        if state.get('_had_cache', False) and self.load_in_memory:
            self._load_all_data()
    
    def _build_sequence_index(self):
        """Build index of valid sequence positions across all files."""
        print(f"📇 Building sequence index (seq_len={self.sequence_length})...")
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # Get episode length
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        episode_length = f[self.obs_key].shape[0]
                elif self.dataset_type == 'npz':
                    data = np.load(file_path, mmap_mode='r')
                    episode_length = data[self.obs_key].shape[0]
                else:
                    print(f"  ⚠️  Unsupported dataset type: {self.dataset_type}")
                    continue
                
                # Add all valid sequence start positions for this episode
                # We need sequence_length+1 timesteps (for obs_t and obs_t+1)
                num_sequences = max(0, episode_length - self.sequence_length)
                
                for start_idx in range(num_sequences):
                    end_idx = start_idx + self.sequence_length
                    self.sequence_indices.append((file_idx, start_idx, end_idx))
                
                if file_idx % 100 == 0 and file_idx > 0:
                    print(f"  📁 Processed {file_idx}/{len(self.file_paths)} files, "
                          f"found {len(self.sequence_indices)} sequences")
            
            except Exception as e:
                print(f"  ⚠️  Error processing {file_path}: {e}")
        
        print(f"✅ Sequence index built: {len(self.sequence_indices)} sequences from {len(self.file_paths)} files")
    
    def _load_all_data(self):
        """Load data files that are referenced in sequence indices into memory."""
        # Determine which files are actually needed
        needed_file_indices = set(file_idx for file_idx, _, _ in self.sequence_indices)
        print(f"💾 Preloading data into memory ({len(needed_file_indices)} files needed)...")
        
        loaded_count = 0
        for file_idx in needed_file_indices:
            try:
                file_path = self.file_paths[file_idx]
                
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        obs = f[self.obs_key][:]
                        actions = f[self.action_key][:]
                elif self.dataset_type == 'npz':
                    data = np.load(file_path)
                    obs = data[self.obs_key]
                    actions = data[self.action_key]
                else:
                    continue
                
                # Store in cache
                self.data_cache[file_idx] = (obs, actions)
                loaded_count += 1
                
                if loaded_count % 100 == 0:
                    print(f"  💾 Loaded {loaded_count}/{len(needed_file_indices)} files into memory")
            
            except Exception as e:
                print(f"  ⚠️  Error loading {file_path}: {e}")
        
        print(f"✅ All data loaded into memory ({len(self.data_cache)} files)")
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence sample.
        
        Args:
            idx: Index of sequence to retrieve
        
        Returns:
            Tuple of:
                - obs: Observations [seq_len, channels, height, width]
                - actions: Actions [seq_len, action_dim]
                - next_obs: Next observations [seq_len, channels, height, width]
        """
        file_idx, start_idx, end_idx = self.sequence_indices[idx]
        
        # Load sequence data (from cache if available)
        if self.load_in_memory and file_idx in self.data_cache:
            # Use cached data - make copies to avoid shared memory issues in multiprocessing
            obs_all, actions_all = self.data_cache[file_idx]
            obs = obs_all[start_idx:end_idx].copy()
            next_obs = obs_all[start_idx+1:end_idx+1].copy()
            actions = actions_all[start_idx:end_idx].copy()
        else:
            # Load from disk
            file_path = self.file_paths[file_idx]
            
            if self.dataset_type == 'hdf5':
                with h5py.File(file_path, 'r') as f:
                    # Load obs_t (start_idx to end_idx)
                    obs = f[self.obs_key][start_idx:end_idx]
                    # Load obs_t+1 (start_idx+1 to end_idx+1)
                    next_obs = f[self.obs_key][start_idx+1:end_idx+1]
                    # Load actions (start_idx to end_idx)
                    actions = f[self.action_key][start_idx:end_idx]
            
            elif self.dataset_type == 'npz':
                data = np.load(file_path)
                obs = data[self.obs_key][start_idx:end_idx]
                next_obs = data[self.obs_key][start_idx+1:end_idx+1]
                actions = data[self.action_key][start_idx:end_idx]
            
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Apply transforms if provided (before converting to tensors)
        if self.transforms is not None:
            # Apply transforms to each frame in sequence
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
        
        return obs, actions, next_obs


class WorldModelLazyDataset(Dataset):
    """
    Memory-efficient dataset that loads sequences on-demand without keeping
    file handles open. Uses caching for frequently accessed sequences.
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        dataset_type: str,
        sequence_length: int = 10,
        transforms=None,
        obs_key: str = 'observations',
        action_key: str = 'actions',
        cache_size: int = 100
    ):
        """
        Args:
            file_paths: List of paths to dataset files
            dataset_type: Type of dataset ('hdf5' or 'npz')
            sequence_length: Length of sequences to extract
            transforms: Optional transforms to apply to observations
            obs_key: Key for observations in files
            action_key: Key for actions in files
            cache_size: Number of sequences to cache in memory
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.obs_key = obs_key
        self.action_key = action_key
        
        # Simple cache: store most recent sequences
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
        
        # Build index of valid sequence start positions
        self.sequence_indices = []
        self._build_sequence_index()
    
    def _build_sequence_index(self):
        """Build index of valid sequence positions across all files."""
        print(f"📇 Building sequence index (seq_len={self.sequence_length})...")
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # Get episode length without loading full data
                if self.dataset_type == 'hdf5':
                    with h5py.File(file_path, 'r') as f:
                        episode_length = f[self.obs_key].shape[0]
                elif self.dataset_type == 'npz':
                    data = np.load(file_path, mmap_mode='r')
                    episode_length = data[self.obs_key].shape[0]
                else:
                    print(f"  ⚠️  Unsupported dataset type: {self.dataset_type}")
                    continue
                
                # Add all valid sequence start positions
                num_sequences = max(0, episode_length - self.sequence_length)
                
                for start_idx in range(num_sequences):
                    end_idx = start_idx + self.sequence_length
                    self.sequence_indices.append((file_idx, start_idx, end_idx))
                
                if file_idx % 100 == 0 and file_idx > 0:
                    print(f"  📁 Processed {file_idx}/{len(self.file_paths)} files, "
                          f"found {len(self.sequence_indices)} sequences")
            
            except Exception as e:
                print(f"  ⚠️  Error processing {file_path}: {e}")
        
        print(f"✅ Sequence index built: {len(self.sequence_indices)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def _load_sequence_from_file(
        self,
        file_idx: int,
        start_idx: int,
        end_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load sequence data from file."""
        file_path = self.file_paths[file_idx]
        
        if self.dataset_type == 'hdf5':
            with h5py.File(file_path, 'r') as f:
                obs = f[self.obs_key][start_idx:end_idx][:]
                next_obs = f[self.obs_key][start_idx+1:end_idx+1][:]
                actions = f[self.action_key][start_idx:end_idx][:]
        
        elif self.dataset_type == 'npz':
            data = np.load(file_path)
            obs = data[self.obs_key][start_idx:end_idx]
            next_obs = data[self.obs_key][start_idx+1:end_idx+1]
            actions = data[self.action_key][start_idx:end_idx]
        
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        return obs, actions, next_obs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sequence sample with caching."""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from file
        file_idx, start_idx, end_idx = self.sequence_indices[idx]
        obs, actions, next_obs = self._load_sequence_from_file(file_idx, start_idx, end_idx)
        
        # Convert to tensors
        obs = torch.from_numpy(obs).float()
        next_obs = torch.from_numpy(next_obs).float()
        actions = torch.from_numpy(actions).float()
        
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
        
        result = (obs, actions, next_obs)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
        
        self.cache[idx] = result
        self.cache_order.append(idx)
        
        return result


def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching sequences.
    
    Args:
        batch: List of (obs, actions, next_obs) tuples
    
    Returns:
        Tuple of batched tensors:
            - obs: [batch, seq_len, channels, height, width]
            - actions: [batch, seq_len, action_dim]
            - next_obs: [batch, seq_len, channels, height, width]
    """
    obs, actions, next_obs = zip(*batch)
    
    obs = torch.stack(obs)  # [batch, seq_len, C, H, W]
    actions = torch.stack(actions)  # [batch, seq_len, action_dim]
    next_obs = torch.stack(next_obs)  # [batch, seq_len, C, H, W]
    
    return obs, actions, next_obs
