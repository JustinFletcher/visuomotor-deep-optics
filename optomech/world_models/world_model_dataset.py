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
from multiprocessing import Pool, cpu_count
from multiprocessing import Pool, cpu_count
from functools import partial


def _load_single_file(file_idx: int, file_path: Path, dataset_type: str, obs_key: str, action_key: str) -> Tuple[int, Optional[Tuple]]:
    """
    Helper function to load a single file in parallel.
    Returns (file_idx, (obs, actions)) or (file_idx, (obs, actions, next_obs)) or (file_idx, None) if error.
    """
    try:
        if dataset_type == 'hdf5':
            with h5py.File(file_path, 'r') as f:
                obs = f[obs_key][:]
                actions = f[action_key][:]
                # Load next_observations if available (new format)
                if 'next_observations' in f:
                    next_obs = f['next_observations'][:]
                    return (file_idx, (obs, actions, next_obs))
                else:
                    return (file_idx, (obs, actions))
        elif dataset_type == 'npz':
            data = np.load(file_path)
            obs = data[obs_key]
            actions = data[action_key]
            # Load next_observations if available (new format)
            if 'next_observations' in data:
                next_obs = data['next_observations']
                return (file_idx, (obs, actions, next_obs))
            else:
                return (file_idx, (obs, actions))
        else:
            return (file_idx, None)
    except Exception as e:
        print(f"  ⚠️  Error loading {file_path}: {e}")
        return (file_idx, None)


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
        max_examples: int = None,
        use_multiprocessing: bool = False
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
            use_multiprocessing: If True, use parallel loading (faster but may have compatibility issues)
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.obs_key = obs_key
        self.action_key = action_key
        self.load_in_memory = load_in_memory
        self.max_examples = max_examples
        self.use_multiprocessing = use_multiprocessing
        
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
                
                # Add non-overlapping sequences from this episode
                # We need sequence_length+1 timesteps (for obs_t and obs_t+1)
                # Reserve one extra timestep for next_obs (unless dataset has explicit next_observations field)
                # Step by sequence_length to avoid redundancy
                for start_idx in range(0, episode_length - self.sequence_length - 1, self.sequence_length):
                    end_idx = start_idx + self.sequence_length
                    # Verify we have room for next_obs: need indices up to end_idx (inclusive)
                    # For next_obs, we'll read obs[start_idx+1:end_idx+1], so max index is end_idx
                    if end_idx < episode_length:  # Ensure we don't read past episode end
                        self.sequence_indices.append((file_idx, start_idx, end_idx))
                
                if file_idx % 100 == 0 and file_idx > 0:
                    print(f"  📁 Processed {file_idx}/{len(self.file_paths)} files, "
                          f"found {len(self.sequence_indices)} sequences")
            
            except Exception as e:
                print(f"  ⚠️  Error processing {file_path}: {e}")
        
        print(f"✅ Sequence index built: {len(self.sequence_indices)} sequences from {len(self.file_paths)} files")
    
    def _load_all_data(self):
        """Load data files that are referenced in sequence indices into memory."""
        import time
        
        # Determine which files are actually needed
        needed_file_indices = set(file_idx for file_idx, _, _ in self.sequence_indices)
        print(f"💾 Preloading data into memory ({len(needed_file_indices)} files needed)...")
        
        # Prepare list of files to load
        files_to_load = [(file_idx, self.file_paths[file_idx]) for file_idx in sorted(needed_file_indices)]
        
        load_start = time.time()
        
        if self.use_multiprocessing:
            # Use parallel loading for faster data loading
            num_workers = min(cpu_count(), len(files_to_load), 16)  # Cap at 16 workers
            print(f"   Using {num_workers} parallel workers for loading...")
            
            # Create partial function with fixed parameters
            load_func = partial(_load_single_file, 
                               dataset_type=self.dataset_type,
                               obs_key=self.obs_key,
                               action_key=self.action_key)
            
            # Load files in parallel with progress updates
            loaded_count = 0
            print(f"   Progress: 0/{len(files_to_load)} files loaded...", end='', flush=True)
            
            with Pool(processes=num_workers) as pool:
                # Use imap_unordered for progress tracking
                for i, (file_idx, data) in enumerate(pool.starmap(load_func, files_to_load), 1):
                    if data is not None:
                        self.data_cache[file_idx] = data
                        loaded_count += 1
                    
                    # Print progress every 100 files or at 10% milestones
                    if i % 100 == 0 or i % max(1, len(files_to_load) // 10) == 0 or i == len(files_to_load):
                        elapsed = time.time() - load_start
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (len(files_to_load) - i) / rate if rate > 0 else 0
                        print(f"\r   Progress: {i}/{len(files_to_load)} files ({100*i/len(files_to_load):.1f}%) | "
                              f"Speed: {rate:.1f} files/s | ETA: {eta/60:.1f} min", end='', flush=True)
            
            print()  # New line after progress
        else:
            # Sequential loading (more compatible, slower)
            print(f"   Using sequential loading (use --use-multiprocessing for faster parallel loading)...")
            loaded_count = 0
            
            for i, (file_idx, file_path) in enumerate(files_to_load, 1):
                try:
                    if self.dataset_type == 'hdf5':
                        with h5py.File(file_path, 'r') as f:
                            obs = f[self.obs_key][:]
                            actions = f[self.action_key][:]
                            # Load next_observations if available (new format)
                            if 'next_observations' in f:
                                next_obs = f['next_observations'][:]
                                self.data_cache[file_idx] = (obs, actions, next_obs)
                            else:
                                self.data_cache[file_idx] = (obs, actions)
                    elif self.dataset_type == 'npz':
                        data = np.load(file_path)
                        obs = data[self.obs_key]
                        actions = data[self.action_key]
                        # Load next_observations if available (new format)
                        if 'next_observations' in data:
                            next_obs = data['next_observations']
                            self.data_cache[file_idx] = (obs, actions, next_obs)
                        else:
                            self.data_cache[file_idx] = (obs, actions)
                    else:
                        continue
                    
                    loaded_count += 1
                    
                    # Print progress every 100 files or at 10% milestones
                    if i % 100 == 0 or i % max(1, len(files_to_load) // 10) == 0 or i == len(files_to_load):
                        elapsed = time.time() - load_start
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (len(files_to_load) - i) / rate if rate > 0 else 0
                        print(f"   Progress: {i}/{len(files_to_load)} files ({100*i/len(files_to_load):.1f}%) | "
                              f"Speed: {rate:.1f} files/s | ETA: {eta/60:.1f} min")
                
                except Exception as e:
                    print(f"  ⚠️  Error loading {file_path}: {e}")
        
        load_time = time.time() - load_start
        
        # Calculate memory usage
        total_memory_gb = 0
        for cached_data in self.data_cache.values():
            if len(cached_data) == 3:
                obs, actions, next_obs = cached_data
                total_memory_gb += (obs.nbytes + actions.nbytes + next_obs.nbytes) / (1024**3)
            else:
                obs, actions = cached_data
                total_memory_gb += (obs.nbytes + actions.nbytes) / (1024**3)
        
        print(f"✅ All data loaded into memory in {load_time:.1f} seconds ({load_time/60:.1f} minutes)")
        print(f"   Loaded {loaded_count}/{len(needed_file_indices)} files successfully")
        print(f"   Total memory usage: {total_memory_gb:.2f} GB")
        print(f"   Average loading speed: {loaded_count/load_time:.1f} files/second")
    
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
            # Check if we have next_observations in the cache (tuple of 3) or just obs and actions (tuple of 2)
            cached_data = self.data_cache[file_idx]
            if len(cached_data) == 3:
                # New format: (obs, actions, next_obs)
                obs_all, actions_all, next_obs_all = cached_data
                obs = obs_all[start_idx:end_idx].copy()
                actions = actions_all[start_idx:end_idx].copy()
                next_obs = next_obs_all[start_idx:end_idx].copy()
            else:
                # Old format: (obs, actions) - compute next_obs from shifted obs
                obs_all, actions_all = cached_data
                obs = obs_all[start_idx:end_idx].copy()
                next_obs = obs_all[start_idx+1:end_idx+1].copy()
                actions = actions_all[start_idx:end_idx].copy()
        else:
            # Load from disk
            file_path = self.file_paths[file_idx]
            
            if self.dataset_type == 'hdf5':
                with h5py.File(file_path, 'r') as f:
                    # Check if next_observations field exists (new format)
                    if 'next_observations' in f:
                        # New format: use explicit next_observations
                        obs = f[self.obs_key][start_idx:end_idx]
                        actions = f[self.action_key][start_idx:end_idx]
                        next_obs = f['next_observations'][start_idx:end_idx]
                    else:
                        # Old format: compute next_obs from shifted observations
                        obs = f[self.obs_key][start_idx:end_idx]
                        next_obs = f[self.obs_key][start_idx+1:end_idx+1]
                        actions = f[self.action_key][start_idx:end_idx]
            
            elif self.dataset_type == 'npz':
                data = np.load(file_path)
                # Check if next_observations field exists (new format)
                if 'next_observations' in data:
                    # New format: use explicit next_observations
                    obs = data[self.obs_key][start_idx:end_idx]
                    actions = data[self.action_key][start_idx:end_idx]
                    next_obs = data['next_observations'][start_idx:end_idx]
                else:
                    # Old format: compute next_obs from shifted observations
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
                
                # Add non-overlapping sequences from this episode
                # Reserve one extra timestep for next_obs (unless dataset has explicit next_observations field)
                # Step by sequence_length to avoid redundancy
                for start_idx in range(0, episode_length - self.sequence_length - 1, self.sequence_length):
                    end_idx = start_idx + self.sequence_length
                    # Verify we have room for next_obs: need indices up to end_idx (inclusive)
                    if end_idx < episode_length:  # Ensure we don't read past episode end
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
                # Check if next_observations field exists (new format)
                if 'next_observations' in f:
                    # New format: use explicit next_observations
                    obs = f[self.obs_key][start_idx:end_idx][:]
                    actions = f[self.action_key][start_idx:end_idx][:]
                    next_obs = f['next_observations'][start_idx:end_idx][:]
                else:
                    # Old format: compute next_obs from shifted observations
                    obs = f[self.obs_key][start_idx:end_idx][:]
                    next_obs = f[self.obs_key][start_idx+1:end_idx+1][:]
                    actions = f[self.action_key][start_idx:end_idx][:]
        
        elif self.dataset_type == 'npz':
            data = np.load(file_path)
            # Check if next_observations field exists (new format)
            if 'next_observations' in data:
                # New format: use explicit next_observations
                obs = data[self.obs_key][start_idx:end_idx]
                actions = data[self.action_key][start_idx:end_idx]
                next_obs = data['next_observations'][start_idx:end_idx]
            else:
                # Old format: compute next_obs from shifted observations
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
