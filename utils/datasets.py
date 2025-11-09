"""
Modular PyTorch Dataset classes for different training paradigms.

Provides unified dataset interfaces that can be reused across autoencoder training,
supervised learning, behavior cloning, and other ML pipelines.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import json

from .data_loading import DatasetDiscovery, FileLoader, CacheManager
from .transforms import ToTensor, center_crop_transform, normalize_transform


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(self, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Args:
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
        """
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class AutoencoderDataset(BaseDataset):
    """Dataset for autoencoder training (reconstruction tasks)."""
    
    def __init__(self, 
                 observations: List[np.ndarray],
                 transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None):
        """
        Args:
            observations: List of observation arrays
            transform: Optional transform to apply to observations
            input_crop_size: If specified, center crop observations to this size
        """
        super().__init__(transform=transform)
        self.observations = observations
        self.input_crop_size = input_crop_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = ToTensor(normalize=True, input_range='auto')
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observation = self.observations[idx]
        
        # Convert to tensor
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        # Apply cropping if specified
        if self.input_crop_size is not None:
            obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
        
        # For autoencoders, input and target are the same
        return obs_tensor, obs_tensor


class SupervisedDataset(BaseDataset):
    """Dataset for supervised learning (observation-action pairs)."""
    
    def __init__(self,
                 pairs: List[Tuple[np.ndarray, np.ndarray]],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None):
        """
        Args:
            pairs: List of (observation, action) tuples
            transform: Transform to apply to observations
            target_transform: Transform to apply to actions
            input_crop_size: If specified, center crop observations to this size
        """
        super().__init__(transform=transform, target_transform=target_transform)
        self.pairs = pairs
        self.input_crop_size = input_crop_size
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = ToTensor(normalize=True, input_range='auto')
        if self.target_transform is None:
            self.target_transform = ToTensor(normalize=False)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        observation, action = self.pairs[idx]
        
        # Transform observation
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        # Apply cropping to observation if specified
        if self.input_crop_size is not None:
            obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
        
        # Transform action
        if self.target_transform:
            action_tensor = self.target_transform(action)
        else:
            action_tensor = torch.from_numpy(action).float()
        
        return obs_tensor, action_tensor


class BehaviorCloningDataset(SupervisedDataset):
    """Dataset for behavior cloning (alias for SupervisedDataset with better semantics)."""
    
    def __init__(self,
                 expert_demonstrations: List[Tuple[np.ndarray, np.ndarray]],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None):
        """
        Args:
            expert_demonstrations: List of (observation, expert_action) tuples
            transform: Transform to apply to observations
            target_transform: Transform to apply to expert actions
            input_crop_size: If specified, center crop observations to this size
        """
        super().__init__(
            pairs=expert_demonstrations,
            transform=transform,
            target_transform=target_transform,
            input_crop_size=input_crop_size
        )


class LazyDataset(BaseDataset):
    """Memory-efficient dataset that loads data on-demand from files."""
    
    def __init__(self,
                 dataset_path: Union[str, Path],
                 task_type: str = 'autoencoder',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None,
                 max_examples: Optional[int] = None,
                 use_cache: bool = True,
                 log_scale: bool = False,
                 target_action_key: str = 'actions'):
        """
        Args:
            dataset_path: Path to dataset directory
            task_type: Type of task ('autoencoder', 'supervised', 'behavior_cloning')
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
            input_crop_size: If specified, center crop inputs to this size
            max_examples: Optional limit on number of examples
            use_cache: Whether to use caching for faster loading
            log_scale: Whether to apply log-scaling to observations
            target_action_key: Key for target actions in dataset (e.g., 'actions', 'sa_incremental_action')
        """
        super().__init__(transform=transform, target_transform=target_transform)
        
        self.dataset_path = Path(dataset_path)
        self.task_type = task_type
        self.input_crop_size = input_crop_size
        self.max_examples = max_examples
        self.log_scale = log_scale
        self.use_cache = use_cache
        self.target_action_key = target_action_key
        
        # Discover dataset files and build index
        self.file_paths, self.dataset_type, self.metadata = DatasetDiscovery.discover_files(dataset_path)
        self.index_map = self._build_index()
        
        # Set default transforms based on task type
        self._set_default_transforms()
    
    def _set_default_transforms(self):
        """Set default transforms based on task type."""
        from utils.transforms import ToTensor, CenterCrop, LogScale, Compose
        
        if self.transform is None:
            transforms = [ToTensor(normalize=True, input_range='auto')]
            
            if self.input_crop_size is not None:
                transforms.append(CenterCrop(self.input_crop_size))
            
            if self.log_scale:
                transforms.append(LogScale())
            
            self.transform = Compose(transforms)
        
        if self.target_transform is None and self.task_type in ['supervised', 'behavior_cloning']:
            self.target_transform = ToTensor(normalize=False)
    
    def _build_index(self) -> List[Tuple[int, int]]:
        """Build index mapping from global index to (file_idx, local_idx)."""
        index_map = []
        total_loaded = 0
        
        print(f"🗂️  Building dataset index for {len(self.file_paths)} files...")
        if self.max_examples:
            print(f"⚠️  Limiting dataset to first {self.max_examples} examples for debugging")
        
        if self.dataset_type == 'hdf5':
            for file_idx, file_path in enumerate(self.file_paths):
                if file_idx % 1000 == 0:
                    print(f"  📁 Processing file {file_idx}/{len(self.file_paths)}")
                
                try:
                    if self.task_type == 'autoencoder':
                        # Load observations only
                        data = FileLoader.load_hdf5_observations(file_path, ['observations'])
                        if 'observations' in data:
                            num_obs = data['observations'].shape[0]
                            for local_idx in range(num_obs):
                                index_map.append((file_idx, local_idx))
                                total_loaded += 1
                                if self.max_examples and total_loaded >= self.max_examples:
                                    break
                    else:
                        # Load observation-action pairs
                        data = FileLoader.load_hdf5_observations(file_path, ['observations', self.target_action_key])
                        if 'observations' in data and self.target_action_key in data:
                            num_pairs = min(data['observations'].shape[0], data[self.target_action_key].shape[0])
                            for local_idx in range(num_pairs):
                                index_map.append((file_idx, local_idx))
                                total_loaded += 1
                                if self.max_examples and total_loaded >= self.max_examples:
                                    break
                    
                    if self.max_examples and total_loaded >= self.max_examples:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        elif self.dataset_type == 'npz':
            obs_key = self.metadata.get('obs_key', 'observations')
            action_key = self.metadata.get('action_key', 'actions')
            
            for file_idx, file_path in enumerate(self.file_paths):
                try:
                    if self.task_type == 'autoencoder':
                        data = FileLoader.load_npz_observations(file_path, [obs_key])
                        if obs_key in data:
                            num_obs = data[obs_key].shape[0]
                            for local_idx in range(num_obs):
                                index_map.append((file_idx, local_idx))
                                total_loaded += 1
                                if self.max_examples and total_loaded >= self.max_examples:
                                    break
                    else:
                        data = FileLoader.load_npz_observations(file_path, [obs_key, action_key])
                        if obs_key in data and action_key in data:
                            num_pairs = min(data[obs_key].shape[0], data[action_key].shape[0])
                            for local_idx in range(num_pairs):
                                index_map.append((file_idx, local_idx))
                                total_loaded += 1
                                if self.max_examples and total_loaded >= self.max_examples:
                                    break
                    
                    if self.max_examples and total_loaded >= self.max_examples:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        elif self.dataset_type == 'json':
            for file_idx, file_path in enumerate(self.file_paths):
                try:
                    data = FileLoader.load_json_observations(file_path)
                    
                    if 'observations' in data:
                        if self.task_type == 'autoencoder':
                            num_obs = len(data['observations'])
                            for local_idx in range(num_obs):
                                index_map.append((file_idx, local_idx))
                                total_loaded += 1
                                if self.max_examples and total_loaded >= self.max_examples:
                                    break
                        else:
                            if 'actions' in data:
                                num_pairs = min(len(data['observations']), len(data['actions']))
                                for local_idx in range(num_pairs):
                                    index_map.append((file_idx, local_idx))
                                    total_loaded += 1
                                    if self.max_examples and total_loaded >= self.max_examples:
                                        break
                    
                    if self.max_examples and total_loaded >= self.max_examples:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        if self.max_examples and len(index_map) >= self.max_examples:
            print(f"📊 Dataset index built: {len(index_map)} observations ready for lazy loading (LIMITED)")
        else:
            print(f"📊 Dataset index built: {len(index_map)} observations ready for lazy loading")
        return index_map
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """Load single item on demand."""
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
        
        file_idx, local_idx = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        
        try:
            if self.dataset_type == 'hdf5':
                return self._load_hdf5_item(file_path, local_idx)
            elif self.dataset_type == 'npz':
                return self._load_npz_item(file_path, local_idx)
            elif self.dataset_type == 'json':
                return self._load_json_item(file_path, local_idx)
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
                
        except Exception as e:
            print(f"❌ Error loading item {idx} from {file_path}: {e}")
            # Return zero tensors as fallback
            return self._get_fallback_item()
    
    def _load_hdf5_item(self, file_path: Path, local_idx: int):
        """Load single item from HDF5 file."""
        data = FileLoader.load_hdf5_observations(file_path, ['observations', self.target_action_key])
        
        observation = data['observations'][local_idx]
        
        # Transform observation (includes normalization, cropping, and log-scaling if enabled)
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        if self.task_type == 'autoencoder':
            return obs_tensor, obs_tensor
        else:
            # Supervised/behavior cloning
            action = data[self.target_action_key][local_idx]
            if self.target_transform:
                action_tensor = self.target_transform(action)
            else:
                action_tensor = torch.from_numpy(action).float()
            
            return obs_tensor, action_tensor
    
    def _load_npz_item(self, file_path: Path, local_idx: int):
        """Load single item from NPZ file."""
        obs_key = self.metadata.get('obs_key', 'observations')
        action_key = self.metadata.get('action_key', 'actions')
        
        data = FileLoader.load_npz_observations(file_path, [obs_key, action_key])
        
        observation = data[obs_key][local_idx]
        
        # Transform observation (includes normalization, cropping, and log-scaling if enabled)
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        if self.task_type == 'autoencoder':
            return obs_tensor, obs_tensor
        else:
            # Supervised/behavior cloning
            action = data[action_key][local_idx]
            if self.target_transform:
                action_tensor = self.target_transform(action)
            else:
                action_tensor = torch.from_numpy(action).float()
            
            return obs_tensor, action_tensor
    
    def _load_json_item(self, file_path: Path, local_idx: int):
        """Load single item from JSON file."""
        data = FileLoader.load_json_observations(file_path)
        
        observation = np.array(data['observations'][local_idx])
        
        # Transform observation (includes normalization, cropping, and log-scaling if enabled)
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        if self.task_type == 'autoencoder':
            return obs_tensor, obs_tensor
        else:
            # Supervised/behavior cloning
            action = np.array(data['actions'][local_idx])
            if self.target_transform:
                action_tensor = self.target_transform(action)
            else:
                action_tensor = torch.from_numpy(action).float()
            
            return obs_tensor, action_tensor
    
    def _get_fallback_item(self):
        """Return zero tensors as fallback for failed loads."""
        obs_shape = self.metadata.get('observation_shape', (1, 256, 256))
        
        if self.input_crop_size is not None:
            obs_shape = (obs_shape[0], self.input_crop_size, self.input_crop_size)
        
        zero_obs = torch.zeros(obs_shape, dtype=torch.float32)
        
        if self.task_type == 'autoencoder':
            return zero_obs, zero_obs
        else:
            action_shape = self.metadata.get('action_shape', (1,))
            zero_action = torch.zeros(action_shape, dtype=torch.float32)
            return zero_obs, zero_action


class InMemoryDataset(BaseDataset):
    """
    Dataset that loads all data into memory for fastest training.
    Use this when you have sufficient RAM to hold the entire dataset.
    Based on InMemoryAutoencoderDataset from models/train_autoencoder.py.
    """
    
    def __init__(self,
                 dataset_path: Union[str, Path],
                 task_type: str = 'autoencoder',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None,
                 max_examples: Optional[int] = None,
                 log_scale: bool = False,
                 target_action_key: str = 'actions'):
        """
        Args:
            dataset_path: Path to dataset directory
            task_type: Type of task ('autoencoder', 'supervised', 'behavior_cloning')
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
            input_crop_size: If specified, center crop inputs to this size
            max_examples: Optional limit on number of examples
            log_scale: Whether to apply log-scaling to observations
            target_action_key: Key for target actions in dataset (e.g., 'actions', 'sa_incremental_action')
        """
        super().__init__(transform=transform, target_transform=target_transform)
        
        print("\n💾 Loading entire dataset into memory...")
        print("⚠️  This requires sufficient RAM for the full dataset")
        
        import time
        load_start = time.time()
        
        self.dataset_path = Path(dataset_path)
        self.task_type = task_type
        self.input_crop_size = input_crop_size
        self.log_scale = log_scale
        self.target_action_key = target_action_key
        
        # Discover and load all files
        file_paths, dataset_type, metadata = DatasetDiscovery.discover_files(dataset_path)
        
        print(f"📂 Found {len(file_paths)} files of type '{dataset_type}'")
        print(f"📊 Loading data from all files...")
        
        all_observations = []
        all_actions = [] if task_type in ['supervised', 'behavior_cloning'] else None
        total_loaded = 0
        
        if dataset_type == 'hdf5':
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 100 == 0:
                    print(f"  📁 Loading file {file_idx}/{len(file_paths)}")
                
                try:
                    if task_type == 'autoencoder':
                        data = FileLoader.load_hdf5_observations(file_path, ['observations'])
                        if 'observations' in data:
                            obs_data = data['observations']
                            all_observations.append(obs_data)
                            total_loaded += len(obs_data)
                    else:
                        data = FileLoader.load_hdf5_observations(file_path, ['observations', target_action_key])
                        if 'observations' in data and target_action_key in data:
                            obs_data = data['observations']
                            action_data = data[target_action_key]
                            # Make sure they're the same length
                            min_len = min(len(obs_data), len(action_data))
                            all_observations.append(obs_data[:min_len])
                            all_actions.append(action_data[:min_len])
                            total_loaded += min_len
                    
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        elif dataset_type == 'npz':
            obs_key = metadata.get('obs_key', 'observations')
            action_key = metadata.get('action_key', 'actions')
            
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 100 == 0:
                    print(f"  📁 Loading file {file_idx}/{len(file_paths)}")
                
                try:
                    if task_type == 'autoencoder':
                        data = FileLoader.load_npz_observations(file_path, [obs_key])
                        if obs_key in data:
                            obs_data = data[obs_key]
                            all_observations.append(obs_data)
                            total_loaded += len(obs_data)
                    else:
                        data = FileLoader.load_npz_observations(file_path, [obs_key, action_key])
                        if obs_key in data and action_key in data:
                            obs_data = data[obs_key]
                            action_data = data[action_key]
                            min_len = min(len(obs_data), len(action_data))
                            all_observations.append(obs_data[:min_len])
                            all_actions.append(action_data[:min_len])
                            total_loaded += min_len
                    
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        elif dataset_type == 'json':
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 100 == 0:
                    print(f"  📁 Loading file {file_idx}/{len(file_paths)}")
                
                try:
                    data = FileLoader.load_json_observations(file_path)
                    if task_type == 'autoencoder':
                        if 'observations' in data:
                            obs_data = np.array(data['observations'])
                            all_observations.append(obs_data)
                            total_loaded += len(obs_data)
                    else:
                        if 'observations' in data and 'actions' in data:
                            obs_data = np.array(data['observations'])
                            action_data = np.array(data['actions'])
                            min_len = min(len(obs_data), len(action_data))
                            all_observations.append(obs_data[:min_len])
                            all_actions.append(action_data[:min_len])
                            total_loaded += min_len
                    
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        # Concatenate all observations
        print(f"📦 Concatenating {len(all_observations)} arrays...")
        self.observations = np.concatenate(all_observations, axis=0)
        
        if all_actions is not None:
            self.actions = np.concatenate(all_actions, axis=0)
        else:
            self.actions = None
        
        # Limit if needed
        if max_examples and len(self.observations) > max_examples:
            print(f"✂️  Trimming to {max_examples} examples")
            self.observations = self.observations[:max_examples]
            if self.actions is not None:
                self.actions = self.actions[:max_examples]
        
        load_time = time.time() - load_start
        memory_gb = self.observations.nbytes / (1024**3)
        if self.actions is not None:
            memory_gb += self.actions.nbytes / (1024**3)
        
        print(f"✅ Dataset loaded into memory in {load_time:.2f} seconds")
        print(f"   Total observations: {len(self.observations)}")
        print(f"   Observation shape: {self.observations.shape[1:]}")
        if self.actions is not None:
            print(f"   Action shape: {self.actions.shape[1:]}")
        print(f"   Memory usage: {memory_gb:.2f} GB")
        print(f"   Data type: {self.observations.dtype}")
        
        # Setup transforms
        from utils.transforms import ToTensor, CenterCrop, LogScale, Compose
        
        if self.transform is None:
            transforms = [ToTensor(normalize=True, input_range='auto')]
            
            if self.input_crop_size is not None:
                transforms.append(CenterCrop(self.input_crop_size))
            
            if self.log_scale:
                transforms.append(LogScale())
            
            self.transform = Compose(transforms)
        
        if self.target_transform is None and task_type in ['supervised', 'behavior_cloning']:
            self.target_transform = ToTensor(normalize=False)
        
        print(f"🔧 Transform pipeline configured:")
        print(f"   Input crop size: {self.input_crop_size}")
        print(f"   Log scale: {self.log_scale}")
        if self.transform:
            transform_list = self.transform.transforms if hasattr(self.transform, 'transforms') else [self.transform]
            print(f"   Number of transforms: {len(transform_list)}")
            for i, t in enumerate(transform_list):
                print(f"   [{i}] {t.__class__.__name__}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        """Get single item - fast since data is already in memory"""
        observation = self.observations[idx]
        
        # Apply input transforms
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        # Apply manual cropping if transform didn't handle it (fallback)
        if self.input_crop_size is not None and obs_tensor.shape[-1] != self.input_crop_size:
            obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
        
        if self.task_type == 'autoencoder':
            # For autoencoder, target is the same as input
            return obs_tensor, obs_tensor.clone()
        else:
            # For supervised/BC, return observation and action
            action = self.actions[idx]
            if self.target_transform:
                action_tensor = self.target_transform(action)
            else:
                action_tensor = torch.from_numpy(action).float()
            
            return obs_tensor, action_tensor


# Export dataset classes
__all__ = [
    'BaseDataset',
    'AutoencoderDataset',
    'SupervisedDataset', 
    'BehaviorCloningDataset',
    'LazyDataset',
    'InMemoryDataset',
    'TrajectoryDataset',
]

class TrajectoryDataset(BaseDataset):
    """
    Dataset for trajectory-based behavior cloning with LSTMs.
    
    Returns sequences of (observation, action) pairs instead of single transitions.
    Supports variable-length sequences with padding and masking.
    """
    
    def __init__(self,
                 dataset_path: Union[str, Path],
                 sequence_length: int = 20,
                 stride: int = 1,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 input_crop_size: Optional[int] = None,
                 max_examples: Optional[int] = None,
                 log_scale: bool = False,
                 target_action_key: str = 'actions',
                 pad_sequences: bool = True):
        """
        Args:
            dataset_path: Path to dataset directory
            sequence_length: Length of sequences to extract
            stride: Stride for sliding window (1 = overlapping sequences)
            transform: Transform to apply to observations
            target_transform: Transform to apply to actions
            input_crop_size: If specified, crop observations to this size
            max_examples: Optional limit on number of sequences
            log_scale: Whether to apply log-scaling to observations
            target_action_key: Key for target actions in dataset
            pad_sequences: If True, pad short sequences to sequence_length
        """
        super().__init__(transform=transform, target_transform=target_transform)
        
        self.dataset_path = Path(dataset_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.input_crop_size = input_crop_size
        self.max_examples = max_examples
        self.log_scale = log_scale
        self.target_action_key = target_action_key
        self.pad_sequences = pad_sequences
        
        # Discover dataset files and build trajectory index
        self.file_paths, self.dataset_type, self.metadata = DatasetDiscovery.discover_files(dataset_path)
        self.trajectory_index = self._build_trajectory_index()
        
        # Set default transforms
        self._set_default_transforms()
        
        print(f"📊 TrajectoryDataset: {len(self.trajectory_index)} sequences of length {sequence_length}")
    
    def _set_default_transforms(self):
        """Set default transforms based on log_scale setting."""
        from utils.transforms import ToTensor, LogScale, Compose
        
        if self.transform is None:
            transforms = []
            # ToTensor must come first to convert numpy array to tensor
            transforms.append(ToTensor(normalize=True, input_range='auto'))
            # Then apply log scaling if requested
            if self.log_scale:
                transforms.append(LogScale())
            self.transform = Compose(transforms)
        
        if self.target_transform is None:
            self.target_transform = ToTensor(normalize=False)
    
    def _build_trajectory_index(self):
        """Build an index of all available trajectory sequences."""
        trajectory_index = []
        
        for file_path in self.file_paths:
            try:
                # Get number of steps based on dataset type
                if self.dataset_type == 'hdf5':
                    data = FileLoader.load_hdf5_observations(file_path, ['observations', self.target_action_key])
                    if 'observations' in data and self.target_action_key in data:
                        num_steps = min(len(data['observations']), len(data[self.target_action_key]))
                    else:
                        continue
                        
                elif self.dataset_type == 'npz':
                    data = FileLoader.load_npz_observations(file_path, ['observations', self.target_action_key])
                    if 'observations' in data and self.target_action_key in data:
                        num_steps = min(len(data['observations']), len(data[self.target_action_key]))
                    else:
                        continue
                        
                elif self.dataset_type == 'json':
                    data = FileLoader.load_json_observations(file_path)
                    if 'observations' in data and self.target_action_key in data:
                        num_steps = min(len(data['observations']), len(data[self.target_action_key]))
                    else:
                        continue
                else:
                    continue
                
                # Create sliding window sequences for this episode
                for start_idx in range(0, num_steps - self.sequence_length + 1, self.stride):
                    end_idx = start_idx + self.sequence_length
                    trajectory_index.append({
                        'file_path': file_path,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'length': self.sequence_length
                    })
                    
                    if self.max_examples and len(trajectory_index) >= self.max_examples:
                        break
                
                if self.max_examples and len(trajectory_index) >= self.max_examples:
                    break
                    
            except Exception as e:
                print(f"  ⚠️  Skipping {file_path}: {e}")
                continue
        
        return trajectory_index
    
    def __len__(self):
        return len(self.trajectory_index)
    
    def __getitem__(self, idx):
        """Get a sequence of (observation, action) pairs."""
        traj_info = self.trajectory_index[idx]
        file_path = traj_info['file_path']
        start_idx = traj_info['start_idx']
        end_idx = traj_info['end_idx']
        
        # Load data from file based on dataset type
        if self.dataset_type == 'hdf5':
            data = FileLoader.load_hdf5_observations(file_path, ['observations', self.target_action_key])
        elif self.dataset_type == 'npz':
            data = FileLoader.load_npz_observations(file_path, ['observations', self.target_action_key])
        elif self.dataset_type == 'json':
            data = FileLoader.load_json_observations(file_path)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        # Extract observations and actions for this sequence
        observations = data['observations'][start_idx:end_idx]
        actions = data[self.target_action_key][start_idx:end_idx]
        
        # Convert to tensors
        obs_tensors = []
        action_tensors = []
        
        for obs, act in zip(observations, actions):
            # Transform observation
            if self.transform:
                obs_tensor = self.transform(obs)
            else:
                obs_tensor = torch.from_numpy(obs).float()
            
            # Apply cropping if specified
            if self.input_crop_size is not None:
                obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
            
            obs_tensors.append(obs_tensor)
            
            # Transform action
            if self.target_transform:
                action_tensor = self.target_transform(act)
            else:
                action_tensor = torch.from_numpy(act).float()
            
            action_tensors.append(action_tensor)
        
        # Stack into sequences: [seq_len, channels, height, width] and [seq_len, action_dim]
        obs_sequence = torch.stack(obs_tensors, dim=0)
        action_sequence = torch.stack(action_tensors, dim=0)
        
        # Create mask for valid timesteps (all True for full sequences)
        mask = torch.ones(len(obs_tensors), dtype=torch.bool)
        
        return obs_sequence, action_sequence, mask
