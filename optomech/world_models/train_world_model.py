#!/usr/bin/env python3
"""
World Model Training Script

Trains recurrent world models for next-observation prediction given current observations and actions.
World models use pretrained autoencoder components with an LSTM for temporal modeling.

Features:
- Uses pretrained autoencoder encoder/decoder components
- LSTM for temporal state representation
- Action encoding and fusion with state
- BPTT (Backpropagation Through Time) for sequence training
- Next observation prediction
- Comprehensive logging and validation
- Model persistence with metadata
- Resume training from checkpoints
- Multi-GPU support
"""

import os
import sys
import json
import random
import argparse
import uuid
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import pickle

# Set CUDA memory allocator config to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

# Add workspace root to path for imports
# This file is at: workspace_root/optomech/world_models/train_world_model.py
# So we need to go up 2 levels to get to workspace root
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Import model architectures and utilities
from models import create_model, AutoEncoderCNN, AutoEncoderResNet
from models.model_utils import save_trained_model, ModelSaver
from models.world_model import WorldModel, create_world_model_from_autoencoder

# Import unified dataset utilities
from utils.datasets import AutoencoderDataset, LazyDataset
from utils.data_loading import DatasetDiscovery, FileLoader, CacheManager
from utils.transforms import center_crop_transform, get_autoencoder_transforms

# Import world model dataset
from optomech.world_models.world_model_dataset import WorldModelSequenceDataset, WorldModelLazyDataset, collate_sequences
from optomech.world_models.episode_dataset import WorldModelEpisodeDataset, collate_episodes, collate_episodes_padded

# Optional HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("⚠️  h5py not available, HDF5 files cannot be loaded")


def discover_dataset_files(dataset_path: str) -> Tuple[List[Path], str, Dict]:
    """
    Discover dataset files without loading them into memory.
    Returns file paths and metadata for lazy loading.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (file_paths, dataset_type, metadata)
    """
    dataset_path = Path(dataset_path)
    
    print(f"📂 Discovering dataset files in: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Try different dataset formats
    
    # 1. Check for HDF5 files
    hdf5_files = list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
    if hdf5_files and HDF5_AVAILABLE:
        print(f"🔍 Found {len(hdf5_files)} HDF5 files")
        
        # Get sample data info from first file
        sample_file = hdf5_files[0]
        try:
            with h5py.File(sample_file, 'r') as f:
                obs_shape = f['observations'].shape
                obs_dtype = f['observations'].dtype
                print(f"📏 Sample observation shape: {obs_shape[1:]}")
                print(f"📈 Data type: {obs_dtype}")
                
                # Count total observations across all files
                total_obs = 0
                for hdf5_file in hdf5_files:
                    try:
                        with h5py.File(hdf5_file, 'r') as f2:
                            total_obs += f2['observations'].shape[0]
                    except Exception as e:
                        print(f"  ⚠️  Could not read {hdf5_file}: {e}")
                
                metadata = {
                    'total_observations': total_obs,
                    'observation_shape': obs_shape[1:],
                    'dtype': obs_dtype,
                    'files_per_sample': 1  # One observation per file access
                }
                
                print(f"📊 Total observations discovered: {total_obs}")
                return sorted(hdf5_files), 'hdf5', metadata
                
        except Exception as e:
            print(f"  ❌ Error reading sample file {sample_file}: {e}")
            raise
    
    # 2. Check for NPZ files  
    elif list(dataset_path.glob("*.npz")):
        npz_files = list(dataset_path.glob("*.npz"))
        print(f"🔍 Found {len(npz_files)} NPZ files")
        
        # Get sample data info from first file
        sample_file = npz_files[0]
        try:
            data = np.load(sample_file)
            if 'observations' in data:
                obs_key = 'observations'
            elif 'obs' in data:
                obs_key = 'obs'
            else:
                obs_keys = [k for k in data.keys() if 'obs' in k.lower()]
                if obs_keys:
                    obs_key = obs_keys[0]
                else:
                    raise ValueError(f"No observation data found in {sample_file}")
            
            obs_shape = data[obs_key].shape
            obs_dtype = data[obs_key].dtype
            print(f"📏 Sample observation shape: {obs_shape[1:]}")
            print(f"📈 Data type: {obs_dtype}")
            
            # Count total observations
            total_obs = 0
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    if obs_key in data:
                        total_obs += data[obs_key].shape[0]
                except Exception as e:
                    print(f"  ⚠️  Could not read {npz_file}: {e}")
            
            metadata = {
                'total_observations': total_obs,
                'observation_shape': obs_shape[1:],
                'dtype': obs_dtype,
                'obs_key': obs_key
            }
            
            print(f"📊 Total observations discovered: {total_obs}")
            return sorted(npz_files), 'npz', metadata
            
        except Exception as e:
            print(f"  ❌ Error reading sample file {sample_file}: {e}")
            raise
    
    # 3. Check for JSON files (individual episodes)
    elif list(dataset_path.glob("episode_*.json")):
        json_files = list(dataset_path.glob("episode_*.json"))
        print(f"🔍 Found {len(json_files)} JSON episode files")
        
        # Get sample data info from first file
        sample_file = json_files[0]
        try:
            with open(sample_file, 'r') as f:
                episode_data = json.load(f)
            
            if 'observations' not in episode_data:
                raise ValueError(f"No observations found in {sample_file}")
            
            obs_array = np.array(episode_data['observations'])
            obs_shape = obs_array.shape
            obs_dtype = obs_array.dtype
            print(f"📏 Sample observation shape: {obs_shape[1:]}")
            print(f"📈 Data type: {obs_dtype}")
            
            # Count total observations
            total_obs = 0
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        episode_data = json.load(f)
                    if 'observations' in episode_data:
                        total_obs += len(episode_data['observations'])
                except Exception as e:
                    print(f"  ⚠️  Could not read {json_file}: {e}")
            
            metadata = {
                'total_observations': total_obs,
                'observation_shape': obs_shape[1:],
                'dtype': obs_dtype
            }
            
            print(f"📊 Total observations discovered: {total_obs}")
            return sorted(json_files), 'json', metadata
            
        except Exception as e:
            print(f"  ❌ Error reading sample file {sample_file}: {e}")
            raise
    
    else:
        raise ValueError(f"No supported dataset files found in {dataset_path}")


def load_data_from_dataset(dataset_path: str, max_examples: int = None) -> List[np.ndarray]:
    """
    DEPRECATED: This function loads all data into memory and will crash with large datasets.
    Use LazyAutoencoderDataset instead for memory-efficient loading.
    
    This function is kept for compatibility but should not be used with large datasets.
    """
    print("⚠️  WARNING: load_data_from_dataset loads all data into memory!")
    print("   For large datasets, use LazyAutoencoderDataset instead.")
    
    # For small datasets or debugging, still support this
    if max_examples and max_examples > 10000:
        raise ValueError(f"load_data_from_dataset not recommended for {max_examples} examples. Use LazyAutoencoderDataset.")
    
    dataset_path = Path(dataset_path)
    observations = []
    
    print(f"📂 Loading data from: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Try different dataset formats
    
    # 1. Check for HDF5 files
    hdf5_files = list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
    if hdf5_files and HDF5_AVAILABLE:
        print(f"🔍 Found {len(hdf5_files)} HDF5 files")
        for hdf5_file in hdf5_files[:10]:  # Limit to first 10 files for safety
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    obs_data = f['observations'][:]
                    observations.extend(obs_data)
                    print(f"  ✅ Loaded {len(obs_data)} observations from {hdf5_file.name}")
                    if max_examples and len(observations) >= max_examples:
                        observations = observations[:max_examples]
                        break
            except Exception as e:
                print(f"  ❌ Error loading {hdf5_file}: {e}")
    
    # 2. Check for NPZ files  
    elif list(dataset_path.glob("*.npz")):
        npz_files = list(dataset_path.glob("*.npz"))
        print(f"🔍 Found {len(npz_files)} NPZ files")
        for npz_file in npz_files[:10]:  # Limit to first 10 files for safety
            try:
                data = np.load(npz_file)
                if 'observations' in data:
                    obs_data = data['observations']
                elif 'obs' in data:
                    obs_data = data['obs']
                else:
                    # Try to find observation data
                    obs_keys = [k for k in data.keys() if 'obs' in k.lower()]
                    if obs_keys:
                        obs_data = data[obs_keys[0]]
                    else:
                        print(f"  ⚠️  No observation data found in {npz_file}")
                        continue
                
                observations.extend(obs_data)
                print(f"  ✅ Loaded {len(obs_data)} observations from {npz_file.name}")
                if max_examples and len(observations) >= max_examples:
                    observations = observations[:max_examples]
                    break
            except Exception as e:
                print(f"  ❌ Error loading {npz_file}: {e}")
    
    # 3. Check for JSON files (individual episodes)
    elif list(dataset_path.glob("episode_*.json")):
        json_files = list(dataset_path.glob("episode_*.json"))
        print(f"🔍 Found {len(json_files)} JSON episode files")
        for json_file in json_files[:10]:  # Limit to first 10 files for safety
            try:
                with open(json_file, 'r') as f:
                    episode_data = json.load(f)
                
                if 'observations' in episode_data:
                    obs_data = np.array(episode_data['observations'])
                    observations.extend(obs_data)
                    print(f"  ✅ Loaded {len(obs_data)} observations from {json_file.name}")
                    
                if max_examples and len(observations) >= max_examples:
                    observations = observations[:max_examples]
                    break
            except Exception as e:
                print(f"  ❌ Error loading {json_file}: {e}")
    
    else:
        raise ValueError(f"No supported dataset files found in {dataset_path}")
    
    if not observations:
        raise ValueError("No observations loaded from dataset")
    
    print(f"📊 Total observations loaded: {len(observations)}")
    
    # Convert to numpy arrays and check format
    observations = [np.array(obs) for obs in observations]
    
    # Print data info
    sample_obs = observations[0]
    print(f"📏 Observation shape: {sample_obs.shape}")
    print(f"📈 Data type: {sample_obs.dtype}")
    print(f"📊 Value range: [{sample_obs.min():.3f}, {sample_obs.max():.3f}]")
    
    return observations


class LazyAutoencoderDataset(Dataset):
    """Memory-efficient PyTorch Dataset that loads observations on-demand"""
    
    def __init__(self, file_paths: List[Path], dataset_type: str, metadata: Dict, 
                 max_examples: int = None, transform=None, input_crop_size: int = None):
        """
        Args:
            file_paths: List of paths to dataset files
            dataset_type: Type of dataset ('hdf5', 'npz', 'json')
            metadata: Dataset metadata from discover_dataset_files
            max_examples: Optional limit on number of examples
            transform: Optional transform to apply to observations
            input_crop_size: If specified, center crop observations to this size
        """
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.metadata = metadata
        self.transform = transform
        self.input_crop_size = input_crop_size
        
        # Build index mapping from global index to (file_idx, local_idx)
        self.index_map = []
        total_loaded = 0
        
        print(f"🗂️  Building dataset index for {len(file_paths)} files...")
        
        if dataset_type == 'hdf5':
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 1000 == 0:  # Progress update every 1000 files
                    print(f"  📁 Processing file {file_idx}/{len(file_paths)}: {file_path.name}")
                try:
                    with h5py.File(file_path, 'r') as f:
                        num_obs = f['observations'].shape[0]
                        for local_idx in range(num_obs):
                            self.index_map.append((file_idx, local_idx))
                            total_loaded += 1
                            if max_examples and total_loaded >= max_examples:
                                break
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
                    
        elif dataset_type == 'npz':
            obs_key = metadata['obs_key']
            for file_idx, file_path in enumerate(file_paths):
                try:
                    data = np.load(file_path)
                    if obs_key in data:
                        num_obs = data[obs_key].shape[0]
                        for local_idx in range(num_obs):
                            self.index_map.append((file_idx, local_idx))
                            total_loaded += 1
                            if max_examples and total_loaded >= max_examples:
                                break
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
                    
        elif dataset_type == 'json':
            for file_idx, file_path in enumerate(file_paths):
                try:
                    with open(file_path, 'r') as f:
                        episode_data = json.load(f)
                    if 'observations' in episode_data:
                        num_obs = len(episode_data['observations'])
                        for local_idx in range(num_obs):
                            self.index_map.append((file_idx, local_idx))
                            total_loaded += 1
                            if max_examples and total_loaded >= max_examples:
                                break
                    if max_examples and total_loaded >= max_examples:
                        break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        print(f"📊 Dataset index built: {len(self.index_map)} observations ready for lazy loading")
        
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """Load single observation on demand"""
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
            
        file_idx, local_idx = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        
        try:
            if self.dataset_type == 'hdf5':
                with h5py.File(file_path, 'r') as f:
                    observation = f['observations'][local_idx]
                    
            elif self.dataset_type == 'npz':
                data = np.load(file_path)
                obs_key = self.metadata['obs_key']
                observation = data[obs_key][local_idx]
                
            elif self.dataset_type == 'json':
                with open(file_path, 'r') as f:
                    episode_data = json.load(f)
                observation = np.array(episode_data['observations'][local_idx])
                
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
            # Convert to torch tensor
            if observation.dtype == np.uint16:
                obs_tensor = torch.from_numpy(observation).float() / 65535.0
            else:
                obs_tensor = torch.from_numpy(observation).float()
                # Normalize to [0, 1] if not already
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / obs_tensor.max()
            
            # Apply center cropping if specified
            if self.input_crop_size is not None:
                from models.models import center_crop_transform
                obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
            
            if self.transform:
                obs_tensor = self.transform(obs_tensor)
                
            return obs_tensor, obs_tensor  # Return same tensor as input and target
            
        except Exception as e:
            print(f"❌ Error loading observation {idx} from {file_path}: {e}")
            # Return a zero tensor as fallback
            shape = self.metadata['observation_shape']
            zero_tensor = torch.zeros(shape, dtype=torch.float32)
            return zero_tensor, zero_tensor


# AutoencoderDataset is now imported from utils.datasets (unified utilities)
# Old local definition removed to avoid shadowing the import


def huber_loss(predictions, targets, delta=1.0):
    """Huber loss (robust to outliers)"""
    residual = torch.abs(predictions - targets)
    condition = residual < delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * residual - 0.5 * delta ** 2
    return torch.where(condition, squared_loss, linear_loss).mean()


def create_loss_function(loss_type: str, huber_delta: float = 0.1):
    """Create loss function"""
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


class InMemoryAutoencoderDataset(Dataset):
    """
    Dataset that loads all data into memory for fastest training.
    Use this when you have sufficient RAM to hold the entire dataset.
    """
    
    def __init__(self, 
                 dataset_path: Union[str, Path],
                 input_crop_size: Optional[int] = None,
                 max_examples: Optional[int] = None,
                 log_scale: bool = False):
        """
        Args:
            dataset_path: Path to dataset directory
            input_crop_size: If specified, center crop inputs to this size
            max_examples: Optional limit on number of examples
            log_scale: Whether to apply log-scaling to observations
        """
        print("\n💾 Loading entire dataset into memory...")
        print("⚠️  This requires sufficient RAM for the full dataset")
        
        load_start = time.time()
        
        self.input_crop_size = input_crop_size
        self.log_scale = log_scale
        
        # Discover and load all files
        file_paths, dataset_type, metadata = DatasetDiscovery.discover_files(dataset_path)
        
        print(f"📂 Found {len(file_paths)} files of type '{dataset_type}'")
        print(f"📊 Loading data from all files...")
        
        all_observations = []
        total_loaded = 0
        
        if dataset_type == 'hdf5':
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 100 == 0:
                    print(f"  📁 Loading file {file_idx}/{len(file_paths)}")
                
                try:
                    data = FileLoader.load_hdf5_observations(file_path, ['observations'])
                    if 'observations' in data:
                        obs_data = data['observations']
                        all_observations.append(obs_data)
                        total_loaded += len(obs_data)
                        
                        if max_examples and total_loaded >= max_examples:
                            break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        elif dataset_type == 'npz':
            obs_key = metadata.get('obs_key', 'observations')
            for file_idx, file_path in enumerate(file_paths):
                if file_idx % 100 == 0:
                    print(f"  📁 Loading file {file_idx}/{len(file_paths)}")
                
                try:
                    data = FileLoader.load_npz_observations(file_path, [obs_key])
                    if obs_key in data:
                        obs_data = data[obs_key]
                        all_observations.append(obs_data)
                        total_loaded += len(obs_data)
                        
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
                    if 'observations' in data:
                        obs_data = np.array(data['observations'])
                        all_observations.append(obs_data)
                        total_loaded += len(obs_data)
                        
                        if max_examples and total_loaded >= max_examples:
                            break
                except Exception as e:
                    print(f"  ⚠️  Skipping {file_path}: {e}")
        
        # Concatenate all observations
        print(f"📦 Concatenating {len(all_observations)} arrays...")
        self.observations = np.concatenate(all_observations, axis=0)
        
        # Limit if needed
        if max_examples and len(self.observations) > max_examples:
            print(f"✂️  Trimming to {max_examples} examples")
            self.observations = self.observations[:max_examples]
        
        load_time = time.time() - load_start
        memory_gb = self.observations.nbytes / (1024**3)
        
        print(f"✅ Dataset loaded into memory in {load_time:.2f} seconds")
        print(f"   Total observations: {len(self.observations)}")
        print(f"   Observation shape: {self.observations.shape[1:]}")
        print(f"   Memory usage: {memory_gb:.2f} GB")
        print(f"   Data type: {self.observations.dtype}")
        
        # Setup transforms
        from utils.transforms import ToTensor, CenterCrop, LogScale, Compose
        
        transforms = [ToTensor(normalize=True, input_range='auto')]
        
        if self.input_crop_size is not None:
            transforms.append(CenterCrop(self.input_crop_size))
        
        if self.log_scale:
            transforms.append(LogScale())
        
        self.transform = Compose(transforms)
        
        print(f"🔧 Transform pipeline configured:")
        print(f"   Input crop size: {self.input_crop_size}")
        print(f"   Log scale: {self.log_scale}")
        print(f"   Number of transforms: {len(transforms)}")
        for i, t in enumerate(transforms):
            print(f"   [{i}] {t.__class__.__name__}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        """Get single item - fast since data is already in memory"""
        observation = self.observations[idx]
        
        # Apply transforms
        if self.transform:
            obs_tensor = self.transform(observation)
        else:
            obs_tensor = torch.from_numpy(observation).float()
        
        # Apply manual cropping if transform didn't work (fallback)
        if self.input_crop_size is not None and obs_tensor.shape[-1] != self.input_crop_size:
            # Import here to avoid circular imports
            from utils.transforms import center_crop_transform
            obs_tensor = center_crop_transform(obs_tensor, self.input_crop_size)
        
        # For autoencoder, target is the same as input (return same transformed tensor for both)
        # This ensures both input and target have the same shape after cropping
        return obs_tensor, obs_tensor.clone()


def create_loss_function(loss_type: str, huber_delta: float = 0.1):
    """
    Factory function to create loss functions for autoencoder training.
    
    Args:
        loss_type: Type of loss function ('mse', 'mae', 'smooth_l1', 'huber')
        huber_delta: Delta parameter for Huber loss
        
    Returns:
        Loss function
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'huber':
        return lambda pred, target: huber_loss(pred, target, delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: mse, mae, smooth_l1, huber")


@dataclass
class WorldModelConfig:
    """Configuration for world model training"""
    dataset_path: str = "datasets/sml_100k_dataset"
    run_name: str = "world_model_default"  # Name for this training run
    runs_dir: str = "runs"  # Root directory for runs (default: ./runs)
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05
    device: str = "auto"  # auto, cuda, mps, cpu
    save_model: bool = True
    model_save_path: str = "saved_models/world_model.pth"
    plot_losses: bool = True
    seed: int = 42
    resume_from: str = None  # Path to checkpoint to resume from
    start_epoch: int = 0  # Starting epoch (for resumed training)
    
    # Model architecture settings
    pretrained_autoencoder_path: str = None  # Path to pretrained autoencoder
    hidden_dim: int = 512  # LSTM hidden dimension
    num_lstm_layers: int = 1  # Number of LSTM layers
    action_hidden_dim: int = 128  # Hidden dimension for action MLP
    fusion_hidden_dim: int = 512  # Hidden dimension for fusion MLP
    freeze_encoder: bool = True  # Whether to freeze encoder weights
    freeze_decoder: bool = False  # Whether to freeze decoder weights
    input_channels: int = 2  # Number of input channels (2 for complex image data)
    latent_dim: int = 256  # Latent representation dimension (from autoencoder)
    input_crop_size: int = None  # Center crop size (None for no cropping)
    
    # Sequence settings for BPTT
    sequence_length: int = 10  # Length of sequences for BPTT
    
    # Episode-based training (NEW)
    use_episodes: bool = False  # Use episode-based training instead of sequence-based
    min_episode_length: int = 20  # Minimum episode length for episode-based training
    max_episode_length: int = None  # Maximum episode length (truncate if longer, None for no limit)
    
    # Dataset keys
    obs_key: str = "observations"  # HDF5/NPZ key for observations
    action_key: str = "sa_incremental_actions"  # HDF5/NPZ key for actions
    
    # Training settings
    loss_function: str = "mse"  # mse, mae, smooth_l1, huber
    huber_delta: float = 0.1  # Delta for Huber loss
    optimizer: str = "adam"  # adam, sgd, adamw
    weight_decay: float = 1e-5  # L2 regularization
    scheduler: str = "cosine"  # cosine, step, none
    
    # Learning rate scheduler settings (ReduceLROnPlateau)
    use_scheduler: bool = True  # Use ReduceLROnPlateau scheduler
    scheduler_patience: int = 50  # Epochs to wait before reducing LR
    scheduler_factor: float = 0.5  # Factor to reduce LR by
    scheduler_min_lr: float = 1e-7  # Minimum learning rate
    
    # Data settings
    max_examples: int = None  # Limit dataset size for debugging
    num_workers: int = 4  # DataLoader workers
    prefetch_factor: int = 2  # Number of batches to prefetch per worker
    persistent_workers: bool = True  # Keep workers alive between epochs
    pin_memory: bool = True  # Pin memory for GPU training
    log_scale: bool = False  # Apply log-scaling to observations
    load_in_memory: bool = False  # Load entire dataset into memory for faster training
    use_data_parallel: bool = False  # Use DataParallel for multi-GPU training (uses all available GPUs)
    reconstruction_interval: int = 1  # Save reconstruction samples every N epochs
    checkpoint_interval: int = 10  # Save model checkpoints every N epochs
    use_multiprocessing: bool = False  # Use parallel loading for in-memory dataset (faster but may have issues)
    
    # Training optimizations
    use_amp: bool = False  # Use automatic mixed precision training


# Keep the old config class for backwards compatibility
@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder training"""
    dataset_path: str = "datasets/sml_100k_dataset"
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05
    device: str = "auto"  # auto, cuda, mps, cpu
    save_model: bool = True
    model_save_path: str = "saved_models/autoencoder.pth"
    plot_losses: bool = True
    seed: int = 42
    resume_from: str = None  # Path to checkpoint to resume from
    start_epoch: int = 0  # Starting epoch (for resumed training)
    
    # Model architecture settings
    arch: str = "autoencoder_cnn"  # autoencoder_cnn, autoencoder_resnet
    input_channels: int = 2  # Number of input channels (2 for complex image data)
    latent_dim: int = 256  # Latent representation dimension
    input_crop_size: int = None  # Center crop size (None for no cropping)
    
    # Training settings
    loss_function: str = "mse"  # mse, mae, smooth_l1, huber
    huber_delta: float = 0.1  # Delta for Huber loss
    optimizer: str = "adam"  # adam, sgd, adamw
    weight_decay: float = 1e-5  # L2 regularization
    scheduler: str = "cosine"  # cosine, step, none
    
    # Data settings
    max_examples: int = None  # Limit dataset size for debugging
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for GPU training
    log_scale: bool = False  # Apply log-scaling to observations
    load_in_memory: bool = False  # Load entire dataset into memory for faster training
    use_data_parallel: bool = False  # Use DataParallel for multi-GPU training (uses all available GPUs)
    reconstruction_interval: int = 1  # Save reconstruction samples every N epochs
    checkpoint_interval: int = 10  # Save model checkpoints every N epochs


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device for training"""
    # Clean up device string (remove trailing commas, whitespace)
    device_str = device_str.strip().rstrip(',').strip()
    
    print(f"🔧 Device string (cleaned): '{device_str}'")
    
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # Validate that it's a proper device string
        if device_str.startswith('cuda:') and device_str[5:].isdigit():
            device = torch.device(device_str)
        elif device_str in ['cuda', 'mps', 'cpu']:
            device = torch.device(device_str)
        else:
            print(f"⚠️  Invalid device string '{device_str}', falling back to auto-detection")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔧 Using device: {device}")
    return device


def normalize_for_display(tensor):
    """
    Normalize a tensor to [0, 1] range for visualization.
    Handles both single images and batches.
    
    Args:
        tensor: Input tensor of shape [C, H, W] or [N, C, H, W]
        
    Returns:
        Normalized tensor in range [0, 1]
    """
    # Normalize each image in the batch independently
    if len(tensor.shape) == 4:
        # Batch: [N, C, H, W]
        normalized = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            img = tensor[i]
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-7:  # Avoid division by zero
                normalized[i] = (img - img_min) / (img_max - img_min)
            else:
                normalized[i] = img
        return normalized
    else:
        # Single image: [C, H, W]
        img_min = tensor.min()
        img_max = tensor.max()
        if img_max - img_min > 1e-7:
            return (tensor - img_min) / (img_max - img_min)
        else:
            return tensor


class MaskedLoss(nn.Module):
    """
    Wrapper for loss functions that handles padding masks.
    Only computes loss on non-padded timesteps in batched episodes.
    
    This is essential for TBPTT (Truncated Backpropagation Through Time) with
    variable-length episodes that are padded to enable batching.
    """
    
    def __init__(self, base_criterion: nn.Module):
        """
        Args:
            base_criterion: Base loss function (e.g., nn.MSELoss(), nn.L1Loss())
        """
        super().__init__()
        # Store the original criterion type and create a version with reduction='none'
        self.criterion_type = type(base_criterion)
        
        # Create element-wise version
        if isinstance(base_criterion, nn.MSELoss):
            self.base_criterion = nn.MSELoss(reduction='none')
        elif isinstance(base_criterion, nn.L1Loss):
            self.base_criterion = nn.L1Loss(reduction='none')
        elif isinstance(base_criterion, nn.SmoothL1Loss):
            self.base_criterion = nn.SmoothL1Loss(reduction='none')
        elif isinstance(base_criterion, nn.HuberLoss):
            delta = base_criterion.delta if hasattr(base_criterion, 'delta') else 1.0
            self.base_criterion = nn.HuberLoss(delta=delta, reduction='none')
        else:
            self.base_criterion = base_criterion
            if hasattr(self.base_criterion, 'reduction'):
                self.base_criterion.reduction = 'none'
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked loss.
        
        Args:
            pred: Predictions [batch, seq_len, ...] 
            target: Targets [batch, seq_len, ...]
            mask: Binary mask [batch, seq_len] where 1=valid, 0=padding
                  If None, compute loss on all timesteps (no masking)
        
        Returns:
            Scalar loss value (mean over valid timesteps only)
        """
        # Compute element-wise loss
        loss = self.base_criterion(pred, target)
        
        # If no mask provided, return mean loss over all elements
        if mask is None:
            return loss.mean()
        
        # Reshape mask to match loss dimensions
        # loss shape: [batch, seq_len, C, H, W]
        # mask shape: [batch, seq_len]
        # Expand mask to: [batch, seq_len, 1, 1, 1]
        mask_expanded = mask
        while mask_expanded.dim() < loss.dim():
            mask_expanded = mask_expanded.unsqueeze(-1)
        
        # Apply mask: zero out loss for padded timesteps
        masked_loss = loss * mask_expanded.float()
        
        # Compute mean over valid (non-padded) elements only
        # We need to count the total number of valid ELEMENTS (pixels), not just timesteps
        # Each valid timestep contributes (C * H * W) elements
        num_valid_timesteps = mask.float().sum()
        if num_valid_timesteps > 0:
            # Get number of elements per timestep
            elements_per_timestep = loss[0, 0].numel()  # C * H * W
            num_valid_elements = num_valid_timesteps * elements_per_timestep
            return masked_loss.sum() / num_valid_elements
        else:
            # All timesteps are padded (shouldn't happen in practice)
            return loss.mean()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    print("🚂 Starting training epoch...")
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    print(f"📊 Training on {total_batches} batches")
    
    batch_times = []
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_batch_time = np.mean(batch_times[-100:])
            batches_remaining = total_batches - batch_idx
            eta_seconds = avg_batch_time * batches_remaining
            eta_minutes = eta_seconds / 60
            print(f"  📦 Batch {batch_idx}/{total_batches} | "
                  f"Avg Loss: {running_loss / batch_idx:.6f} | "
                  f"Batch Time: {avg_batch_time:.3f}s | "
                  f"ETA: {eta_minutes:.1f}m")
        
        if batch_idx == 0:  # Detailed debug for first batch only
            print(f"🔄 Batch {batch_idx}: Moving data to device...")
            data_transfer_start = time.time()
        
        data, target = data.to(device), target.to(device)
        
        if batch_idx == 0:
            data_transfer_time = time.time() - data_transfer_start
            print(f"✅ Batch {batch_idx}: Data moved to {device} in {data_transfer_time:.3f}s")
            print(f"   Data shape: {data.shape}")
            print(f"   Target shape: {target.shape}")
            print(f"   Data dtype: {data.dtype}")
            print(f"   Data device: {data.device}")
        
        if batch_idx == 0:
            print(f"🧹 Batch {batch_idx}: Zeroing gradients...")
        optimizer.zero_grad()
        
        # Forward pass
        if batch_idx == 0:
            print(f"➡️  Batch {batch_idx}: Forward pass...")
            forward_start = time.time()
        
        if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
            # Model returns both reconstruction and latent (like our autoencoders)
            reconstruction, latent = model(data)
        else:
            # Model returns only reconstruction
            reconstruction = model(data)
        
        if batch_idx == 0:
            forward_time = time.time() - forward_start
            print(f"✅ Batch {batch_idx}: Forward pass complete in {forward_time:.3f}s")
            print(f"   Reconstruction shape: {reconstruction.shape}")
        
        # Compute reconstruction loss
        if batch_idx == 0:
            print(f"📊 Batch {batch_idx}: Computing loss...")
            loss_start = time.time()
        
        loss = criterion(reconstruction, target)
        
        if batch_idx == 0:
            loss_time = time.time() - loss_start
            print(f"✅ Batch {batch_idx}: Loss computed in {loss_time:.3f}s")
            print(f"   Loss value: {loss.item():.6f}")
        
        # Backward pass
        if batch_idx == 0:
            print(f"⬅️  Batch {batch_idx}: Backward pass...")
            backward_start = time.time()
        
        loss.backward()
        
        if batch_idx == 0:
            backward_time = time.time() - backward_start
            print(f"✅ Batch {batch_idx}: Backward pass complete in {backward_time:.3f}s")
        
        if batch_idx == 0:
            print(f"⚡ Batch {batch_idx}: Optimizer step...")
            opt_start = time.time()
        
        optimizer.step()
        
        if batch_idx == 0:
            opt_time = time.time() - opt_start
            print(f"✅ Batch {batch_idx}: Optimizer step complete in {opt_time:.3f}s")
        
        running_loss += loss.item()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if batch_idx == 0:
            print(f"🎯 First batch completed successfully!")
            print(f"   Total batch time: {batch_time:.3f}s")
            print(f"   - Data transfer: {data_transfer_time:.3f}s")
            print(f"   - Forward pass: {forward_time:.3f}s")
            print(f"   - Loss computation: {loss_time:.3f}s")
            print(f"   - Backward pass: {backward_time:.3f}s")
            print(f"   - Optimizer step: {opt_time:.3f}s")
    
    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_loader)
    print(f"🏁 Training epoch complete in {epoch_time/60:.2f} minutes")
    print(f"   Average loss: {avg_loss:.6f}")
    print(f"   Average batch time: {np.mean(batch_times):.3f}s")
    return avg_loss


def train_world_model_epoch_episodes_batched(
    model, 
    episode_loader, 
    criterion, 
    optimizer, 
    device, 
    sequence_length: int = 10,
    scaler=None
):
    """
    Train world model for one epoch using episode-based TBPTT with proper batching.
    
    Implements Truncated Backpropagation Through Time (TBPTT) where:
    - Multiple variable-length episodes are batched together with padding
    - Hidden states are maintained within episodes
    - Hidden states are detached every sequence_length steps (TBPTT truncation)
    - Masked loss ignores padded timesteps
    
    This enables full GPU utilization by processing multiple episodes in parallel.
    
    Args:
        model: World model
        episode_loader: DataLoader yielding batched, padded episodes
        criterion: Masked loss function (MaskedLoss instance)
        optimizer: Optimizer
        device: Device to train on
        sequence_length: TBPTT truncation length (detach hidden every k steps)
        scaler: GradScaler for mixed precision training (None to disable AMP)
    
    Returns:
        Tuple of (average_loss, total_sequences_processed)
    """
    model.train()
    running_loss = 0.0
    total_sequences = 0
    total_valid_timesteps = 0
    use_amp = scaler is not None
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0
    
    batch_fetch_start = time.time()
    first_batch = True
    
    # Timing breakdown accumulators
    total_fetch_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_data_move_time = 0.0
    batch_count = 0
    
    for batch_idx, batch_data in enumerate(episode_loader):
        # Unpack batched, padded episodes
        obs_padded, actions_padded, next_obs_padded, lengths, mask = batch_data
        
        batch_fetch_time = time.time() - batch_fetch_start
        total_fetch_time += batch_fetch_time
        batch_count += 1
        
        if first_batch:
            batch_size, max_ep_len = obs_padded.shape[0], obs_padded.shape[1]
            print(f"📦 Batched episode processing with TBPTT:")
            print(f"   Batch size: {batch_size} episodes")
            print(f"   Max episode length in batch: {max_ep_len} timesteps")
            print(f"   Sequence length (TBPTT truncation): {sequence_length}")
            print(f"  ⏱️  First batch fetched in {batch_fetch_time:.2f}s")
            first_batch = False
        
        # Move batch to device
        data_move_start = time.time()
        obs_padded = obs_padded.to(device)
        actions_padded = actions_padded.to(device)
        next_obs_padded = next_obs_padded.to(device)
        mask = mask.to(device)
        total_data_move_time += time.time() - data_move_start
        
        batch_size, max_ep_len = obs_padded.shape[0], obs_padded.shape[1]
        
        # Initialize hidden states for all episodes in batch
        hidden = model.get_zero_hidden(batch_size, device)
        
        # Process episodes in sequence_length chunks (TBPTT)
        # This implements "k1 = k2 = sequence_length" style TBPTT
        for start_idx in range(0, max_ep_len, sequence_length):
            end_idx = min(start_idx + sequence_length, max_ep_len)
            
            # Extract sequence chunk for this TBPTT step
            obs_chunk = obs_padded[:, start_idx:end_idx]  # [batch, seq_len, C, H, W]
            actions_chunk = actions_padded[:, start_idx:end_idx]  # [batch, seq_len, action_dim]
            next_obs_chunk = next_obs_padded[:, start_idx:end_idx]  # [batch, seq_len, C, H, W]
            mask_chunk = mask[:, start_idx:end_idx]  # [batch, seq_len]
            
            # Skip if this chunk has no valid timesteps
            if not mask_chunk.any():
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            forward_start = time.time()
            if use_amp:
                with torch.cuda.amp.autocast():
                    next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
            else:
                next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
            
            # Match target size to prediction size (in case encoder crops)
            if next_obs_pred.shape != next_obs_chunk.shape:
                from models.models import center_crop_transform
                batch_size_chunk, seq_len_chunk = next_obs_chunk.shape[0], next_obs_chunk.shape[1]
                next_obs_flat = next_obs_chunk.reshape(batch_size_chunk * seq_len_chunk, *next_obs_chunk.shape[2:])
                target_size = next_obs_pred.shape[-1]
                next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
                next_obs_chunk = next_obs_cropped.reshape(batch_size_chunk, seq_len_chunk, *next_obs_cropped.shape[1:])
            
            # Compute masked loss (only on valid timesteps)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(next_obs_pred, next_obs_chunk, mask_chunk)
            else:
                loss = criterion(next_obs_pred, next_obs_chunk, mask_chunk)
            total_forward_time += time.time() - forward_start
            
            # Backward pass
            backward_start = time.time()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_backward_time += time.time() - backward_start
            
            # TBPTT: Detach hidden state to truncate gradient flow
            # This is the key to TBPTT - we carry forward the hidden state values
            # but prevent gradients from flowing back beyond sequence_length steps
            hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            # Accumulate loss weighted by number of valid timesteps
            num_valid = mask_chunk.float().sum().item()
            running_loss += loss.item() * num_valid
            total_valid_timesteps += num_valid
            total_sequences += 1
        
        # Periodic cache clearing (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # Prepare for next batch fetch timing
        batch_fetch_start = time.time()
        
        # Progress reporting
        current_time = time.time()
        if (current_time - last_print_time) >= print_interval or batch_idx == 0:
            avg_loss = running_loss / max(1, total_valid_timesteps)
            print(f'\r🚂 Train Batches: {batch_idx+1} | '
                  f'Sequences: {total_sequences} | '
                  f'Avg Loss: {avg_loss:.6f}', end='', flush=True)
            last_print_time = current_time
    
    print()  # New line
    avg_loss = running_loss / max(1, total_valid_timesteps)
    print(f"📊 Epoch complete: {total_sequences} TBPTT sequences processed")
    
    # Print timing breakdown
    if batch_count > 0:
        print(f"⏱️  Timing breakdown:")
        print(f"   Batch fetch:  {total_fetch_time:.2f}s ({total_fetch_time/batch_count:.3f}s/batch)")
        print(f"   Data->GPU:    {total_data_move_time:.2f}s")
        print(f"   Forward pass: {total_forward_time:.2f}s")
        print(f"   Backward pass: {total_backward_time:.2f}s")
    
    return avg_loss, total_sequences


def train_world_model_epoch_episodes_old(
    model, 
    episode_loader, 
    criterion, 
    optimizer, 
    device, 
    sequence_length: int = 10,
    scaler=None
):
    """
    Train world model for one epoch using episode-based BPTT.
    
    Processes full episodes while maintaining hidden state continuity.
    Episodes are chunked into sequences of sequence_length for memory efficiency,
    but hidden state is carried across chunks within the same episode.
    
    Args:
        model: World model
        episode_loader: DataLoader yielding full episodes
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        sequence_length: Length of sequences for chunked processing
        scaler: GradScaler for mixed precision training (None to disable AMP)
    """
    model.train()
    running_loss = 0.0
    total_episodes = 0
    total_sequences = 0
    use_amp = scaler is not None
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0
    
    print(f"⚠️  Note: Episode-based training processes episodes sequentially (batch_size=1 per episode)")
    print(f"   For maximum GPU utilization, consider:")
    print(f"   1. Increasing sequence_length (currently {sequence_length})")
    print(f"   2. Using sequence-based training with larger batch_size")
    
    batch_fetch_start = time.time()
    first_batch = True
    
    # Timing breakdown accumulators
    total_fetch_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_data_move_time = 0.0
    batch_count = 0
    
    for batch_idx, episode_batch in enumerate(episode_loader):
        # episode_batch is a list of episodes: [(obs, actions, next_obs, length), ...]
        
        batch_fetch_time = time.time() - batch_fetch_start
        total_fetch_time += batch_fetch_time
        batch_count += 1
        
        if first_batch:
            estimated_total = batch_fetch_time * len(episode_loader)
            print(f"  ⏱️  First batch fetched in {batch_fetch_time:.2f}s (estimated total: {estimated_total:.1f}s)")
            first_batch = False
        
        for episode_idx, (obs, actions, next_obs, episode_length) in enumerate(episode_batch):
            if episode_length == 0:
                continue
                
            # Move episode data to device
            data_move_start = time.time()
            obs = obs.to(device)
            actions = actions.to(device)
            next_obs = next_obs.to(device)
            total_data_move_time += time.time() - data_move_start
            
            # Initialize hidden state for this episode
            hidden = model.get_zero_hidden(1, device)  # batch_size=1 for single episode
            
            # Process episode in chunks to maintain memory efficiency
            episode_loss = 0.0
            num_sequences_in_episode = 0
            
            for start_idx in range(0, episode_length, sequence_length):
                end_idx = min(start_idx + sequence_length, episode_length)
                
                # Extract sequence chunk
                obs_chunk = obs[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, C, H, W]
                actions_chunk = actions[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, action_dim]
                next_obs_chunk = next_obs[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, C, H, W]
                
                # Zero gradients for this sequence
                optimizer.zero_grad()
                
                # Forward pass with carried hidden state
                forward_start = time.time()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
                else:
                    next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
                
                # Match target size to prediction size (in case encoder crops)
                if next_obs_pred.shape != next_obs_chunk.shape:
                    from models.models import center_crop_transform
                    batch_size, seq_len = next_obs_chunk.shape[0], next_obs_chunk.shape[1]
                    next_obs_flat = next_obs_chunk.reshape(batch_size * seq_len, *next_obs_chunk.shape[2:])
                    target_size = next_obs_pred.shape[-1]
                    next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
                    next_obs_chunk = next_obs_cropped.reshape(batch_size, seq_len, *next_obs_cropped.shape[1:])
                
                # Compute loss
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = criterion(next_obs_pred, next_obs_chunk)
                else:
                    loss = criterion(next_obs_pred, next_obs_chunk)
                total_forward_time += time.time() - forward_start
                
                # Backward pass
                backward_start = time.time()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_backward_time += time.time() - backward_start
                
                # Update hidden state for next sequence (detach to prevent gradient explosion)
                hidden = (new_hidden[0].detach(), new_hidden[1].detach())
                
                episode_loss += loss.item()
                num_sequences_in_episode += 1
                total_sequences += 1
                
                # Clean up tensors (but don't clear cache every iteration - expensive)
                del obs_chunk, actions_chunk, next_obs_chunk, next_obs_pred, latent, new_hidden, loss
            
            # Average loss for this episode
            if num_sequences_in_episode > 0:
                episode_avg_loss = episode_loss / num_sequences_in_episode
                running_loss += episode_avg_loss
            
            total_episodes += 1
            
            # Clean up episode tensors
            del obs, actions, next_obs, hidden
        
        # Clear cache only once per batch (not per sequence/episode)
        if device.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Prepare for next batch fetch timing
        batch_fetch_start = time.time()
        
        # Progress reporting
        current_time = time.time()
        if (current_time - last_print_time) >= print_interval or batch_idx == 0:
            avg_loss = running_loss / max(1, total_episodes)
            print(f'\r🚂 Train Episodes: Batch {batch_idx+1} | '
                  f'Episodes: {total_episodes} | Sequences: {total_sequences} | '
                  f'Avg Loss: {avg_loss:.6f}', end='', flush=True)
            last_print_time = current_time
    
    print()  # New line
    avg_loss = running_loss / max(1, total_episodes)
    print(f"📊 Epoch complete: {total_episodes} episodes, {total_sequences} sequences processed")
    
    # Print timing breakdown
    if batch_count > 0:
        print(f"⏱️  Timing breakdown:")
        print(f"   Batch fetch:  {total_fetch_time:.2f}s ({total_fetch_time/batch_count:.3f}s/batch)")
        print(f"   Data->GPU:    {total_data_move_time:.2f}s")
        print(f"   Forward pass: {total_forward_time:.2f}s")
        print(f"   Backward pass: {total_backward_time:.2f}s")
    
    return avg_loss, total_sequences


def train_world_model_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train world model for one epoch with BPTT
    
    Args:
        model: World model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training (None to disable AMP)
    
    NOTE: Since dataloader shuffles sequences from different episodes, we cannot
    carry hidden state between batches (they come from different episodes).
    Each batch is treated as an independent sequence with zero initial hidden state.
    This is correct behavior given the shuffled data.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    use_amp = scaler is not None
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0  # Print progress every second
    
    for batch_idx, (obs, actions, next_obs) in enumerate(train_loader):
        # Zero gradients before forward pass
        optimizer.zero_grad()
        
        # Move data to device
        obs = obs.to(device)
        actions = actions.to(device)
        next_obs = next_obs.to(device)
        
        # Forward pass with BPTT (with optional AMP)
        # Note: hidden state starts at zero for each batch since sequences are shuffled
        # and come from different episodes. This is correct behavior.
        if use_amp:
            with torch.cuda.amp.autocast():
                next_obs_pred, latent, hidden = model(obs, actions)
        else:
            next_obs_pred, latent, hidden = model(obs, actions)
        
        # Match target size to prediction size (in case encoder crops the input)
        if next_obs_pred.shape != next_obs.shape:
            from models.models import center_crop_transform
            batch_size, seq_len = next_obs.shape[0], next_obs.shape[1]
            next_obs_flat = next_obs.reshape(batch_size * seq_len, *next_obs.shape[2:])
            target_size = next_obs_pred.shape[-1]
            next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
            next_obs = next_obs_cropped.reshape(batch_size, seq_len, *next_obs_cropped.shape[1:])
        
        # Compute prediction loss (with optional AMP)
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = criterion(next_obs_pred, next_obs)
        else:
            loss = criterion(next_obs_pred, next_obs)
        
        # Backward pass (with optional AMP)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with optional AMP)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        running_loss += loss.item()
        
        # Progress bar style printing (update every second or at milestones)
        current_time = time.time()
        should_print = (
            batch_idx == 0 or  # First batch
            batch_idx == total_batches - 1 or  # Last batch
            (current_time - last_print_time) >= print_interval or  # Time interval
            (batch_idx + 1) % max(1, total_batches // 10) == 0  # Every 10%
        )
        
        if should_print:
            progress = (batch_idx + 1) / total_batches
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            avg_loss = running_loss / (batch_idx + 1)
            
            print(f'\r🚂 Train: [{bar}] {progress*100:5.1f}% | '
                  f'Batch {batch_idx+1}/{total_batches} | '
                  f'Loss: {avg_loss:.6f}', end='', flush=True)
            last_print_time = current_time
        
        # Explicitly delete tensors and clear cache to prevent memory buildup
        del obs, actions, next_obs, next_obs_pred, latent, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print()  # New line after progress bar
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    print("🔍 Starting validation epoch...")
    val_start = time.time()
    model.eval()
    running_loss = 0.0
    total_batches = len(val_loader)
    
    print(f"📊 Validating on {total_batches} batches")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # Print progress every 100 batches for validation
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  🔍 Validating batch {batch_idx}/{total_batches} | "
                      f"Avg Loss: {running_loss / batch_idx:.6f}")
            
            if batch_idx == 0:  # Detailed debug for first batch only
                print(f"🔄 Val Batch {batch_idx}: Moving data to device...")
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if batch_idx == 0:
                print(f"➡️  Val Batch {batch_idx}: Forward pass...")
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                reconstruction, latent = model(data)
            else:
                reconstruction = model(data)
            if batch_idx == 0:
                print(f"✅ Val Batch {batch_idx}: Forward pass complete")
            
            if batch_idx == 0:
                print(f"📊 Val Batch {batch_idx}: Computing loss...")
            loss = criterion(reconstruction, target)
            
            if batch_idx == 0:
                print(f"✅ Val Batch {batch_idx}: Loss computed: {loss.item():.6f}")
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(val_loader)
    val_time = time.time() - val_start
    print(f"🏁 Validation complete in {val_time/60:.2f} minutes")
    print(f"   Average loss: {avg_loss:.6f}")
    return avg_loss


def validate_world_model_epoch_episodes_batched(
    model, 
    episode_loader, 
    criterion, 
    device,
    sequence_length: int = 10
):
    """
    Validate world model for one epoch using batched episode-based processing with TBPTT.
    """
    model.eval()
    running_loss = 0.0
    total_sequences = 0
    total_valid_timesteps = 0
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0
    
    batch_fetch_start = time.time()
    first_batch = True
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(episode_loader):
            # Unpack batched, padded episodes
            obs_padded, actions_padded, next_obs_padded, lengths, mask = batch_data
            
            batch_fetch_time = time.time() - batch_fetch_start
            batch_count += 1
            
            if first_batch:
                batch_size, max_ep_len = obs_padded.shape[0], obs_padded.shape[1]
                print(f"📦 Batched validation:")
                print(f"   Batch size: {batch_size} episodes")
                print(f"   Max episode length: {max_ep_len} timesteps")
                print(f"  ⏱️  First batch fetched in {batch_fetch_time:.2f}s")
                first_batch = False
            
            # Move batch to device
            obs_padded = obs_padded.to(device)
            actions_padded = actions_padded.to(device)
            next_obs_padded = next_obs_padded.to(device)
            mask = mask.to(device)
            
            batch_size, max_ep_len = obs_padded.shape[0], obs_padded.shape[1]
            
            # Initialize hidden states for all episodes in batch
            hidden = model.get_zero_hidden(batch_size, device)
            
            # Process episodes in sequence_length chunks
            for start_idx in range(0, max_ep_len, sequence_length):
                end_idx = min(start_idx + sequence_length, max_ep_len)
                
                # Extract sequence chunk
                obs_chunk = obs_padded[:, start_idx:end_idx]
                actions_chunk = actions_padded[:, start_idx:end_idx]
                next_obs_chunk = next_obs_padded[:, start_idx:end_idx]
                mask_chunk = mask[:, start_idx:end_idx]
                
                # Skip if this chunk has no valid timesteps
                if not mask_chunk.any():
                    continue
                
                # Forward pass
                next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
                
                # Match target size to prediction size (in case encoder crops)
                if next_obs_pred.shape != next_obs_chunk.shape:
                    from models.models import center_crop_transform
                    batch_size_chunk, seq_len_chunk = next_obs_chunk.shape[0], next_obs_chunk.shape[1]
                    next_obs_flat = next_obs_chunk.reshape(batch_size_chunk * seq_len_chunk, *next_obs_chunk.shape[2:])
                    target_size = next_obs_pred.shape[-1]
                    next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
                    next_obs_chunk = next_obs_cropped.reshape(batch_size_chunk, seq_len_chunk, *next_obs_cropped.shape[1:])
                
                # Compute masked loss
                loss = criterion(next_obs_pred, next_obs_chunk, mask_chunk)
                
                # Update hidden state for next sequence
                hidden = (new_hidden[0].detach(), new_hidden[1].detach())
                
                # Accumulate loss weighted by number of valid timesteps
                num_valid = mask_chunk.float().sum().item()
                running_loss += loss.item() * num_valid
                total_valid_timesteps += num_valid
                total_sequences += 1
            
            # Clear cache periodically
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Prepare for next batch fetch timing
            batch_fetch_start = time.time()
            
            # Progress reporting
            current_time = time.time()
            if (current_time - last_print_time) >= print_interval or batch_idx == 0:
                avg_loss = running_loss / max(1, total_valid_timesteps)
                print(f'\r🔍 Valid Batches: {batch_idx+1} | '
                      f'Sequences: {total_sequences} | '
                      f'Avg Loss: {avg_loss:.6f}', end='', flush=True)
                last_print_time = current_time
    
    print()  # New line
    avg_loss = running_loss / max(1, total_valid_timesteps)
    print(f"📊 Validation complete: {total_sequences} sequences processed")
    
    return avg_loss, total_sequences


def validate_world_model_epoch_episodes_old(
    model, 
    episode_loader, 
    criterion, 
    device,
    sequence_length: int = 10
):
    """
    Validate world model for one epoch using episode-based processing.
    """
    model.eval()
    running_loss = 0.0
    total_episodes = 0
    total_sequences = 0
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0
    
    batch_fetch_start = time.time()
    first_batch = True
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, episode_batch in enumerate(episode_loader):
            # episode_batch is a list of episodes: [(obs, actions, next_obs, length), ...]
            
            batch_fetch_time = time.time() - batch_fetch_start
            batch_count += 1
            
            if first_batch:
                estimated_total = batch_fetch_time * len(episode_loader)
                print(f"  ⏱️  First batch fetched in {batch_fetch_time:.2f}s (estimated total: {estimated_total:.1f}s)")
                first_batch = False
            
            for episode_idx, (obs, actions, next_obs, episode_length) in enumerate(episode_batch):
                if episode_length == 0:
                    continue
                    
                # Move episode data to device
                obs = obs.to(device)
                actions = actions.to(device)
                next_obs = next_obs.to(device)
                
                # Initialize hidden state for this episode
                hidden = model.get_zero_hidden(1, device)  # batch_size=1 for single episode
                
                # Process episode in chunks
                episode_loss = 0.0
                num_sequences_in_episode = 0
                
                for start_idx in range(0, episode_length, sequence_length):
                    end_idx = min(start_idx + sequence_length, episode_length)
                    
                    # Extract sequence chunk
                    obs_chunk = obs[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, C, H, W]
                    actions_chunk = actions[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, action_dim]
                    next_obs_chunk = next_obs[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, C, H, W]
                    
                    # Forward pass with carried hidden state
                    next_obs_pred, latent, new_hidden = model(obs_chunk, actions_chunk, hidden)
                    
                    # Match target size to prediction size (in case encoder crops)
                    if next_obs_pred.shape != next_obs_chunk.shape:
                        from models.models import center_crop_transform
                        batch_size, seq_len = next_obs_chunk.shape[0], next_obs_chunk.shape[1]
                        next_obs_flat = next_obs_chunk.reshape(batch_size * seq_len, *next_obs_chunk.shape[2:])
                        target_size = next_obs_pred.shape[-1]
                        next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
                        next_obs_chunk = next_obs_cropped.reshape(batch_size, seq_len, *next_obs_cropped.shape[1:])
                    
                    # Compute loss
                    loss = criterion(next_obs_pred, next_obs_chunk)
                    
                    # Update hidden state for next sequence
                    hidden = (new_hidden[0].detach(), new_hidden[1].detach())
                    
                    episode_loss += loss.item()
                    num_sequences_in_episode += 1
                    total_sequences += 1
                    
                    # Clean up tensors (but don't clear cache every iteration)
                    del obs_chunk, actions_chunk, next_obs_chunk, next_obs_pred, latent, new_hidden
                
                # Average loss for this episode
                if num_sequences_in_episode > 0:
                    episode_avg_loss = episode_loss / num_sequences_in_episode
                    running_loss += episode_avg_loss
                
                total_episodes += 1
                
                # Clean up episode tensors
                del obs, actions, next_obs, hidden
            
            # Clear cache only once per batch (not per sequence/episode)
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Prepare for next batch fetch timing
            batch_fetch_start = time.time()
            
            # Progress reporting
            current_time = time.time()
            if (current_time - last_print_time) >= print_interval or batch_idx == 0:
                avg_loss = running_loss / max(1, total_episodes)
                print(f'\r🔍 Valid Episodes: Batch {batch_idx+1} | '
                      f'Episodes: {total_episodes} | Sequences: {total_sequences} | '
                      f'Avg Loss: {avg_loss:.6f}', end='', flush=True)
                last_print_time = current_time
    
    print()  # New line
    avg_loss = running_loss / max(1, total_episodes)
    print(f"📊 Validation complete: {total_episodes} episodes, {total_sequences} sequences processed")
    return avg_loss, total_sequences


def validate_world_model_epoch(model, val_loader, criterion, device):
    """Validate world model for one epoch"""
    model.eval()
    running_loss = 0.0
    total_batches = len(val_loader)
    
    # Progress tracking
    last_print_time = time.time()
    print_interval = 1.0  # Print progress every second
    
    with torch.no_grad():
        for batch_idx, (obs, actions, next_obs) in enumerate(val_loader):
            # Move data to device
            obs = obs.to(device)
            actions = actions.to(device)
            next_obs = next_obs.to(device)
            
            # Forward pass
            next_obs_pred, latent, hidden = model(obs, actions)
            
            # Match target size to prediction size (in case encoder crops the input)
            if next_obs_pred.shape != next_obs.shape:
                from models.models import center_crop_transform
                batch_size, seq_len = next_obs.shape[0], next_obs.shape[1]
                next_obs_flat = next_obs.reshape(batch_size * seq_len, *next_obs.shape[2:])
                target_size = next_obs_pred.shape[-1]
                next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
                next_obs = next_obs_cropped.reshape(batch_size, seq_len, *next_obs_cropped.shape[1:])
            
            # Compute loss
            loss = criterion(next_obs_pred, next_obs)
            running_loss += loss.item()
            
            # Progress bar style printing (update every second or at milestones)
            current_time = time.time()
            should_print = (
                batch_idx == 0 or  # First batch
                batch_idx == total_batches - 1 or  # Last batch
                (current_time - last_print_time) >= print_interval or  # Time interval
                (batch_idx + 1) % max(1, total_batches // 10) == 0  # Every 10%
            )
            
            if should_print:
                progress = (batch_idx + 1) / total_batches
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                avg_loss = running_loss / (batch_idx + 1)
                
                print(f'\r🔍 Valid: [{bar}] {progress*100:5.1f}% | '
                      f'Batch {batch_idx+1}/{total_batches} | '
                      f'Loss: {avg_loss:.6f}', end='', flush=True)
                last_print_time = current_time
    
    print()  # New line after progress bar
    avg_loss = running_loss / len(val_loader)
    return avg_loss


def visualize_world_model_predictions(model, val_loader, device, writer, epoch, num_timesteps=4, prefix='Validation'):
    """
    Visualize world model predictions for debugging.
    Handles both episode-based and sequence-based data loaders.
    
    Shows 4 timesteps from a sequence, each column containing:
    - Residual (error magnitude)
    - Input observation
    - Action vector (as heatmap)
    - Hidden state (as heatmap)
    - Target next observation
    - Model prediction
    
    Args:
        model: World model
        val_loader: Data loader (train or validation)
        device: Device to run on
        writer: TensorBoard writer
        epoch: Current epoch number
        num_timesteps: Number of timesteps to visualize (default: 4)
        prefix: Prefix for TensorBoard tag (default: 'Validation')
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from models.models import center_crop_transform
    
    model.eval()
    
    with torch.no_grad():
        try:
            # Get first batch from validation set - handle both episode and sequence formats
            first_batch = next(iter(val_loader))
            
            # Robustly determine batch format by inspecting structure
            if isinstance(first_batch, list) and len(first_batch) > 0:
                first_item = first_batch[0]
                
                # Handle nested list structure (DataLoader may wrap collate output in extra list)
                if isinstance(first_item, list) and len(first_item) > 0:
                    # Unwrap one level
                    first_batch = first_item
                    first_item = first_batch[0]
                
                if isinstance(first_item, tuple):
                    if len(first_item) == 4:
                        # Episode-based format: list of (obs, actions, next_obs, length) tuples
                        obs, actions, next_obs, episode_length = first_item
                        # Take first num_timesteps from the episode
                        max_timesteps = min(num_timesteps, episode_length)
                        obs = obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                        actions = actions[:max_timesteps].unsqueeze(0)  # [1, timesteps, action_dim]
                        next_obs = next_obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                    elif len(first_item) == 3:
                        # Sequence-based format as list: list of (obs, actions, next_obs) tuples
                        obs, actions, next_obs = first_item
                        # Take first num_timesteps from the sequence
                        max_timesteps = min(num_timesteps, obs.shape[0])
                        obs = obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                        actions = actions[:max_timesteps].unsqueeze(0)  # [1, timesteps, action_dim]
                        next_obs = next_obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                    else:
                        raise ValueError(f"Unexpected tuple length: {len(first_item)}")
                elif torch.is_tensor(first_item):
                    # List of tensors - likely [obs, actions, next_obs] or [obs, actions, next_obs, lengths] or pre-collated batch
                    if len(first_batch) == 5:
                        # Pre-collated batch format with mask: [obs_padded, actions_padded, next_obs_padded, lengths, mask]
                        obs, actions, next_obs, lengths, mask = first_batch
                        # Select first item in batch
                        if obs.ndim == 5:  # [batch, seq, C, H, W]
                            obs = obs[0]  # [seq, C, H, W]
                            actions = actions[0]  # [seq, action_dim]
                            next_obs = next_obs[0]  # [seq, C, H, W]
                            actual_length = lengths[0].item() if torch.is_tensor(lengths) else lengths[0]
                            # Trim to actual length (remove padding)
                            obs = obs[:actual_length]
                            actions = actions[:actual_length]
                            next_obs = next_obs[:actual_length]
                        max_timesteps = min(num_timesteps, obs.shape[0])
                        obs = obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                        actions = actions[:max_timesteps].unsqueeze(0)  # [1, timesteps, action_dim]
                        next_obs = next_obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                    elif len(first_batch) == 4:
                        # Episode format with lengths
                        obs, actions, next_obs, episode_lengths = first_batch
                        # Select first item in batch
                        if obs.ndim == 5:  # [batch, seq, C, H, W]
                            obs = obs[0]  # [seq, C, H, W]
                            actions = actions[0]  # [seq, action_dim]
                            next_obs = next_obs[0]  # [seq, C, H, W]
                        max_timesteps = min(num_timesteps, obs.shape[0])
                        obs = obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                        actions = actions[:max_timesteps].unsqueeze(0)  # [1, timesteps, action_dim]
                        next_obs = next_obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                    elif len(first_batch) == 3:
                        # Standard format [obs, actions, next_obs]
                        obs, actions, next_obs = first_batch
                        # Select first item in batch
                        if obs.ndim == 5:  # [batch, seq, C, H, W]
                            obs = obs[0]  # [seq, C, H, W]
                            actions = actions[0]  # [seq, action_dim]
                            next_obs = next_obs[0]  # [seq, C, H, W]
                        max_timesteps = min(num_timesteps, obs.shape[0])
                        obs = obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                        actions = actions[:max_timesteps].unsqueeze(0)  # [1, timesteps, action_dim]
                        next_obs = next_obs[:max_timesteps].unsqueeze(0)  # [1, timesteps, C, H, W]
                    else:
                        raise ValueError(f"Unexpected number of tensors in list: {len(first_batch)}")
                else:
                    raise ValueError(f"Expected tuple or tensor in list, got {type(first_item)}")
            elif isinstance(first_batch, tuple) and len(first_batch) == 3:
                # Direct tuple format (obs, actions, next_obs) with batch dimension
                obs, actions, next_obs = first_batch
                # Select first item only and take first num_timesteps
                max_timesteps = min(num_timesteps, obs.shape[1])
                obs = obs[0:1, :max_timesteps]  # [1, timesteps, C, H, W]
                actions = actions[0:1, :max_timesteps]  # [1, timesteps, action_dim]
                next_obs = next_obs[0:1, :max_timesteps]  # [1, timesteps, C, H, W]
            elif isinstance(first_batch, dict):
                # Dictionary format
                obs = first_batch['observations']
                actions = first_batch['actions']
                next_obs = first_batch['next_observations']
                # Select first item only and take first num_timesteps
                max_timesteps = min(num_timesteps, obs.shape[1])
                obs = obs[0:1, :max_timesteps]
                actions = actions[0:1, :max_timesteps]
                next_obs = next_obs[0:1, :max_timesteps]
            else:
                raise ValueError(f"Unexpected batch format: {type(first_batch)}")
            
            # Ensure we only have one batch item for visualization
            if obs.shape[0] > 1:
                obs = obs[0:1]  # Take only first item, keep batch dimension
                actions = actions[0:1]
                next_obs = next_obs[0:1]
            
            obs = obs.to(device)
            actions = actions.to(device)
            next_obs = next_obs.to(device)
            
        except Exception as e:
            print(f"⚠️  Failed to load visualization data: {e}")
            return
        
        # Get predictions - we need to manually call forward to get lstm_out
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        device = obs.device
        
        # Encode observations
        obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        latent_flat = model.encoder(obs_flat)
        latent = latent_flat.reshape(batch_size, seq_len, -1)
        
        # LSTM forward pass - get lstm_out for visualization
        hidden = model.get_zero_hidden(batch_size, device)
        lstm_out, hidden = model.lstm(latent, hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        
        # Rest of forward pass with new architecture
        action_encoded = model.action_encoder(actions)  # [batch, seq_len, latent_dim]
        
        # Concatenate [latent, lstm_out, action_encoded]
        concat_features = torch.cat([latent, lstm_out, action_encoded], dim=-1)
        concat_flat = concat_features.reshape(batch_size * seq_len, -1)
        
        # Fusion MLP
        fused_flat = model.fusion_mlp(concat_flat)
        
        # Decoder
        next_obs_pred_flat = model.decoder(fused_flat)
        decoder_output_shape = next_obs_pred_flat.shape[1:]
        next_obs_pred = next_obs_pred_flat.reshape(batch_size, seq_len, *decoder_output_shape)
        
        # Crop observations and targets to match predictions if needed
        if next_obs_pred.shape != next_obs.shape:
            batch_size, seq_len = next_obs.shape[0], next_obs.shape[1]
            target_size = next_obs_pred.shape[-1]
            
            # Crop input observations
            obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
            obs_cropped = center_crop_transform(obs_flat, target_size)
            obs = obs_cropped.reshape(batch_size, seq_len, *obs_cropped.shape[1:])
            
            # Crop target observations
            next_obs_flat = next_obs.reshape(batch_size * seq_len, *next_obs.shape[2:])
            next_obs_cropped = center_crop_transform(next_obs_flat, target_size)
            next_obs = next_obs_cropped.reshape(batch_size, seq_len, *next_obs_cropped.shape[1:])
        
        # Move to CPU immediately and delete GPU tensors
        obs_seq = obs[0].cpu()  # [seq_len, C, H, W]
        actions_seq = actions[0].cpu()  # [seq_len, action_dim]
        next_obs_seq = next_obs[0].cpu()  # [seq_len, C, H, W]
        pred_seq = next_obs_pred[0].cpu()  # [seq_len, C, H, W]
        lstm_out_seq = lstm_out[0].cpu()  # [seq_len, hidden_dim] - use lstm output instead of hidden state
        
        # Delete large GPU tensors immediately
        del obs, actions, next_obs, obs_flat, latent_flat, latent, hidden, lstm_out
        del action_encoded, concat_features, concat_flat, fused_flat, next_obs_pred_flat, next_obs_pred
        if 'obs_cropped' in locals():
            del obs_cropped
        if 'next_obs_cropped' in locals():
            del next_obs_cropped
        torch.cuda.empty_cache()
        
        # Compute residuals
        residuals = torch.abs(next_obs_seq - pred_seq)
        
        # Compute global min/max for consistent scaling across all observation images
        all_obs_data = torch.cat([obs_seq, next_obs_seq, pred_seq], dim=0)
        global_vmin = all_obs_data.min().item()
        global_vmax = all_obs_data.max().item()
        
        # Create figure: 6 rows (residual, obs, action, hidden, target, pred) × num_timesteps columns
        fig, axes = plt.subplots(6, num_timesteps, figsize=(4*num_timesteps, 24))
        fig.suptitle(f'{prefix} World Model Predictions - Epoch {epoch+1}', fontsize=16, fontweight='bold')
        
        for t in range(num_timesteps):
            col = t
            
            # Row 0: Residual (FIRST for quick assessment)
            residual_img = residuals[t, 0].numpy()
            im0 = axes[0, col].imshow(residual_img, cmap='hot', vmin=0, vmax=residual_img.max())
            axes[0, col].set_title(f'Step {t+1}\nResidual (MAE={residual_img.mean():.6f})', fontsize=10)
            axes[0, col].axis('off')
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
            
            # Row 1: Input observation
            obs_img = obs_seq[t, 0].numpy()  # [H, W]
            im1 = axes[1, col].imshow(obs_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[1, col].set_title('Input Obs', fontsize=10)
            axes[1, col].axis('off')
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
            
            # Row 2: Action vector (as heatmap)
            action_vec = actions_seq[t].numpy()
            # Reshape to match hidden state width for consistent visualization
            action_size = len(action_vec)
            # Create a 1D horizontal bar visualization with same width as hidden state
            action_2d = action_vec.reshape(1, -1)  # [1, action_dim]
            
            im2 = axes[2, col].imshow(action_2d, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[2, col].set_title(f'Action ({action_size} dims)', fontsize=10)
            axes[2, col].axis('off')
            plt.colorbar(im2, ax=axes[2, col], fraction=0.046)
            
            # Row 3: Hidden state (as heatmap)
            if lstm_out_seq is not None:
                hidden_vec = lstm_out_seq[t].numpy()
                # Flatten if multi-dimensional (e.g., from multi-layer or bidirectional LSTM)
                if hidden_vec.ndim > 1:
                    hidden_vec = hidden_vec.flatten()
                # Reshape hidden state into 2D for visualization
                hidden_size = len(hidden_vec)
                # Find factors close to square root for better visualization
                h_rows = int(np.sqrt(hidden_size))
                while hidden_size % h_rows != 0 and h_rows > 1:
                    h_rows -= 1
                h_cols = hidden_size // h_rows
                # Truncate to exact size needed for reshape
                hidden_2d = hidden_vec[:h_rows*h_cols].reshape(h_rows, h_cols)
                
                im3 = axes[3, col].imshow(hidden_2d, cmap='coolwarm', aspect='auto')
                axes[3, col].set_title(f'Hidden State ({h_rows}x{h_cols})', fontsize=10)
                axes[3, col].axis('off')
                plt.colorbar(im3, ax=axes[3, col], fraction=0.046)
            else:
                axes[3, col].text(0.5, 0.5, 'No hidden\nstate', ha='center', va='center')
                axes[3, col].axis('off')
            
            # Row 4: Target next observation
            target_img = next_obs_seq[t, 0].numpy()
            im4 = axes[4, col].imshow(target_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[4, col].set_title('Target Next Obs', fontsize=10)
            axes[4, col].axis('off')
            plt.colorbar(im4, ax=axes[4, col], fraction=0.046)
            
            # Row 5: Model prediction
            pred_img = pred_seq[t, 0].numpy()
            im5 = axes[5, col].imshow(pred_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[5, col].set_title('Prediction', fontsize=10)
            axes[5, col].axis('off')
            plt.colorbar(im5, ax=axes[5, col], fraction=0.046)
        
        plt.tight_layout()
        
        # Convert plot to image and log to TensorBoard
        fig.canvas.draw()
        
        try:
            # Get the actual canvas size after tight_layout
            width, height = fig.canvas.get_width_height()
            
            # Get buffer and calculate actual size
            buffer = fig.canvas.buffer_rgba()
            img_array = np.frombuffer(buffer, dtype=np.uint8)
            
            # Calculate expected size (width * height * 4 channels)
            expected_size = width * height * 4
            actual_size = img_array.size
            
            if actual_size != expected_size:
                # Buffer size mismatch - likely due to DPI scaling
                # Calculate actual dimensions from buffer size
                pixels = actual_size // 4
                actual_height = int(np.sqrt(pixels * height / width))
                actual_width = pixels // actual_height
                img_array = img_array.reshape(actual_height, actual_width, 4)
            else:
                img_array = img_array.reshape(height, width, 4)
            
            # Remove alpha channel to get RGB
            img_array = img_array[:, :, :3]
            
            # Convert to tensor [C, H, W] format for TensorBoard
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            writer.add_image(f'{prefix}/WorldModelPredictions', img_tensor, epoch)
            
            print(f"📊 Saved {prefix.lower()} world model prediction visualization to TensorBoard (step {epoch})")
            
        except Exception as e:
            print(f"⚠️  Failed to save visualization to TensorBoard: {e}")
            import traceback
            traceback.print_exc()
        
        plt.close(fig)


def save_reconstruction_samples(model, test_loader, device, save_path, num_samples=4, use_log_scale=False, sample_randomly=True):
    """Save sample reconstructions for visual inspection with residuals
    
    Args:
        model: The autoencoder model
        test_loader: DataLoader to sample from
        device: Device to run inference on
        save_path: Path to save the reconstruction image
        num_samples: Number of samples to show
        use_log_scale: Whether to apply log-scale to displayed images
        sample_randomly: If True, randomly select samples from different batches. If False, use first batch (consistent tracking)
    """
    model.eval()
    
    with torch.no_grad():
        if sample_randomly:
            # Randomly select a batch to visualize
            import random
            batch_idx = random.randint(0, len(test_loader) - 1)
            for i, (data, _) in enumerate(test_loader):
                if i == batch_idx:
                    break
        else:
            # Always use first batch for consistent tracking
            data, _ = next(iter(test_loader))
        
        data = data.to(device)
        
        # Get reconstructions
        if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
            reconstruction, latent = model(data)
        else:
            reconstruction = model(data)
        
        # Move to CPU for plotting
        data = data.cpu()
        reconstruction = reconstruction.cpu()
        
        # Calculate residuals
        residuals = torch.abs(data - reconstruction)
        
        # Create figure with 3 rows: original, reconstruction, residuals
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        fig.suptitle('Autoencoder Reconstructions (Log Scale)' if use_log_scale else 'Autoencoder Reconstructions')
        
        for i in range(min(num_samples, data.size(0))):
            # Prepare images
      
            original_img = data[i, 0].numpy()
            recon_img = reconstruction[i, 0].numpy()
            residual_img = original_img - recon_img
            
            # Note: If use_log_scale=True, data is already log-scaled during loading
            # So we DON'T apply additional log-scaling here for display
            # The use_log_scale parameter is just for labeling the plot title
            
            # Original
            im0 = axes[0, i].imshow((original_img), cmap='viridis')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            plt.colorbar(im0, ax=axes[0, i], fraction=0.046)
            
            # Reconstruction
            im1 = axes[1, i].imshow(recon_img, cmap='viridis')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
            
            # Residual
            im2 = axes[2, i].imshow(residual_img, cmap='hot')
            axes[2, i].set_title(f'Residual {i+1}')
            axes[2, i].axis('off')
            plt.colorbar(im2, ax=axes[2, i], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, save_path, is_best=False):
    """Save training checkpoint"""
    # If model is wrapped in DataParallel, get the underlying model
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)


def train_world_model(config: WorldModelConfig):
    """Main training function for world models"""
    print("🚀 Starting World Model Training")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Load pretrained autoencoder
    if config.pretrained_autoencoder_path is None:
        raise ValueError("Must provide --pretrained-autoencoder-path for world model training")
    
    print(f"\n📦 Loading pretrained autoencoder from: {config.pretrained_autoencoder_path}")
    
    # Try to load checkpoint - it may have pickled config objects
    try:
        autoencoder_checkpoint = torch.load(config.pretrained_autoencoder_path, map_location=device, weights_only=False)
    except (AttributeError, pickle.UnpicklingError) as e:
        print(f"  ⚠️  Warning: Could not unpickle config from checkpoint: {e}")
        print(f"  Trying to load just the state dict...")
        # Load without unpickling the config
        autoencoder_checkpoint = torch.load(config.pretrained_autoencoder_path, map_location=device, weights_only=True)
    
    # Extract autoencoder model - handle different checkpoint formats
    if isinstance(autoencoder_checkpoint, dict):
        if 'model_state_dict' in autoencoder_checkpoint:
            autoencoder_state = autoencoder_checkpoint['model_state_dict']
            model_config = autoencoder_checkpoint.get('model_config', {})
            arch = autoencoder_checkpoint.get('architecture', None)
        elif 'state_dict' in autoencoder_checkpoint:
            autoencoder_state = autoencoder_checkpoint['state_dict']
            model_config = {}
            arch = None
        else:
            # Assume the checkpoint is just the state dict
            autoencoder_state = autoencoder_checkpoint
            arch = None
            model_config = {}
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(autoencoder_checkpoint)}")
    
    # Infer architecture from state dict if not in checkpoint
    if arch is None:
        # Check if it's an Impala/ResNet-based model (has residual blocks or groupnorm)
        if 'encoder.4.0.conv1.weight' in autoencoder_state or 'encoder.1.weight' in autoencoder_state:
            # Has residual blocks - it's ResNet-based autoencoder
            print(f"  🔍 Detected ResNet-based autoencoder architecture from state dict")
            arch = 'autoencoder_resnet'
        else:
            # Default to CNN
            arch = 'autoencoder_cnn'
    
    # Create autoencoder with detected architecture
    print(f"🔧 Creating autoencoder architecture: {arch}")
    
    # Determine input channels from state dict
    first_conv_key = 'encoder.0.weight' if 'encoder.0.weight' in autoencoder_state else 'encoder.1.weight'
    if first_conv_key in autoencoder_state:
        detected_input_channels = autoencoder_state[first_conv_key].shape[1]
        print(f"  🔍 Detected input channels: {detected_input_channels}")
        config.input_channels = detected_input_channels
    
    # Determine latent dim from bottleneck
    if 'bottleneck_encode.0.weight' in autoencoder_state:
        detected_latent_dim = autoencoder_state['bottleneck_encode.0.weight'].shape[0]
        print(f"  � Detected latent dim: {detected_latent_dim}")
        config.latent_dim = detected_latent_dim
    
    autoencoder = create_model(
        arch=arch,
        input_channels=config.input_channels,
        latent_dim=config.latent_dim,
        input_crop_size=config.input_crop_size,
        **model_config
    )
    autoencoder.load_state_dict(autoencoder_state)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    print(f"✅ Loaded pretrained autoencoder (latent_dim={config.latent_dim})")
    
    # Discover dataset files
    print(f"\n📂 Discovering dataset files in: {config.dataset_path}")
    dataset_abs_path = Path(config.dataset_path).resolve()
    print(f"   Absolute path: {dataset_abs_path}")
    file_paths, dataset_type, metadata = DatasetDiscovery.discover_files(config.dataset_path)
    
    print(f"📊 Found {len(file_paths)} files of type '{dataset_type}'")
    print(f"   Sequence length: {config.sequence_length}")
    
    # Create transforms
    print(f"\n🎨 Setting up data transforms...")
    from utils.transforms import get_autoencoder_transforms
    transforms = get_autoencoder_transforms(
        crop_size=None,  # Cropping handled by encoder
        normalize=True,
        log_scale=config.log_scale
    )
    print(f"   Normalize: True")
    print(f"   Log scale: {config.log_scale}")
    
    # Create world model dataset
    if config.use_episodes:
        print(f"\n📂 Creating EPISODE-BASED dataset...")
        print(f"   Observation key: {config.obs_key}")
        print(f"   Action key: {config.action_key}")
        print(f"   Min episode length: {config.min_episode_length}")
        print(f"   Max episode length: {config.max_episode_length if hasattr(config, 'max_episode_length') else 'None (unlimited)'}")
        print(f"   Batch size: {config.batch_size}")
        
        dataset = WorldModelEpisodeDataset(
            file_paths=file_paths,
            dataset_type=dataset_type,
            transforms=transforms,
            obs_key=config.obs_key,
            action_key=config.action_key,
            min_episode_length=config.min_episode_length,
            max_episode_length=config.max_episode_length if hasattr(config, 'max_episode_length') else None,
            load_in_memory=False,  # Don't preload yet, we need to filter first
            batch_size=None  # Will be set after filtering if preloading
        )
        
        # Filter by max_examples if specified (take first N episodes)
        if config.max_examples:
            original_count = len(dataset.episode_data)
            dataset.episode_data = dataset.episode_data[:config.max_examples]
            print(f"   Limited to {len(dataset.episode_data)} episodes (max_examples={config.max_examples}, was {original_count})")
        
        # Now preload if requested (after filtering)
        # Enable batch-level preloading to eliminate per-batch padding overhead
        if config.load_in_memory:
            dataset.load_in_memory = True
            dataset.batch_size = config.batch_size  # Enable batch preloading
            if dataset.batch_size is not None:
                dataset._preload_batches()  # Pre-collate batches
            else:
                dataset._preload_episodes()  # Just preload episodes
        
        # Collate function and batch_size depend on whether batches are pre-collated
        if dataset.preloaded_batches is not None:
            # Batches are pre-collated, use identity collate and batch_size=1
            collate_fn = lambda x: x[0]  # Just return the pre-collated batch
            batch_size = 1  # DataLoader fetches one pre-collated batch at a time
            print(f"   ✅ Using pre-collated batches (batch_size={config.batch_size} episodes per batch)")
        else:
            # Normal mode: collate episodes per-batch
            collate_fn = collate_episodes_padded
            batch_size = config.batch_size
            print(f"   Using dynamic collation (batch_size={config.batch_size})")
        
    else:
        print(f"\n📂 Creating SEQUENCE-BASED dataset...")
        print(f"   Observation key: {config.obs_key}")
        print(f"   Action key: {config.action_key}")
        print(f"   Load in memory: {config.load_in_memory}")
        
        if config.load_in_memory:
            # Load all data into memory for faster training
            print(f"💾 Loading dataset into memory...")
            if config.use_multiprocessing:
                print(f"   Multiprocessing enabled for parallel data loading")
            dataset = WorldModelSequenceDataset(
                file_paths=file_paths,
                dataset_type=dataset_type,
                sequence_length=config.sequence_length,
                transforms=transforms,
                obs_key=config.obs_key,
                action_key=config.action_key,
                load_in_memory=True,
                max_examples=config.max_examples,
                use_multiprocessing=config.use_multiprocessing
            )
            print(f"✅ Dataset loaded into memory")
        else:
            # Lazy loading with caching
            dataset = WorldModelLazyDataset(
                file_paths=file_paths,
                dataset_type=dataset_type,
                sequence_length=config.sequence_length,
                transforms=transforms,
                obs_key=config.obs_key,
                action_key=config.action_key
            )
        
        collate_fn = collate_sequences
        batch_size = config.batch_size
    
    # Dataset instrumentation - show structure of first example
    print(f"\n📋 Dataset Information:")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Dataset type: {type(dataset).__name__}")
    
    # Safely inspect first example without breaking on structure changes
    try:
        first_example = dataset[0]
        print(f"   Example structure: {type(first_example).__name__}")
        
        # Handle different return types gracefully
        if isinstance(first_example, (tuple, list)):
            print(f"   - Length: {len(first_example)}")
            for i, item in enumerate(first_example):
                if isinstance(item, torch.Tensor):
                    print(f"   - Item {i}: Tensor with shape {tuple(item.shape)}, dtype={item.dtype}")
                elif isinstance(item, np.ndarray):
                    print(f"   - Item {i}: Array with shape {tuple(item.shape)}, dtype={item.dtype}")
                else:
                    print(f"   - Item {i}: {type(item).__name__} = {item if not hasattr(item, '__len__') or len(str(item)) < 50 else '...'}")
        elif isinstance(first_example, dict):
            print(f"   - Keys: {list(first_example.keys())}")
            for key, value in first_example.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: Tensor with shape {tuple(value.shape)}, dtype={value.dtype}")
                elif isinstance(value, np.ndarray):
                    print(f"   - {key}: Array with shape {tuple(value.shape)}, dtype={value.dtype}")
                else:
                    print(f"   - {key}: {type(value).__name__}")
        else:
            print(f"   - Type: {type(first_example).__name__}")
    except Exception as e:
        print(f"   ⚠️  Could not inspect example structure: {e}")
    
    # Split dataset
    total_size = len(dataset)
    if config.max_examples and total_size > config.max_examples:
        print(f"⚠️  Dataset size ({total_size}) exceeds max_examples ({config.max_examples})")
        print(f"   This shouldn't happen - check dataset limiting logic")
    
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"📊 Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders with appropriate collate function and batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    # Infer action_dim from the first batch
    print(f"\n🔍 Inferring action dimension from dataset...")
    first_batch = next(iter(train_loader))
    
    if config.use_episodes:
        # Episode-based with padded batching: first_batch is (obs_padded, actions_padded, next_obs_padded, lengths, mask)
        _, actions_padded, _, _, _ = first_batch
        action_dim = actions_padded.shape[-1]
    else:
        # Sequence-based: first_batch is a tuple of tensors
        _, actions, _ = first_batch
        action_dim = actions.shape[-1]
    
    print(f"   Detected action_dim: {action_dim}")
    
    # Create world model
    print(f"\n🌍 Creating world model...")
    print(f"   Action dim: {action_dim}")
    print(f"   LSTM hidden dim: {config.hidden_dim}")
    print(f"   LSTM layers: {config.num_lstm_layers}")
    print(f"   Action hidden dim: {config.action_hidden_dim}")
    print(f"   Fusion hidden dim: {config.fusion_hidden_dim}")
    print(f"   Freeze encoder: {config.freeze_encoder}")
    print(f"   Freeze decoder: {config.freeze_decoder}")
    
    model = create_world_model_from_autoencoder(
        autoencoder=autoencoder,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_lstm_layers,
        action_hidden_dim=config.action_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        freeze_encoder=config.freeze_encoder,
        freeze_decoder=config.freeze_decoder
    )
    model = model.to(device)
    
    # Wrap model in DataParallel if requested
    if config.use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🔧 Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif config.use_data_parallel:
        print(f"⚠️  DataParallel requested but only {torch.cuda.device_count()} GPU(s) available - using single GPU/CPU")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ World model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Check if there are any trainable parameters
    if trainable_params == 0:
        raise ValueError(
            "❌ No trainable parameters! Cannot train model.\n"
            "   You have frozen both encoder and decoder, leaving no parameters to train.\n"
            "   The world model consists of: encoder (frozen) → LSTM → decoder (frozen)\n"
            "   At least one of encoder or decoder must be trainable.\n"
            "   Suggestion: Remove --freeze-decoder flag to train the decoder."
        )
    
    # Setup loss function
    if config.loss_function == "mse":
        base_criterion = nn.MSELoss()
    elif config.loss_function == "mae":
        base_criterion = nn.L1Loss()
    elif config.loss_function == "smooth_l1":
        base_criterion = nn.SmoothL1Loss()
    elif config.loss_function == "huber":
        base_criterion = nn.HuberLoss(delta=config.huber_delta)
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")
    
    # Wrap with MaskedLoss for episode-based training with padding
    if config.use_episodes:
        criterion = MaskedLoss(base_criterion)
        print(f"📉 Loss function: {config.loss_function} (wrapped with MaskedLoss for episode batching)")
    else:
        criterion = base_criterion
        print(f"📉 Loss function: {config.loss_function}")
    
    # Setup optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    print(f"⚡ Optimizer: {config.optimizer} (lr={config.learning_rate})")
    
    # Setup learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )
        print(f"📉 Learning Rate Scheduler: ReduceLROnPlateau")
        print(f"   Patience: {config.scheduler_patience} epochs")
        print(f"   Factor: {config.scheduler_factor}")
        print(f"   Min LR: {config.scheduler_min_lr}")
    else:
        print(f"📉 Learning Rate Scheduler: None (constant LR)")
    
    # Setup Automatic Mixed Precision (AMP)
    scaler = None
    if config.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print(f"⚡ Automatic Mixed Precision (AMP): Enabled")
        print(f"   Using torch.cuda.amp.GradScaler for gradient scaling")
    elif config.use_amp and device.type != 'cuda':
        print(f"⚠️  AMP requested but device is {device.type}, disabling AMP")
        print(f"   AMP only works with CUDA devices")
    else:
        print(f"⚡ Automatic Mixed Precision (AMP): Disabled")
    
    # Setup TensorBoard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path(config.runs_dir) / f"{config.run_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir))
    
    print(f"📁 Results directory: {save_dir}")
    print(f"📊 TensorBoard: tensorboard --logdir={config.runs_dir}")
    
    # Save model summary to file
    print(f"\n💾 Saving model summary and configuration...")
    summary_file = save_dir / "model_summary.txt"
    config_file = save_dir / "config.txt"
    
    # Write model summary using torchinfo
    model_summary_text = ""
    config_text = ""
    
    try:
        # Get observation shape from a sample batch
        sample_batch = next(iter(train_loader))
        
        # Determine batch format by inspecting the actual structure
        if isinstance(sample_batch, list):
            # Both episode and sequence formats can return lists
            # Check first element structure
            first_item = sample_batch[0]
            
            # Handle nested list structure (DataLoader may wrap collate output in extra list)
            if isinstance(first_item, list) and len(first_item) > 0:
                # Unwrap one level
                sample_batch = first_item
                first_item = sample_batch[0]
            
            if isinstance(first_item, tuple) and len(first_item) == 4:
                # Episode format: list of (obs, actions, next_obs, length) tuples
                sample_obs, sample_actions, _, _ = first_item
                obs_shape = sample_obs.shape  # [seq_len, C, H, W]
                seq_len, C, H, W = obs_shape
            elif isinstance(first_item, tuple) and len(first_item) == 3:
                # Sequence format returned as list: list of (obs, actions, next_obs) tuples
                sample_obs, sample_actions, _ = first_item
                obs_shape = sample_obs.shape  # [seq_len, C, H, W]
                seq_len, C, H, W = obs_shape
            else:
                raise ValueError(f"Unexpected batch format: list with items of type {type(first_item)}")
        elif isinstance(sample_batch, dict):
            # Dictionary format (less common)
            sample_obs = sample_batch['observations']
            obs_shape = sample_obs.shape
            B, seq_len, C, H, W = obs_shape
        else:
            raise ValueError(f"Unexpected batch format: {type(sample_batch)}")
        
        # Create dummy input to get model summary
        dummy_obs = torch.randn(config.batch_size, config.sequence_length, C, H, W).to(device)
        dummy_action = torch.randn(config.batch_size, config.sequence_length, action_dim).to(device)
        
        # Get summary as string
        model_summary = summary(
            model, 
            input_data=[dummy_obs, dummy_action],
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=4
        )
        
        model_summary_text = "=" * 80 + "\n"
        model_summary_text += "WORLD MODEL ARCHITECTURE SUMMARY\n"
        model_summary_text += "=" * 80 + "\n\n"
        model_summary_text += str(model_summary)
        model_summary_text += "\n\n" + "=" * 80 + "\n"
        model_summary_text += "PARAMETER SUMMARY\n"
        model_summary_text += "=" * 80 + "\n"
        model_summary_text += f"Total parameters:      {total_params:,}\n"
        model_summary_text += f"Trainable parameters:  {trainable_params:,}\n"
        model_summary_text += f"Frozen parameters:     {total_params - trainable_params:,}\n"
        model_summary_text += f"Trainable percentage:  {100 * trainable_params / total_params:.2f}%\n"
        
        # Write to file
        with open(summary_file, 'w') as f:
            f.write(model_summary_text)
        
        # Add to TensorBoard as text (using code block for better formatting)
        try:
            # Format as code block for TensorBoard
            tb_summary = f"```\n{model_summary_text}\n```"
            writer.add_text('Model_Architecture', tb_summary, global_step=0)
            print(f"   ✅ Model summary saved to: {summary_file}")
            print(f"   ✅ Model summary added to TensorBoard")
        except Exception as tb_e:
            print(f"   ⚠️  Failed to add model summary to TensorBoard: {tb_e}")
            print(f"   ✅ Model summary saved to file: {summary_file}")
    except Exception as e:
        print(f"   ⚠️  Failed to generate model summary: {e}")
        import traceback
        traceback.print_exc()
        print(f"   ⚠️  Failed to save model summary: {e}")
    
    # Write configuration to file
    try:
        config_text = "=" * 80 + "\n"
        config_text += "TRAINING CONFIGURATION\n"
        config_text += "=" * 80 + "\n\n"
        
        # Training settings
        config_text += "TRAINING SETTINGS:\n"
        config_text += f"  Run name:              {config.run_name}\n"
        config_text += f"  Batch size:            {config.batch_size}\n"
        config_text += f"  Learning rate:         {config.learning_rate}\n"
        config_text += f"  Number of epochs:      {config.num_epochs}\n"
        config_text += f"  Device:                {device}\n"
        config_text += f"  Optimizer:             {config.optimizer}\n"
        config_text += f"  Loss function:         {config.loss_function}\n"
        config_text += f"  Weight decay:          {config.weight_decay}\n"
        config_text += f"  Random seed:           {config.seed}\n"
        config_text += f"  Use AMP:               {config.use_amp}\n"
        
        # Scheduler settings
        config_text += f"\nSCHEDULER SETTINGS:\n"
        config_text += f"  Use scheduler:         {config.use_scheduler}\n"
        if config.use_scheduler:
            config_text += f"  Scheduler type:        ReduceLROnPlateau\n"
            config_text += f"  Patience:              {config.scheduler_patience}\n"
            config_text += f"  Factor:                {config.scheduler_factor}\n"
            config_text += f"  Min LR:                {config.scheduler_min_lr}\n"
        
        # Model architecture
        config_text += f"\nMODEL ARCHITECTURE:\n"
        config_text += f"  Hidden dim (LSTM):     {config.hidden_dim}\n"
        config_text += f"  Num LSTM layers:       {config.num_lstm_layers}\n"
        config_text += f"  Action hidden dim:     {config.action_hidden_dim}\n"
        config_text += f"  Fusion hidden dim:     {config.fusion_hidden_dim}\n"
        config_text += f"  Latent dim:            {config.latent_dim}\n"
        config_text += f"  Input channels:        {config.input_channels}\n"
        config_text += f"  Input crop size:       {config.input_crop_size}\n"
        config_text += f"  Sequence length:       {config.sequence_length}\n"
        config_text += f"  Freeze encoder:        {config.freeze_encoder}\n"
        config_text += f"  Freeze decoder:        {config.freeze_decoder}\n"
        
        # Data settings
        config_text += f"\nDATA SETTINGS:\n"
        config_text += f"  Dataset path:          {config.dataset_path}\n"
        config_text += f"  Dataset absolute path: {dataset_abs_path}\n"
        config_text += f"  Dataset type:          {dataset_type}\n"
        config_text += f"  Total files:           {len(file_paths)}\n"
        config_text += f"  Total examples:        {len(dataset)}\n"
        config_text += f"  Observation key:       {config.obs_key}\n"
        config_text += f"  Action key:            {config.action_key}\n"
        config_text += f"  Load in memory:        {config.load_in_memory}\n"
        config_text += f"  Log scale:             {config.log_scale}\n"
        config_text += f"  Max examples:          {config.max_examples}\n"
        config_text += f"  Num workers:           {config.num_workers}\n"
        config_text += f"  Prefetch factor:       {config.prefetch_factor}\n"
        config_text += f"  Persistent workers:    {config.persistent_workers}\n"
        
        # Output settings
        config_text += f"\nOUTPUT SETTINGS:\n"
        config_text += f"  Model save path:       {config.model_save_path}\n"
        config_text += f"  Save model:            {config.save_model}\n"
        config_text += f"  Checkpoint interval:   {config.checkpoint_interval}\n"
        config_text += f"  Reconstruction interval: {config.reconstruction_interval}\n"
        
        # Autoencoder settings
        config_text += f"\nPRETRAINED AUTOENCODER:\n"
        config_text += f"  Path:                  {config.pretrained_autoencoder_path}\n"
        
        # Write to file
        with open(config_file, 'w') as f:
            f.write(config_text)
        
        # Add to TensorBoard as text (using code block for better formatting)
        try:
            # Format as code block for TensorBoard
            tb_config = f"```\n{config_text}\n```"
            writer.add_text('Training_Configuration', tb_config, global_step=0)
            print(f"   ✅ Configuration saved to: {config_file}")
            print(f"   ✅ Configuration added to TensorBoard")
        except Exception as tb_e:
            print(f"   ⚠️  Failed to add configuration to TensorBoard: {tb_e}")
            print(f"   ✅ Configuration saved to file: {config_file}")
    except Exception as e:
        print(f"   ⚠️  Failed to save configuration: {e}")
        import traceback
        traceback.print_exc()
    
    # Flush writer to ensure all data is written
    try:
        writer.flush()
        print(f"   ✅ TensorBoard writer flushed")
    except Exception as e:
        print(f"   ⚠️  Failed to flush TensorBoard writer: {e}")
    
    # Training loop
    print(f"\n🏋️  Starting training for {config.num_epochs} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_times = []
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Sequence length: {config.sequence_length} steps")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Training
        train_start = time.time()
        print(f"\n⏱️  Starting training phase...")
        if config.use_episodes:
            train_loss, train_sequences = train_world_model_epoch_episodes_batched(
                model, train_loader, criterion, optimizer, device, 
                sequence_length=config.sequence_length, scaler=scaler
            )
        else:
            train_loss = train_world_model_epoch(model, train_loader, criterion, optimizer, device, scaler)
            train_sequences = len(train_loader) * config.batch_size  # Approximate
        train_losses.append(train_loss)
        train_time = time.time() - train_start
        print(f"⏱️  Training completed in {train_time:.2f}s")
        
        # Calculate training throughput
        train_sequences_per_sec = train_sequences / train_time if train_time > 0 else 0
        
        # Validation
        val_start = time.time()
        print(f"⏱️  Starting validation phase...")
        if config.use_episodes:
            val_loss, val_sequences = validate_world_model_epoch_episodes_batched(
                model, val_loader, criterion, device, 
                sequence_length=config.sequence_length
            )
        else:
            val_loss = validate_world_model_epoch(model, val_loader, criterion, device)
            val_sequences = len(val_loader) * config.batch_size  # Approximate
        val_losses.append(val_loss)
        val_time = time.time() - val_start
        print(f"⏱️  Validation completed in {val_time:.2f}s")
        
        # Calculate validation throughput
        val_sequences_per_sec = val_sequences / val_time if val_time > 0 else 0
        
        # Visualize world model predictions (only on certain epochs to save memory)
        if (epoch + 1) % config.reconstruction_interval == 0:
            # Training visualization
            print(f"🎨 Creating training world model prediction visualization...")
            try:
                visualize_world_model_predictions(model, train_loader, device, writer, epoch, num_timesteps=4, prefix='Train')
            except Exception as e:
                print(f"⚠️  Training visualization failed: {e}")
                print(f"   Continuing without training visualization...")
            
            # Validation visualization
            print(f"🎨 Creating validation world model prediction visualization...")
            try:
                visualize_world_model_predictions(model, val_loader, device, writer, epoch, num_timesteps=4, prefix='Validation')
            except Exception as e:
                print(f"⚠️  Validation visualization failed: {e}")
                print(f"   Continuing without validation visualization...")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Throughput/Train_Sequences_Per_Second', train_sequences_per_sec, epoch)
        writer.add_scalar('Throughput/Val_Sequences_Per_Second', val_sequences_per_sec, epoch)
        
        # Update learning rate based on validation loss
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Calculate epoch timing and ETA
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = config.num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_hours = eta_seconds / 3600
        
        # Log timing metrics to TensorBoard
        writer.add_scalar('Timing/Epoch_Time_Seconds', epoch_time, epoch)
        writer.add_scalar('Timing/Epoch_Time_Minutes', epoch_time / 60, epoch)
        writer.add_scalar('Timing/Average_Epoch_Time_Minutes', avg_epoch_time / 60, epoch)
        writer.add_scalar('Timing/ETA_Hours', eta_hours, epoch)
        
        # Print epoch summary
        if eta_hours >= 1:
            eta_str = f"{eta_hours:.1f}h"
        elif eta_hours * 60 >= 1:
            eta_str = f"{eta_hours * 60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"
        
        print(f"\n📊 Epoch {epoch+1}/{config.num_epochs} Summary:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   Best Val Loss: {best_val_loss:.6f} (Epoch {val_losses.index(min(val_losses)) + 1})")
        print(f"   Epoch Time: {epoch_time/60:.2f}m | Avg: {avg_epoch_time/60:.2f}m | ETA: {eta_str}")
        print(f"   Sequences per epoch: {len(train_loader) * config.batch_size}")
        print(f"   Total timesteps per epoch: {len(train_loader) * config.batch_size * config.sequence_length}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🏆 New best validation loss: {best_val_loss:.6f}")
            if config.save_model:
                # Get the underlying model if wrapped in DataParallel
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                checkpoint_path = save_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_path)
                print(f"💾 Saved best model to: {checkpoint_path}")
        
        # Save periodic checkpoints
        if (epoch + 1) % config.checkpoint_interval == 0:
            # Get the underlying model if wrapped in DataParallel
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint to: {checkpoint_path}")
    
    # Test set evaluation
    print(f"\n🧪 Evaluating on test set...")
    if config.use_episodes:
        test_loss = validate_world_model_epoch_episodes(
            model, test_loader, criterion, device, sequence_length=config.sequence_length
        )
    else:
        test_loss = validate_world_model_epoch(model, test_loader, criterion, device)
    print(f"📊 Test Loss: {test_loss:.6f}")
    
    writer.add_scalar('Loss/Test', test_loss, config.num_epochs)
    
    # Save final model
    if config.save_model:
        # Get the underlying model if wrapped in DataParallel
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        final_model_path = save_dir / "final_model.pth"
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': config,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss
        }, final_model_path)
        print(f"💾 Saved final model to: {final_model_path}")
    
    # Plot training curves
    if config.plot_losses:
        plot_path = save_dir / "training_curves.png"
        plot_training_curves(train_losses, val_losses, str(plot_path))
    
    writer.flush()
    writer.close()
    
    print("\n✅ Training completed successfully!")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    print(f"📊 Final test loss: {test_loss:.6f}")
    print(f"📁 Results saved to: {save_dir}")
    print("=" * 60)


def load_config_from_json(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="World Model Training for Next-Observation Prediction")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Run settings
    parser.add_argument("--run-name", type=str, default="world_model_default",
                       help="Name for this training run (used in output directory)")
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Root directory for runs (default: runs)")
    
    # Dataset settings
    parser.add_argument("--dataset-path", type=str,
                       help="Path to dataset directory with observations and actions")
    parser.add_argument("--obs-key", type=str, default="observations",
                       help="HDF5/NPZ key for observations (default: observations)")
    parser.add_argument("--action-key", type=str, default="sa_incremental_actions",
                       help="HDF5/NPZ key for actions (default: sa_incremental_actions)")
    
    # Training settings
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/mps/cpu)")
    parser.add_argument("--use-data-parallel", action="store_true",
                       help="Use DataParallel for multi-GPU training (uses all available GPUs)")
    
    # Model settings - pretrained autoencoder
    parser.add_argument("--pretrained-autoencoder-path", type=str,
                       help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze encoder weights")
    parser.add_argument("--freeze-decoder", action="store_true", default=False,
                       help="Freeze decoder weights (default: False)")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension from autoencoder")
    parser.add_argument("--input-crop-size", type=int, default=None, help="Input crop size")
    
    # World model architecture
    parser.add_argument("--hidden-dim", type=int, default=512, help="LSTM hidden dimension")
    parser.add_argument("--num-lstm-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--action-hidden-dim", type=int, default=128, help="Action MLP hidden dimension")
    parser.add_argument("--fusion-hidden-dim", type=int, default=512, help="Fusion MLP hidden dimension")
    
    # Sequence settings for BPTT
    parser.add_argument("--sequence-length", type=int, help="Sequence length for BPTT")
    
    # Episode-based training (NEW)
    parser.add_argument("--use-episodes", action="store_true",
                       help="Use episode-based training instead of sequence-based (enables proper long-term learning)")
    parser.add_argument("--episode-batch-size", type=int, default=4,
                       help="Number of episodes per batch (for episode-based training)")
    parser.add_argument("--min-episode-length", type=int, default=20,
                       help="Minimum episode length for episode-based training")
    
    # Loss and optimization
    parser.add_argument("--loss-function", type=str,
                       choices=["mse", "mae", "smooth_l1", "huber"],
                       help="Loss function")
    parser.add_argument("--optimizer", type=str,
                       choices=["adam", "adamw", "sgd"],
                       help="Optimizer")
    
    # Learning rate scheduler settings
    parser.add_argument("--use-scheduler", action="store_true",
                       help="Use learning rate scheduler (ReduceLROnPlateau)")
    parser.add_argument("--no-scheduler", action="store_true",
                       help="Disable learning rate scheduler")
    parser.add_argument("--scheduler-patience", type=int, default=50,
                       help="Epochs to wait before reducing LR (default: 50)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                       help="Factor to reduce LR by (default: 0.5)")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-7,
                       help="Minimum learning rate (default: 1e-7)")
    
    # Output settings
    parser.add_argument("--model-save-path", type=str,
                       help="Path to save trained model")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    
    # Data settings
    parser.add_argument("--max-examples", type=int, help="Limit dataset size for debugging")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                       help="Number of batches to prefetch per worker (default: 2)")
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers",
                       help="Disable persistent workers (restart workers between epochs)")
    parser.add_argument("--use-multiprocessing", action="store_true",
                       help="Enable parallel data loading (8-16x faster but may have compatibility issues on some systems)")
    parser.add_argument("--checkpoint-interval", type=int,
                       help="Save model checkpoints every N epochs")
    parser.add_argument("--reconstruction-interval", type=int,
                       help="Save reconstruction visualizations every N epochs")
    
    # Training optimizations
    parser.add_argument("--use-amp", action="store_true",
                       help="Use Automatic Mixed Precision (AMP) training")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp",
                       help="Disable AMP training")
    
    # Parse arguments but don't use defaults yet
    args, unknown = parser.parse_known_args()
    
    # Load config from JSON file if provided
    config_dict = {}
    if args.config:
        print(f"📄 Loading config from: {args.config}")
        config_dict = load_config_from_json(args.config)
        print(f"✅ Loaded {len(config_dict)} config values from {args.config}")
    
    # Build a set of which arguments were explicitly provided on command line
    import sys
    cli_provided = set()
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            arg_name = sys.argv[i][2:].replace('-', '_')
            cli_provided.add(arg_name)
        i += 1
    
    # Helper function with correct priority: CLI > config > default
    def get_value(arg_name, default_value=None):
        """Get config value with priority: command line > config file > default"""
        # 1. Command line (highest priority)
        if arg_name in cli_provided:
            cli_value = getattr(args, arg_name, None)
            # For store_true/store_false actions, the value will be set even if not provided
            # So we trust that if it's in cli_provided, we should use the args value
            return cli_value
        
        # 2. Config file (medium priority)
        if arg_name in config_dict:
            return config_dict[arg_name]
        
        # 3. Default value (lowest priority)
        return default_value
    
    # Validate required arguments
    if not get_value('dataset_path'):
        parser.error("--dataset-path is required (or provide via config file)")
    if not get_value('pretrained_autoencoder_path'):
        parser.error("--pretrained-autoencoder-path is required (or provide via config file)")
    
    # Special handling for use_scheduler (can be set with --use-scheduler or --no-scheduler)
    use_scheduler_value = True  # Default
    if 'no_scheduler' in cli_provided:
        use_scheduler_value = False
    elif 'use_scheduler' in cli_provided:
        use_scheduler_value = True
    elif 'use_scheduler' in config_dict:
        use_scheduler_value = config_dict['use_scheduler']
    
    # Create config with proper priority: CLI > config file > defaults
    config = WorldModelConfig(
        dataset_path=get_value('dataset_path'),
        run_name=get_value('run_name', 'world_model_default'),
        runs_dir=get_value('runs_dir', 'runs'),
        batch_size=get_value('batch_size', 4),
        learning_rate=get_value('learning_rate', 1e-4),
        num_epochs=get_value('num_epochs', 100),
        device=get_value('device', 'auto'),
        pretrained_autoencoder_path=get_value('pretrained_autoencoder_path'),
        hidden_dim=get_value('hidden_dim', 512),
        num_lstm_layers=get_value('num_lstm_layers', 1),
        action_hidden_dim=get_value('action_hidden_dim', 128),
        fusion_hidden_dim=get_value('fusion_hidden_dim', 512),
        freeze_encoder=get_value('freeze_encoder', False),
        freeze_decoder=get_value('freeze_decoder', False),
        latent_dim=get_value('latent_dim', 256),
        input_crop_size=get_value('input_crop_size'),
        sequence_length=get_value('sequence_length', 10),
        use_episodes=get_value('use_episodes', False),
        min_episode_length=get_value('min_episode_length', 20),
        max_episode_length=get_value('max_episode_length'),
        obs_key=get_value('obs_key', 'observations'),
        action_key=get_value('action_key', 'sa_incremental_actions'),
        loss_function=get_value('loss_function', 'mse'),
        optimizer=get_value('optimizer', 'adam'),
        use_scheduler=use_scheduler_value,
        scheduler_patience=get_value('scheduler_patience', 50),
        scheduler_factor=get_value('scheduler_factor', 0.5),
        scheduler_min_lr=get_value('scheduler_min_lr', 1e-7),
        model_save_path=get_value('model_save_path', 'saved_models/world_model.pth'),
        save_model=not get_value('no_save', False),
        max_examples=get_value('max_examples'),
        seed=get_value('seed', 42),
        num_workers=get_value('num_workers', 4),
        prefetch_factor=get_value('prefetch_factor', 2),
        persistent_workers=get_value('persistent_workers', True),
        use_multiprocessing=get_value('use_multiprocessing', False),
        use_data_parallel=get_value('use_data_parallel', False),
        checkpoint_interval=get_value('checkpoint_interval', 10),
        reconstruction_interval=get_value('reconstruction_interval', 1),
        log_scale=get_value('log_scale', False),
        load_in_memory=get_value('load_in_memory', False),
        use_amp=get_value('use_amp', False)
    )
    
    # Print configuration
    print("🔧 World Model Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_world_model(config)


if __name__ == "__main__":
    main()
