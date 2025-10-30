#!/usr/bin/env python3
"""
Autoencoder Training Script

Trains convolutional autoencoders on optomech observation data for unsupervised representation learning.
Based on the proven infrastructure from optomech/supervised_ml/train_sml_model.py.

Features:
- Multiple autoencoder architectures (CNN, ResNet-based)
- Separable encoder/decoder components for later reuse
- Reconstruction loss optimization
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model architectures and utilities
from models import create_model, AutoEncoderCNN, AutoEncoderResNet
from models.model_utils import save_trained_model, ModelSaver

# Import unified dataset utilities
from utils.datasets import AutoencoderDataset, LazyDataset
from utils.data_loading import DatasetDiscovery, FileLoader, CacheManager
from utils.transforms import center_crop_transform, get_autoencoder_transforms

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
        
        # For autoencoder, target is the same as input
        return obs_tensor, obs_tensor


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
    no_dataparallel: bool = False  # Disable DataParallel for multi-GPU training


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device for training"""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    print(f"🔧 Using device: {device}")
    return device


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
            running_loss += loss.item()
            if batch_idx == 0:
                print(f"✅ Val Batch {batch_idx}: Loss = {loss.item():.6f}")
                print(f"🎯 First validation batch completed successfully!")
    
    val_time = time.time() - val_start
    avg_loss = running_loss / len(val_loader)
    print(f"🏁 Validation epoch complete in {val_time/60:.2f} minutes")
    print(f"   Average loss: {avg_loss:.6f}")
    return avg_loss


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
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def train_autoencoder(config: AutoencoderConfig):
    """Main training function"""
    print("🚀 Starting Autoencoder Training")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Discover dataset files (memory efficient)
    print(f"\n📂 Discovering dataset files in: {config.dataset_path}")
    file_paths, dataset_type, metadata = DatasetDiscovery.discover_files(config.dataset_path)
    
    if config.log_scale:
        print(f"📈 Log-scaling enabled: Observations will be log-transformed for better dynamic range")
    
    # Determine if we should use lazy loading or in-memory loading
    total_obs = metadata['total_observations']
    
    # User-requested in-memory loading
    if config.load_in_memory:
        use_lazy_loading = False
        print(f"💾 User requested in-memory loading for dataset ({total_obs} observations)")
        print(f"⚠️  This will load the entire dataset into RAM")
    # For very small datasets, we can still load into memory
    elif config.max_examples and config.max_examples <= 1000:
        use_lazy_loading = False
        print(f"💡 Using in-memory loading for small dataset ({config.max_examples} examples)")
    elif total_obs <= 1000:
        use_lazy_loading = False
        print(f"💡 Using in-memory loading for small dataset ({total_obs} examples)")
    else:
        use_lazy_loading = True
        print(f"💾 Using lazy loading for large dataset ({total_obs} total observations)")
        print(f"💡 Use --load-in-memory flag to load entire dataset into RAM for faster training")
    
    # Create dataset using unified utilities
    if use_lazy_loading:
        print(f"\n📂 Creating lazy-loading dataset...")
        dataset = LazyDataset(
            dataset_path=config.dataset_path,
            task_type='autoencoder',
            input_crop_size=config.input_crop_size,
            max_examples=config.max_examples,
            use_cache=True,
            log_scale=config.log_scale
        )
    elif config.load_in_memory:
        # Use new in-memory dataset for fastest training
        print(f"\n📂 Loading dataset into memory...")
        dataset = InMemoryAutoencoderDataset(
            dataset_path=config.dataset_path,
            input_crop_size=config.input_crop_size,
            max_examples=config.max_examples,
            log_scale=config.log_scale
        )
    else:
        # Use old method for small datasets
        print(f"\n📂 Loading small dataset into memory...")
        observations = load_data_from_dataset(config.dataset_path, config.max_examples)
        dataset = AutoencoderDataset(
            observations=observations,
            input_crop_size=config.input_crop_size
        )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"📊 Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda'
    )
    
    # Get input dimensions from metadata
    obs_shape = metadata['observation_shape']
    
    # Apply cropping if specified
    if config.input_crop_size:
        effective_shape = (obs_shape[0], config.input_crop_size, config.input_crop_size)
        print(f"🔄 Input will be center-cropped from {obs_shape} to {effective_shape}")
        obs_shape = effective_shape
    
    input_channels = obs_shape[0] if len(obs_shape) == 3 else 1
    
    print(f"🔍 Input channels detected: {input_channels}")
    print(f"🔍 Input shape: {obs_shape}")
    
    # Create model
    # Determine effective input size after any cropping
    effective_input_size = config.input_crop_size if config.input_crop_size else obs_shape[-1]
    
    print(f"\n🏗️  Creating model: {config.arch}")
    print(f"🔍 Effective input size: {effective_input_size}x{effective_input_size}")
    model = create_model(
        arch=config.arch,
        input_channels=input_channels,
        latent_dim=config.latent_dim,
        input_size=effective_input_size,
        input_crop_size=config.input_crop_size
    )
    
    # Print model summary
    print("\n📋 Model Summary:")
    try:
        sample_input = torch.randn(1, input_channels, obs_shape[-2], obs_shape[-1])
        summary(model, input_data=sample_input, verbose=0)
    except:
        print(f"  Model: {config.arch}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Multi-GPU setup
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1 and not config.no_dataparallel:
        print(f"🔧 Using DataParallel with {gpu_count} GPUs")
        model = DataParallel(model)
    elif gpu_count > 1 and config.no_dataparallel:
        print(f"⚠️  Multiple GPUs detected ({gpu_count}) but DataParallel is disabled")
        print(f"   Training will only use GPU 0")
    
    model.to(device)
    
    # Create loss function
    criterion = create_loss_function(config.loss_function, config.huber_delta)
    
    # Create optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create scheduler
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.num_epochs // 3, gamma=0.1)
    else:
        scheduler = None
    
    # Setup output directory
    timestamp = int(time.time())
    save_dir = Path(f"runs/autoencoder_{config.arch}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Setup TensorBoard
    writer = SummaryWriter(save_dir / "tensorboard")
    
    # Resume from checkpoint if specified
    start_epoch = config.start_epoch
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if config.resume_from:
        print(f"📂 Resuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        
        if gpu_count > 1 and not config.no_dataparallel:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"🎯 Training Configuration:")
    print(f"  Architecture: {config.arch}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Loss function: {config.loss_function}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {device}")
    print(f"  Lazy loading: {use_lazy_loading}")
    print(f"  DataParallel: {'Disabled' if config.no_dataparallel else 'Enabled' if gpu_count > 1 else 'N/A (single GPU)'}")
    
    # Training loop
    print(f"\n🏃 Starting training for {config.num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n🎯 Starting epoch {epoch+1}/{config.num_epochs}")
        epoch_start_time = time.time()
        
        # Train
        print(f"🚂 Epoch {epoch+1}: Starting training phase...")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"✅ Epoch {epoch+1}: Training complete. Loss: {train_loss:.6f}")
        
        # Validate
        print(f"🔍 Epoch {epoch+1}: Starting validation phase...")
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"✅ Epoch {epoch+1}: Validation complete. Loss: {val_loss:.6f}")
        
        # Save reconstruction samples after validation
        print(f"🖼️  Epoch {epoch+1}: Saving reconstruction samples...")
        samples_path = save_dir / f"reconstruction_epoch_{epoch+1:03d}.png"
        save_reconstruction_samples(model, val_loader, device, str(samples_path), 
                                   use_log_scale=config.log_scale, sample_randomly=True)
        print(f"✅ Epoch {epoch+1}: Reconstruction samples saved")
        
        # Update scheduler
        print(f"⚙️  Epoch {epoch+1}: Updating scheduler...")
        if scheduler:
            scheduler.step()
        print(f"✅ Epoch {epoch+1}: Scheduler updated")
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log to TensorBoard
        print(f"📊 Epoch {epoch+1}: Logging to TensorBoard...")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"✅ Epoch {epoch+1}: TensorBoard logging complete")
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎉 New best validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint periodically
        print(f"💾 Epoch {epoch+1}: Checking if checkpoint save needed...")
        if (epoch + 1) % 10 == 0 or is_best:
            checkpoint_path = save_dir / f"autoencoder_checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, 
                          config, str(checkpoint_path), is_best)
    
    # Save final model using model utilities
    if config.save_model:
        # Get the actual model (unwrap from DataParallel if needed)
        save_model = model.module if gpu_count > 1 else model
        
        # Create model ID based on config
        model_id = f"autoencoder_{config.arch}_{timestamp}"
        
        # Prepare model config for reproducibility
        model_config = {
            'input_channels': input_channels,
            'latent_dim': config.latent_dim,
            'input_crop_size': config.input_crop_size
        }
        
        # Prepare training info
        training_info = {
            'dataset_path': config.dataset_path,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'best_val_loss': float(best_val_loss),
            'loss_function': config.loss_function,
            'optimizer': config.optimizer,
            'train_split': config.train_split,
            'val_split': config.val_split,
            'test_split': config.test_split,
            'lazy_loading_used': use_lazy_loading
        }
        
        # Create example input for TorchScript
        example_input = torch.randn(1, input_channels, 
                                   obs_shape[-2], obs_shape[-1]).to(device)
        
        # Save with full metadata
        save_path = save_trained_model(
            model=save_model,
            model_id=model_id,
            architecture=config.arch,
            model_config=model_config,
            training_info=training_info,
            task="autoencoder",
            save_dir=str(save_dir.parent / "saved_models"),
            example_input=example_input
        )
        
        print(f"\n💾 Model saved with full metadata:")
        print(f"   Model ID: {model_id}")
        print(f"   Path: {save_path}")
        print(f"   Encoder/Decoder: Available for reuse")
        print(f"   TorchScript: Available for deployment")
    
    # Test set evaluation
    print(f"\n🧪 Evaluating on test set...")
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"📊 Test Loss: {test_loss:.6f}")
    
    # Save reconstruction samples
    print(f"🖼️  Saving reconstruction samples...")
    samples_path = save_dir / "reconstruction_samples.png"
    save_reconstruction_samples(model, test_loader, device, str(samples_path), 
                               use_log_scale=config.log_scale, sample_randomly=False)
    
    # Plot training curves
    if config.plot_losses:
        plot_path = save_dir / "training_curves.png"
        plot_training_curves(train_losses, val_losses, str(plot_path))
    
    # Close TensorBoard writer
    writer.close()
    
    print("\n✅ Training completed successfully!")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    print(f"📊 Final test loss: {test_loss:.6f}")
    print(f"📁 Results saved to: {save_dir}")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Autoencoder Training for Representation Learning")
    
    # Dataset settings
    parser.add_argument("--dataset-path", type=str, default="datasets/sml_100k_dataset",
                       help="Path to dataset directory")
    
    # Training settings
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    # Model settings
    parser.add_argument("--arch", type=str, default="autoencoder_cnn",
                       choices=["autoencoder_cnn", "autoencoder_resnet"],
                       help="Autoencoder architecture")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--input-crop-size", type=int, default=None, help="Input crop size")
    
    # Loss and optimization
    parser.add_argument("--loss-function", type=str, default="mse",
                       choices=["mse", "mae", "smooth_l1", "huber"],
                       help="Loss function")
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "adamw", "sgd"],
                       help="Optimizer")
    
    # Output settings
    parser.add_argument("--model-save-path", type=str, default="saved_models/autoencoder.pth",
                       help="Path to save trained model")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint")
    
    # Data settings
    parser.add_argument("--max-examples", type=int, help="Limit dataset size for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-scale", action="store_true", help="Apply log-scaling to observations")
    parser.add_argument("--load-in-memory", action="store_true", 
                       help="Load entire dataset into memory for fastest training (requires sufficient RAM)")
    parser.add_argument("--no-dataparallel", action="store_true",
                       help="Disable DataParallel for multi-GPU training")
    
    args = parser.parse_args()
    
    # Create config
    config = AutoencoderConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        arch=args.arch,
        latent_dim=args.latent_dim,
        input_crop_size=args.input_crop_size,
        loss_function=args.loss_function,
        optimizer=args.optimizer,
        model_save_path=args.model_save_path,
        save_model=not args.no_save,
        resume_from=args.resume_from,
        max_examples=args.max_examples,
        seed=args.seed,
        log_scale=args.log_scale,
        load_in_memory=args.load_in_memory,
        no_dataparallel=args.no_dataparallel
    )
    
    # Print configuration
    print("🔧 Autoencoder Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_autoencoder(config)


if __name__ == "__main__":
    main()
