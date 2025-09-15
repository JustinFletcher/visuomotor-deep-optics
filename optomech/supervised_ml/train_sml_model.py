#!/usr/bin/env python3
"""
Training script for Supervised ML (SML) model on Optomech dataset.
Loads observation-action pairs, splits into train/val/test, and trains a CNN model.
"""

import os
import sys
import json
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
import uuid
import time
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

# Optional HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("⚠️  h5py not available, HDF5 files cannot be loaded")

# Add parent directory for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    dataset_path: str = "datasets/sml_100k_dataset"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    device: str = "auto"  # auto, cuda, mps, cpu
    save_model: bool = True
    model_save_path: str = "saved_models/sml_model.pth"
    plot_losses: bool = True
    seed: int = 42
    resume_from: str = None  # Path to checkpoint to resume from
    start_epoch: int = 0  # Starting epoch (for resumed training)


class OptomechDataset(Dataset):
    """PyTorch Dataset for Optomech observation-action pairs"""
    
    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]], transform=None):
        """
        Args:
            pairs: List of (observation, perfect_action) tuples
            transform: Optional transform to apply to observations
        """
        self.pairs = pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        observation, perfect_action = self.pairs[idx]
        
        # Convert to torch tensors
        # Observation: uint16 -> float32, normalize to [0, 1]
        obs_tensor = torch.from_numpy(observation).float() / 65535.0
        action_tensor = torch.from_numpy(perfect_action).float()
        
        if self.transform:
            obs_tensor = self.transform(obs_tensor)
            
        return obs_tensor, action_tensor


class SMLModel(nn.Module):
    """CNN model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15):
        """
        Args:
            input_channels: Number of observation channels (2 for real/imag)
            action_dim: Dimension of action space (15 for optomech segments)
        """
        super(SMLModel, self).__init__()
        
        # CNN feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class SMLResNet(nn.Module):
    """ResNet-like model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15):
        super(SMLResNet, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()
        
    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = []
        if stride != 1 or in_planes != planes:
            layers.append(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(planes))
        
        for _ in range(blocks):
            layers.extend([
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes),
            ])
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class SMLSimple(nn.Module):
    """Simple lightweight model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15):
        super(SMLSimple, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SMLHRNet(nn.Module):
    """HRNet-like model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15):
        super(SMLHRNet, self).__init__()
        
        # Stem network
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # High-resolution branches
        self.stage1 = self._make_stage(64, [32], [1])
        self.stage2 = self._make_stage(32, [32, 64], [1, 2])
        self.stage3 = self._make_stage([32, 64], [32, 64, 128], [1, 2, 4])
        
        # Fusion layers for combining multi-resolution features
        self.fusion = nn.ModuleList([
            nn.Conv2d(32, 32, 1),
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(128, 32, 1)
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(32, 256, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, action_dim)
        
    def _make_stage(self, in_channels, out_channels_list, stride_list):
        """Create a stage with multiple resolution branches"""
        if isinstance(in_channels, int):
            in_channels = [in_channels]
            
        branches = nn.ModuleList()
        for i, (out_ch, stride) in enumerate(zip(out_channels_list, stride_list)):
            if i < len(in_channels):
                in_ch = in_channels[i]
            else:
                in_ch = in_channels[0]  # Use first channel for new branches
                
            branch = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            branches.append(branch)
        return branches
        
    def _fuse_layers(self, x_list, fusion_layers):
        """Fuse features from different resolution branches"""
        if len(x_list) == 1:
            return x_list[0]
            
        # Upsample all features to highest resolution and fuse
        target_size = x_list[0].shape[2:]
        fused = None
        
        for i, (x, fusion) in enumerate(zip(x_list, fusion_layers)):
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = fusion(x)
            
            if fused is None:
                fused = x
            else:
                fused = fused + x
                
        return fused
        
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x1 = self.stage1[0](x)
        
        # Stage 2
        x1_s2 = self.stage2[0](x1)
        x2_s2 = self.stage2[1](x1)
        
        # Stage 3
        x1_s3 = self.stage3[0](x1_s2)
        x2_s3 = self.stage3[1](x2_s2)
        x3_s3 = self.stage3[2](x2_s2)
        
        # Fuse multi-resolution features
        x_fused = self._fuse_layers([x1_s3, x2_s3, x3_s3], self.fusion)
        
        # Final prediction
        x = self.final_conv(x_fused)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class SMLVanillaConv(nn.Module):
    """Vanilla convolutional model with minimal parameters for rapid training"""
    
    def __init__(self, input_channels=2, action_dim=15, channel_scale=16, mlp_scale=128):
        super(SMLVanillaConv, self).__init__()
        
        # Five small conv blocks with configurable channel scaling, no downsampling
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, channel_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 2
            nn.Conv2d(channel_scale, channel_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 3
            nn.Conv2d(channel_scale, channel_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 4
            nn.Conv2d(channel_scale, channel_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv block 5
            nn.Conv2d(channel_scale, channel_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Global average pooling to reduce spatial dimensions
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Configurable MLP for output
        self.classifier = nn.Sequential(
            nn.Linear(channel_scale * 8 * 8, mlp_scale),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mlp_scale, action_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_model(arch: str, input_channels: int, action_dim: int, channel_scale: int = 16, mlp_scale: int = 128) -> nn.Module:
    """Factory function to create different model architectures"""
    if arch == "sml_cnn":
        return SMLModel(input_channels=input_channels, action_dim=action_dim)
    elif arch == "sml_resnet":
        return SMLResNet(input_channels=input_channels, action_dim=action_dim)
    elif arch == "sml_simple":
        return SMLSimple(input_channels=input_channels, action_dim=action_dim)
    elif arch == "sml_hrnet":
        return SMLHRNet(input_channels=input_channels, action_dim=action_dim)
    elif arch == "sml_vanilla":
        return SMLVanillaConv(input_channels=input_channels, action_dim=action_dim, 
                             channel_scale=channel_scale, mlp_scale=mlp_scale)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def load_single_episode(episode_file: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a single episode file and return pairs
    
    Args:
        episode_file: Path to episode JSON file
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    pairs = []
    
    try:
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        # Extract pairs from metadata (new format)
        if 'sample_pairs' in data['metadata']:
            episode_pairs = data['metadata']['sample_pairs']
            for pair in episode_pairs:
                obs = np.array(pair['observation'], dtype=np.uint16)
                action = np.array(pair['perfect_action'], dtype=np.float32)
                
                # Only include pairs with non-empty actions
                if action.size > 0:
                    pairs.append((obs, action))
        
        # Fallback to legacy format if needed
        elif 'observations' in data and 'perfect_actions' in data:
            observations = data['observations']
            perfect_actions = data['perfect_actions']
            for obs, action in zip(observations, perfect_actions):
                obs = np.array(obs, dtype=np.uint16)
                action = np.array(action, dtype=np.float32)
                
                # Only include pairs with non-empty actions
                if action.size > 0:
                    pairs.append((obs, action))
                    
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Failed to load {episode_file}: {e}")
        return []
    
    return pairs


def load_dataset_pairs_sequential(dataset_path: str, use_cache: bool = True, max_examples: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load dataset with sequential processing, supporting HDF5, NPZ, and JSON formats
    
    Args:
        dataset_path: Path to dataset directory
        use_cache: Whether to use cached data if available
        max_examples: Maximum number of examples to load (stops early if specified)
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    dataset_path = Path(dataset_path)
    
    # Skip cache when limiting examples
    cache_file = dataset_path / "processed_pairs_cache.pkl"
    if use_cache and max_examples is None and cache_file.exists():
        print("📦 Loading cached data...")
        try:
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                pairs = pickle.load(f)
            load_time = time.time() - start_time
            print(f"✅ Loaded {len(pairs)} cached pairs in {load_time:.1f}s")
            return pairs
        except Exception as e:
            print(f"Warning: Cache loading failed: {e}, loading fresh...")

    print(f"📂 Loading dataset from {dataset_path}")
    pairs = []
    start_time = time.time()
    
    # Find all dataset files (prioritize HDF5, then NPZ, then JSON)
    h5_files = list(dataset_path.glob("*.h5"))
    npz_files = list(dataset_path.glob("*.npz"))
    json_files = list(dataset_path.glob("episode_*.json")) + list(dataset_path.glob("batch_*.json"))
    
    print(f"Found {len(h5_files)} H5, {len(npz_files)} NPZ, {len(json_files)} JSON files")
    
    # Load HDF5 files (preferred format)
    if h5_files and HDF5_AVAILABLE:
        print("Loading from HDF5 files...")
        for h5_file in sorted(h5_files):
            if max_examples and len(pairs) >= max_examples:
                break
                
            try:
                with h5py.File(h5_file, 'r') as f:
                    observations = f['observations'][:]
                    perfect_actions = f['perfect_actions'][:]
                    
                    # Add pairs
                    for obs, action in zip(observations, perfect_actions):
                        if max_examples and len(pairs) >= max_examples:
                            break
                        pairs.append((obs, action))
                    
                print(f"  Loaded {len(observations)} pairs from {h5_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {h5_file.name}: {e}")
    
    # Load NPZ files (fallback format)
    elif npz_files:
        print("Loading from NPZ files...")
        for npz_file in sorted(npz_files):
            if max_examples and len(pairs) >= max_examples:
                break
                
            try:
                data = np.load(npz_file)
                observations = data['observations']
                perfect_actions = data['perfect_actions']
                
                # Add pairs
                for obs, action in zip(observations, perfect_actions):
                    if max_examples and len(pairs) >= max_examples:
                        break
                    pairs.append((obs, action))
                
                print(f"  Loaded {len(observations)} pairs from {npz_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {npz_file.name}: {e}")
    
    # Load JSON files (legacy format)
    elif json_files:
        print("Loading from JSON files (legacy format)...")
        for episode_file in sorted(json_files):
            if max_examples and len(pairs) >= max_examples:
                break
                
            episode_pairs = load_single_episode(episode_file)
            
            # Add pairs, respecting max_examples limit
            for pair in episode_pairs:
                if max_examples and len(pairs) >= max_examples:
                    break
                pairs.append(pair)
            
            if len(episode_pairs) > 0:
                print(f"  Loaded {len(episode_pairs)} pairs from {episode_file.name}")
    else:
        print("❌ No dataset files found!")
        return []
    
    load_time = time.time() - start_time
    
    if len(pairs) > 0:
        print(f"✅ Total pairs loaded: {len(pairs)}")
        print(f"⏱️  Loading time: {load_time:.1f}s")
        print(f"📊 Speed: {len(pairs)/load_time:.1f} pairs/second")
    
    # Cache results only if we loaded the full dataset
    if use_cache and len(pairs) > 0 and max_examples is None:
        try:
            print("💾 Caching processed data...")
            cache_start = time.time()
            with open(cache_file, 'wb') as f:
                pickle.dump(pairs, f)
            cache_time = time.time() - cache_start
            print(f"✅ Cached {len(pairs)} pairs in {cache_time:.1f}s")
        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")
    
    return pairs


def load_dataset_pairs(dataset_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all observation-action pairs from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    # Use the sequential loading (simple and reliable)
    return load_dataset_pairs_sequential(dataset_path, use_cache=True)


def split_dataset(pairs: List[Tuple[np.ndarray, np.ndarray]], 
                 train_split: float, val_split: float, test_split: float,
                 seed: int = 42) -> Tuple[List, List, List]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        pairs: List of (observation, action) pairs
        train_split: Fraction for training
        val_split: Fraction for validation  
        test_split: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle pairs
    pairs_shuffled = pairs.copy()
    random.shuffle(pairs_shuffled)
    
    # Calculate split indices
    total_samples = len(pairs_shuffled)
    train_end = int(total_samples * train_split)
    val_end = train_end + int(total_samples * val_split)
    
    # Split the data
    train_pairs = pairs_shuffled[:train_end]
    val_pairs = pairs_shuffled[train_end:val_end]
    test_pairs = pairs_shuffled[val_end:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_pairs)} samples ({len(train_pairs)/total_samples:.1%})")
    print(f"  Val:   {len(val_pairs)} samples ({len(val_pairs)/total_samples:.1%})")
    print(f"  Test:  {len(test_pairs)} samples ({len(test_pairs)/total_samples:.1%})")
    
    return train_pairs, val_pairs, test_pairs


def get_device(device_str: str = "auto") -> tuple[torch.device, int]:
    """Get the best available device for training and return device + GPU count"""
    if device_str == "auto":
        if torch.cuda.is_available():
            # Clear CUDA cache and reset any existing contexts
            torch.cuda.empty_cache()
            
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            
            # Check GPU memory availability
            try:
                memory_free, memory_total = torch.cuda.mem_get_info()
                print(f"GPU Memory: {memory_free / 1e9:.1f}GB free / {memory_total / 1e9:.1f}GB total")
                
                if memory_free < 2e9:  # Less than 2GB free
                    print("⚠️  Warning: Low GPU memory available, consider reducing batch size")
            except:
                print("Could not check GPU memory")
            
            if gpu_count > 1:
                print(f"Found {gpu_count} CUDA GPUs - DataParallel will be enabled")
                for i in range(gpu_count):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return device, gpu_count
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
            return device, 1
        else:
            device = torch.device("cpu")
            print("Using CPU device")
            return device, 1
    else:
        device = torch.device(device_str)
        print(f"Using specified device: {device}")
        
        # Clear CUDA cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
            try:
                memory_free, memory_total = torch.cuda.mem_get_info(device)
                print(f"GPU Memory: {memory_free / 1e9:.1f}GB free / {memory_total / 1e9:.1f}GB total")
            except:
                print("Could not check GPU memory")
        
        gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
        return device, gpu_count


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch and return average loss"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for observations, actions in tqdm(dataloader, desc="Training", leave=False):
        observations = observations.to(device)
        actions = actions.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(observations)
        loss = criterion(predictions, actions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss without .item() to avoid GPU/CPU sync
        total_loss += loss.detach()
        num_batches += 1
    
    # Only convert to Python float at the end
    return (total_loss / num_batches).item()


def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                  criterion: nn.Module, device: torch.device) -> float:
    """Validate for one epoch and return average loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for observations, actions in tqdm(dataloader, desc="Validating", leave=False):
            observations = observations.to(device)
            actions = actions.to(device)
            
            # Forward pass
            predictions = model(observations)
            loss = criterion(predictions, actions)
            
            # Accumulate loss without .item() to avoid GPU/CPU sync
            total_loss += loss.detach()
            num_batches += 1
    
    # Only convert to Python float at the end
    return (total_loss / num_batches).item()


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        save_path: str = None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set log scale if losses are very different
    if max(train_losses) / min(train_losses) > 10:
        plt.yscale('log')
        plt.title('Training and Validation Loss (Log Scale)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """
    Load checkpoint and return loaded state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (start_epoch, train_losses, val_losses, best_val_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract training info
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"  Resuming from epoch {start_epoch}")
    print(f"  Previous train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Previous val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Training history: {len(train_losses)} epochs")
    
    return start_epoch, train_losses, val_losses, best_val_loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train SML model on Optomech dataset")
    parser.add_argument("--dataset_path", type=str, default="optomech/supervised_ml/datasets/sml_100k_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/mps/cpu/cuda:0/cuda:1)")
    parser.add_argument("--no_dataparallel", action="store_true", 
                       help="Disable DataParallel even with multiple GPUs")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU training (useful for debugging GPU issues)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--log_dir", type=str, default="runs",
                       help="Base directory for TensorBoard logs, plots, and saved models")
    parser.add_argument("--model_arch", type=str, default="sml_cnn", 
                       choices=["sml_cnn", "sml_resnet", "sml_simple", "sml_hrnet", "sml_vanilla"],
                       help="Model architecture to use")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to load (disables cache)")
    parser.add_argument("--channel_scale", type=int, default=16,
                       help="Number of channels for VanillaConv architecture")
    parser.add_argument("--mlp_scale", type=int, default=128,
                       help="Hidden layer size for VanillaConv MLP")
    
    args = parser.parse_args()
    
    # Initialize CUDA environment safely
    if torch.cuda.is_available():
        # Set CUDA device and initialize context early
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # Set memory allocation settings to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Disable cudnn benchmark for more stable behavior
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print(f"🔧 CUDA initialized with {torch.cuda.device_count()} devices")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set up log directory for this run
    run_id = str(uuid.uuid4())[:8]
    log_dir = Path(args.log_dir) / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Logging to: {log_dir}")

    # Update config paths to use log_dir
    model_save_path = str(log_dir / "sml_model.pth")
    config = TrainingConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume_from,
        model_save_path=model_save_path,
        plot_losses=True
    )
    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=str(log_dir))
    
    # Log hyperparameters/flags to TensorBoard as a markdown table
    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    print("🚀 Starting SML Model Training")
    print("=" * 50)
    
    # Load and split dataset
    print("� Loading dataset...")
    # Disable cache when max_examples is specified
    use_cache = args.max_examples is None
    if args.max_examples is not None:
        print(f"  Max examples: {args.max_examples} (cache disabled)")
    pairs = load_dataset_pairs_sequential(config.dataset_path, use_cache=use_cache, max_examples=args.max_examples)
    
    if len(pairs) == 0:
        print("❌ No valid observation-action pairs found!")
        return
    
    # Check action dimensions
    sample_action = pairs[0][1]
    action_dim = len(sample_action) if sample_action.size > 0 else 15
    print(f"Action dimension: {action_dim}")
    
    # Check observation shape
    sample_obs = pairs[0][0]
    print(f"Observation shape: {sample_obs.shape}")
    
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, config.train_split, config.val_split, config.test_split, config.seed
    )
    
    # Create datasets and dataloaders
    train_dataset = OptomechDataset(train_pairs)
    val_dataset = OptomechDataset(val_pairs)
    test_dataset = OptomechDataset(test_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
    #                        shuffle=False, num_workers=0)
    
    # DEBUG: Test on the same distribution to isolate issues
    val_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Initialize model, optimizer, and criterion
    device_arg = "cpu" if args.force_cpu else config.device
    device, gpu_count = get_device(device_arg)
    
    input_channels = sample_obs.shape[0] if len(sample_obs.shape) == 3 else 1
    model = create_model(args.model_arch, input_channels=input_channels, action_dim=action_dim, 
                        channel_scale=args.channel_scale, mlp_scale=args.mlp_scale).to(device)
    
    # Enable DataParallel for multi-GPU training
    if gpu_count > 1 and device.type == "cuda" and not args.no_dataparallel:
        model = DataParallel(model)
        print(f"🚀 DataParallel enabled across {gpu_count} GPUs")
        
        # Adjust effective batch size for multi-GPU
        effective_batch_size = config.batch_size * gpu_count
        print(f"  Effective batch size: {config.batch_size} × {gpu_count} = {effective_batch_size}")
        
        # Report GPU memory
        for i in range(gpu_count):
            if torch.cuda.is_available():
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i} memory: {memory:.1f} GB")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    print(f"\nModel architecture:")
    print(f"  Architecture: {args.model_arch}")
    print(f"  Input channels: {input_channels}")
    print(f"  Action dimension: {action_dim}")
    if args.model_arch == "sml_vanilla":
        print(f"  Channel scale: {args.channel_scale}")
        print(f"  MLP scale: {args.mlp_scale}")
    
    # Count parameters (handle DataParallel wrapper)
    param_model = model.module if isinstance(model, DataParallel) else model
    total_params = sum(p.numel() for p in param_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    if gpu_count > 1 and device.type == "cuda":
        print(f"  Parameters per GPU: ~{total_params // gpu_count:,}")
    
    # Log model summary to TensorBoard
    try:
        # Create a sample input for torchinfo summary
        sample_obs_shape = sample_obs.shape
        if len(sample_obs_shape) == 3:
            sample_input_shape = (1, sample_obs_shape[0], sample_obs_shape[1], sample_obs_shape[2])
        else:
            sample_input_shape = (1, 1, sample_obs_shape[0], sample_obs_shape[1])
        
        model_summary = summary(param_model, input_size=sample_input_shape, verbose=0)
        
        # Convert summary to string and log to TensorBoard
        summary_str = str(model_summary)
        tb_writer.add_text("model_summary", f"```\n{summary_str}\n```")
        print(f"  Model summary logged to TensorBoard")
    except Exception as e:
        print(f"  Warning: Could not generate model summary: {e}")
    
    # Training loop
    print(f"\n🏋️ Training for {config.num_epochs} epochs...")
    print("=" * 50)
    
    # Initialize training state
    train_losses = []
    val_losses = []
    epoch_times = []
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Load checkpoint if resuming
    if config.resume_from:
        start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            config.resume_from, model, optimizer, device
        )
        config.start_epoch = start_epoch
        print(f"  Continuing training from epoch {start_epoch}")
    else:
        print("  Starting fresh training")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # TensorBoard logging
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Val', val_loss, epoch)

        # Calculate timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Calculate ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = config.num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60

        if eta_hours >= 1:
            eta_str = f"{eta_hours:.1f}h"
        elif eta_minutes >= 1:
            eta_str = f"{eta_minutes:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        print(f"  ETA:        {eta_str}")

        # Show GPU memory usage for multi-GPU setups
        if gpu_count > 1 and device.type == "cuda":
            memory_info = []
            for i in range(gpu_count):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                memory_info.append(f"GPU{i}: {allocated:.1f}/{cached:.1f}GB")
            print(f"  GPU Memory: {' | '.join(memory_info)}")
        elif device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {allocated:.1f}/{cached:.1f}GB")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'config': config
                }, config.model_save_path)
                print(f"  ✅ New best model saved at {config.model_save_path}!")
    
    total_training_time = time.time() - training_start_time
    
    # Test final model
    print(f"\n🧪 Testing final model...")
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Check if training loss decreased
    if len(train_losses) > 0:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n📊 Training Results:")
        print(f"  Initial train loss: {initial_loss:.6f}")
        print(f"  Final train loss:   {final_loss:.6f}")
        print(f"  Improvement:        {improvement:.1f}%")
        print(f"  Best val loss:      {best_val_loss:.6f}")
        print(f"  Final test loss:    {test_loss:.6f}")
        print(f"  Total training time: {total_training_time/60:.1f} minutes")
        print(f"  Avg epoch time:      {np.mean(epoch_times):.1f} seconds")
        
        if improvement > 5:
            print("✅ Training loss decreased significantly - model is learning!")
        elif improvement > 0:
            print("⚠️  Training loss decreased slightly - consider longer training")
        else:
            print("❌ Training loss did not decrease - check model/data/hyperparameters")
    else:
        print(f"\n📊 Training Results:")
        print(f"  Best val loss:      {best_val_loss:.6f}")
        print(f"  Final test loss:    {test_loss:.6f}")
        print("  No new training completed (resumed from final epoch)")
    
    # Plot training curves
    if config.plot_losses:
        plot_save_path = str(log_dir / "training_curves.png")
        plot_training_curves(train_losses, val_losses, plot_save_path)
        print(f"📈 Training curves saved to {plot_save_path}")

    tb_writer.close()
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
