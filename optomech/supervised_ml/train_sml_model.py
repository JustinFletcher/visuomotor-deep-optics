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
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import pickle

import matplotlib.pyplot as plt
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


def center_crop_transform(tensor, crop_size):
    """
    Center crop a tensor to crop_size × crop_size pixels.
    
    Args:
        tensor: Input tensor of shape [channels, height, width] or [batch, channels, height, width]
        crop_size: Size of the center crop (crop_size × crop_size)
        
    Returns:
        Center-cropped tensor of shape [channels, crop_size, crop_size] or [batch, channels, crop_size, crop_size]
    """
    if len(tensor.shape) == 3:
        # Single image: [C, H, W]
        c, h, w = tensor.shape
        
        if crop_size > min(h, w):
            raise ValueError(f"Crop size {crop_size} is larger than image dimensions {h}×{w}")
        
        # Calculate center crop coordinates
        center_h, center_w = h // 2, w // 2
        half_crop = crop_size // 2
        
        start_h = center_h - half_crop
        end_h = start_h + crop_size
        start_w = center_w - half_crop
        end_w = start_w + crop_size
        
        # Perform center crop
        return tensor[:, start_h:end_h, start_w:end_w]
        
    elif len(tensor.shape) == 4:
        # Batch of images: [N, C, H, W]
        n, c, h, w = tensor.shape
        
        if crop_size > min(h, w):
            raise ValueError(f"Crop size {crop_size} is larger than image dimensions {h}×{w}")
        
        # Calculate center crop coordinates
        center_h, center_w = h // 2, w // 2
        half_crop = crop_size // 2
        
        start_h = center_h - half_crop
        end_h = start_h + crop_size
        start_w = center_w - half_crop
        end_w = start_w + crop_size
        
        # Perform center crop
        return tensor[:, :, start_h:end_h, start_w:end_w]
    
    else:
        raise ValueError(f"Expected 3D tensor [C, H, W] or 4D tensor [N, C, H, W], got shape {tensor.shape}")


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
    
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        """
        Args:
            input_channels: Number of observation channels (2 for real/imag)
            action_dim: Dimension of action space (15 for optomech segments)
            input_crop_size: If specified, center crop input to this size (e.g., 128)
        """
        super(SMLModel, self).__init__()
        self.input_crop_size = input_crop_size
        
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
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class SMLResNetGN(nn.Module):
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        super().__init__()
        self.input_crop_size = input_crop_size
        # Input group default
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)  # Use 8 groups for 64 channels
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Input group default
        # self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)  # Use 8 groups for 64 channels
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # ResNet18 layer configuration: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=1)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)  # Added missing layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, action_dim)  # Changed from 256 to 512
        self.tanh = nn.Tanh()

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = []
        
        # First block (may have downsample)
        layers.append(BasicBlockGN(in_planes, planes, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(planes, planes, stride=1))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Added layer4 forward pass
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class BasicBlockGN(nn.Module):
    """Basic ResNet block with GroupNorm"""
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Add residual connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class SMLResNet(nn.Module):
    """ResNet-like model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        super(SMLResNet, self).__init__()
        self.input_crop_size = input_crop_size
        
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
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
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
    
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        super(SMLSimple, self).__init__()
        self.input_crop_size = input_crop_size
        
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
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SMLHRNet(nn.Module):
    """HRNet-like model for predicting perfect actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        super(SMLHRNet, self).__init__()
        self.input_crop_size = input_crop_size
        
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
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
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
    
    def __init__(self, input_channels=2, action_dim=15, channel_scale=16, mlp_scale=128, input_crop_size=None):
        super(SMLVanillaConv, self).__init__()
        self.input_crop_size = input_crop_size
        
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
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_model(arch: str, input_channels: int, action_dim: int, channel_scale: int = 16, mlp_scale: int = 128, input_crop_size: int = None) -> nn.Module:
    """Factory function to create different model architectures"""
    if arch == "sml_cnn":
        return SMLModel(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "sml_resnet":
        return SMLResNet(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "sml_resnet_gn":
        return SMLResNetGN(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "sml_simple":
        return SMLSimple(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "sml_hrnet":
        return SMLHRNet(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "sml_vanilla":
        return SMLVanillaConv(input_channels=input_channels, action_dim=action_dim, 
                             channel_scale=channel_scale, mlp_scale=mlp_scale, input_crop_size=input_crop_size)
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

    use_dp = isinstance(model, torch.nn.DataParallel)

    for observations, actions in tqdm(dataloader, desc="Training", leave=False):
        # With DataParallel: keep inputs on CPU; DP will scatter/move them.
        # Without DP (single GPU / DDP rank local): move to device yourself.
        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(observations)


        loss = criterion(predictions, actions)

        # Backward + step
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.detach()
        num_batches += 1

    # Only convert to Python float at the end
    return (total_loss / num_batches).item()


def train_epoch_with_metrics(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                            criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float], np.ndarray, Dict[str, float]]:
    """Train for one epoch and return loss, MAE metrics, raw error distribution, and absolute error metrics"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_mae_errors = []
    all_raw_errors = []
    all_abs_errors = []  # All absolute errors across all actions (flattened)

    use_dp = isinstance(model, torch.nn.DataParallel)

    for observations, actions in tqdm(dataloader, desc="Training", leave=False):
        # With DataParallel: keep inputs on CPU; DP will scatter/move them.
        # Without DP (single GPU / DDP rank local): move to device yourself.
        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(observations)
        loss = criterion(predictions, actions)

        # Calculate raw errors for histogram (predictions - actions)
        raw_errors = predictions - actions  # [batch_size, action_dim]
        all_raw_errors.extend(raw_errors.detach().cpu().numpy().flatten())
        
        # Calculate absolute errors and store all individual action errors
        abs_errors = torch.abs(raw_errors)  # [batch_size, action_dim]
        all_abs_errors.extend(abs_errors.detach().cpu().numpy().flatten())
        
        # Calculate MAE for each sample in the batch for statistics (example-level)
        mae_per_sample = torch.mean(abs_errors, dim=1)  # [batch_size]
        all_mae_errors.extend(mae_per_sample.detach().cpu().numpy())

        # Backward + step
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.detach()
        num_batches += 1

    # Convert to numpy for statistics
    mae_errors_array = np.array(all_mae_errors)
    raw_errors_array = np.array(all_raw_errors)
    abs_errors_array = np.array(all_abs_errors)
    
    # Calculate MAE statistics (example-level aggregation)
    mae_metrics = {
        'mae_mean': np.mean(mae_errors_array),
        'mae_median': np.median(mae_errors_array),
        'mae_min': np.min(mae_errors_array),
        'mae_max': np.max(mae_errors_array),
        'mae_std': np.std(mae_errors_array),
        'mae_q25': np.percentile(mae_errors_array, 25),
        'mae_q75': np.percentile(mae_errors_array, 75)
    }
    
    # Calculate absolute error statistics (across all individual action values)
    abs_error_metrics = {
        'abs_error_mean': np.mean(abs_errors_array),
        'abs_error_median': np.median(abs_errors_array),
        'abs_error_min': np.min(abs_errors_array),
        'abs_error_max': np.max(abs_errors_array),
        'abs_error_std': np.std(abs_errors_array),
        'abs_error_q25': np.percentile(abs_errors_array, 25),
        'abs_error_q75': np.percentile(abs_errors_array, 75)
    }

    # Convert loss to Python float at the end
    avg_loss = (total_loss / num_batches).item()
    
    return avg_loss, mae_metrics, raw_errors_array, abs_error_metrics


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


def validate_epoch_with_metrics(model: nn.Module, dataloader: DataLoader, 
                               criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float], np.ndarray, Dict[str, float]]:
    """Validate for one epoch and return loss, MAE metrics, raw error distribution, and absolute error metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_mae_errors = []
    all_raw_errors = []
    all_abs_errors = []  # All absolute errors across all actions (flattened)
    
    with torch.no_grad():
        for observations, actions in tqdm(dataloader, desc="Validating", leave=False):
            observations = observations.to(device)
            actions = actions.to(device)
            
            # Forward pass
            predictions = model(observations)
            loss = criterion(predictions, actions)
            
            # Calculate raw errors for histogram (predictions - actions)
            raw_errors = predictions - actions  # [batch_size, action_dim]
            all_raw_errors.extend(raw_errors.cpu().numpy().flatten())
            
            # Calculate absolute errors and store all individual action errors
            abs_errors = torch.abs(raw_errors)  # [batch_size, action_dim]
            all_abs_errors.extend(abs_errors.cpu().numpy().flatten())
            
            # Calculate MAE for each sample in the batch for statistics (example-level)
            mae_per_sample = torch.mean(abs_errors, dim=1)  # [batch_size]
            all_mae_errors.extend(mae_per_sample.cpu().numpy())
            
            # Accumulate loss without .item() to avoid GPU/CPU sync
            total_loss += loss.detach()
            num_batches += 1
    
    # Convert to numpy for statistics
    mae_errors_array = np.array(all_mae_errors)
    raw_errors_array = np.array(all_raw_errors)
    abs_errors_array = np.array(all_abs_errors)
    
    # Calculate MAE statistics (example-level aggregation)
    mae_metrics = {
        'mae_mean': np.mean(mae_errors_array),
        'mae_median': np.median(mae_errors_array),
        'mae_min': np.min(mae_errors_array),
        'mae_max': np.max(mae_errors_array),
        'mae_std': np.std(mae_errors_array),
        'mae_q25': np.percentile(mae_errors_array, 25),
        'mae_q75': np.percentile(mae_errors_array, 75)
    }
    
    # Calculate absolute error statistics (across all individual action values)
    abs_error_metrics = {
        'abs_error_mean': np.mean(abs_errors_array),
        'abs_error_median': np.median(abs_errors_array),
        'abs_error_min': np.min(abs_errors_array),
        'abs_error_max': np.max(abs_errors_array),
        'abs_error_std': np.std(abs_errors_array),
        'abs_error_q25': np.percentile(abs_errors_array, 25),
        'abs_error_q75': np.percentile(abs_errors_array, 75)
    }
    
    # Convert loss to Python float at the end
    avg_loss = (total_loss / num_batches).item()
    
    return avg_loss, mae_metrics, raw_errors_array, abs_error_metrics


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


def find_most_recent_model(runs_dir: str = "runs") -> Tuple[str, str]:
    """
    Find the most recently saved model in the runs directory.
    
    Args:
        runs_dir: Base directory containing training runs
        
    Returns:
        Tuple of (model_path, run_directory)
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Find all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    
    # Sort by modification time to get most recent
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Look for model files in the most recent runs
    for run_dir in run_dirs:
        model_path = run_dir / "sml_model.pth"
        if model_path.exists():
            print(f"📁 Found most recent model: {model_path}")
            return str(model_path), str(run_dir)
    
    raise FileNotFoundError(f"No model files found in any run directory")


def load_dataset_config(dataset_path: str) -> Dict:
    """
    Load environment configuration from dataset job config.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary containing environment configuration
    """
    dataset_path = Path(dataset_path)
    
    # Look for job config file
    possible_configs = [
        dataset_path / f"{dataset_path.name}_job_config.json",
        dataset_path / "job_config.json",
        dataset_path / "dataset_job_config.json"
    ]
    
    for config_path in possible_configs:
        if config_path.exists():
            print(f"📋 Loading dataset config: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
    
    raise FileNotFoundError(f"No job config found for dataset: {dataset_path}")


def perform_rollout_instrumentation(
    model_path: str = None,
    num_seeds: int = 32,
    rollout_steps: int = 250,
    runs_dir: str = "runs",
    save_results: bool = True,
    output_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform rollout instrumentation using the most recent model.
    
    Args:
        model_path: Path to model (if None, finds most recent)
        num_seeds: Number of random seeds to evaluate
        rollout_steps: Number of steps per rollout
        runs_dir: Directory containing training runs
        save_results: Whether to save results to disk
        output_dir: Directory to save results (if None, uses model's run dir)
        
    Returns:
        Tuple of (mean_rewards, std_rewards, metadata)
    """
    print("\n🎯 Starting Rollout Instrumentation")
    print("=" * 50)
    
    # Import rollout functionality directly
    try:
        # Add the parent directory to the path for imports
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        sys.path.insert(0, str(project_root))
        
        from optomech.optomech_rollout import UniversalRolloutEngine, create_model_interface
        from argparse import Namespace
        import gymnasium as gym
    except ImportError as e:
        print(f"❌ Failed to import rollout dependencies: {e}")
        # Return empty results if imports fail
        return np.array([]), np.array([]), {
            'model_path': model_path or "unknown",
            'successful_seeds': [],
            'num_successful_rollouts': 0,
            'rollout_steps': rollout_steps,
            'final_mean_reward': 0.0,
            'final_std_reward': 0.0,
            'total_mean_return': 0.0,
            'total_std_return': 0.0,
            'output_dir': str(output_dir or "unknown")
        }
    
    # Find model if not provided
    if model_path is None:
        model_path, run_dir = find_most_recent_model(runs_dir)
        if output_dir is None:
            output_dir = run_dir
    else:
        run_dir = str(Path(model_path).parent)
        if output_dir is None:
            output_dir = run_dir
    
    # Load environment configuration from sml_job_config.json
    sml_config_path = Path("optomech/supervised_ml/sml_job_config.json")
    if sml_config_path.exists():
        print(f"📋 Loading environment config from: {sml_config_path}")
        with open(sml_config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract environment configuration
        dataset_config = {
            "env_id": config_data.get("env_id", "optomech-v1"),
            "object_type": config_data.get("object_type", "single"),
            "aperture_type": config_data.get("aperture_type", "elf"),
            "reward_function": config_data.get("reward_function", "align"),
            "observation_mode": config_data.get("observation_mode", "image_only"),
            "focal_plane_image_size_pixels": config_data.get("focal_plane_image_size_pixels", 256),
            "environment_flags": config_data.get("environment_flags", [])
        }
        print(f"🔍 Environment flags from config: {dataset_config['environment_flags']}")
    else:
        print("⚠️  sml_job_config.json not found, using default environment settings")
        dataset_config = {
            "env_id": "optomech-v1",
            "object_type": "single", 
            "aperture_type": "elf",
            "reward_function": "align",
            "observation_mode": "image_only",
            "focal_plane_image_size_pixels": 256,
            "environment_flags": []
        }
    
    # Create output directory for results
    output_path = Path(output_dir)
    rollout_results_dir = output_path / "rollout_results"
    rollout_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎲 Running {num_seeds} rollouts with {rollout_steps} steps each")
    print(f"📊 Results will be saved to: {rollout_results_dir}")
    
    # Store rewards for all seeds and steps
    all_episode_rewards = []  # Shape: [num_seeds, rollout_steps]
    successful_seeds = []
    
    # Get the optomech_rollout.py path
    optomech_rollout_path = Path(__file__).parent.parent / "optomech_rollout.py"
    if not optomech_rollout_path.exists():
        raise FileNotFoundError(f"optomech_rollout.py not found at: {optomech_rollout_path}")
    
    # Prepare base environment configuration from dataset config
    env_config_path = rollout_results_dir / "rollout_env_config.json"
    
    # Create environment config based on dataset config
    rollout_env_config = {
        "env_id": dataset_config.get("env_id", "optomech-v1"),
        "object_type": dataset_config.get("object_type", "single"),
        "aperture_type": dataset_config.get("aperture_type", "elf"),
        "reward_function": dataset_config.get("reward_function", "align"),
        "observation_mode": dataset_config.get("observation_mode", "image_only"),
        "focal_plane_image_size_pixels": dataset_config.get("focal_plane_image_size_pixels", 256),
        "environment_flags": dataset_config.get("environment_flags", [])
    }
    
    # Save environment config for rollout script
    with open(env_config_path, 'w') as f:
        json.dump(rollout_env_config, f, indent=2)
    
    print(f"💾 Saved rollout environment config to: {env_config_path}")
    
    # Create environment arguments from config with comprehensive defaults
    env_args = Namespace()
    
    # Set all default values based on the Args class from build_optomech_dataset.py
    # This ensures we have all required attributes with sensible defaults
    default_values = {
        # Core environment settings
        'env_id': 'optomech-v1',
        'total_timesteps': 100_000_000,
        'action_type': 'none',
        'object_type': 'single',
        'aperture_type': 'elf',
        'max_episode_steps': rollout_steps,
        'num_envs': 1,
        'observation_mode': 'image_only',
        'focal_plane_image_size_pixels': 256,
        'observation_window_size': 2,
        'num_tensioners': 0,
        'num_atmosphere_layers': 0,
        'optomech_version': 'test',
        'reward_function': 'strehl',
        'silence': True,  # Keep quiet during rollouts
        
        # Control settings
        'incremental_control': False,
        'command_tensioners': False,
        'command_secondaries': False,
        'command_tip_tilt': False,
        'command_dm': False,
        'discrete_control': False,
        'discrete_control_steps': 128,
        
        # Rendering and logging
        'render': False,
        'render_frequency': 1,
        'render_dpi': 100.0,
        'record_env_state_info': False,
        'write_env_state_info': False,
        'write_state_interval': 1,
        'state_info_save_dir': './tmp/',
        'report_time': False,
        
        # Simulation timing
        'ao_loop_active': False,
        'ao_interval_ms': 1.0,
        'control_interval_ms': 2.0,
        'frame_interval_ms': 4.0,
        'decision_interval_ms': 8.0,
        
        # Optomech modeling toggles
        'init_differential_motion': False,
        'simulate_differential_motion': False,
        'randomize_dm': False,
        'model_wind_diff_motion': False,
        'model_gravity_diff_motion': False,
        'model_temp_diff_motion': False,
        
        # Hardware
        'gpu_list': '0',
        
        # Extended object parameters
        'extended_object_image_file': '.\\resources\\sample_image.png',
        'extended_object_distance': None,
        'extended_object_extent': None,
    }
    
    # Apply all defaults first
    for key, value in default_values.items():
        setattr(env_args, key, value)
    
    # Override with values from config
    env_args.env_id = dataset_config["env_id"]
    env_args.object_type = dataset_config["object_type"]
    env_args.aperture_type = dataset_config["aperture_type"]
    env_args.reward_function = dataset_config["reward_function"]
    env_args.observation_mode = dataset_config["observation_mode"]
    env_args.focal_plane_image_size_pixels = dataset_config["focal_plane_image_size_pixels"]
    
    # Parse environment_flags from config and apply them (these take highest precedence)
    for flag in dataset_config.get("environment_flags", []):
        if "=" in flag:
            key, value = flag.replace("--", "").split("=", 1)
            # Convert value to appropriate type
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                value = float(value)
            print(f"🔧 Setting env_args.{key} = {value} (type: {type(value).__name__})")
            setattr(env_args, key, value)
        else:
            # Boolean flag without value (e.g., --command_secondaries)
            key = flag.replace("--", "")
            print(f"🔧 Setting env_args.{key} = True")
            setattr(env_args, key, True)
    
    # Debug: Print the critical interval values
    print(f"🔍 Final interval values:")
    print(f"  ao_interval_ms: {getattr(env_args, 'ao_interval_ms', 'NOT_SET')}")
    print(f"  control_interval_ms: {getattr(env_args, 'control_interval_ms', 'NOT_SET')}")
    print(f"  frame_interval_ms: {getattr(env_args, 'frame_interval_ms', 'NOT_SET')}")
    print(f"  decision_interval_ms: {getattr(env_args, 'decision_interval_ms', 'NOT_SET')}")
    
    # Register optomech environment (only if not already registered)
    try:
        gym.envs.registration.register(
            id='optomech-v1',
            entry_point='optomech.optomech:OptomechEnv',
            max_episode_steps=rollout_steps,
        )
    except gym.error.Error:
        # Environment already registered, that's fine
        pass
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Create model interface
    print(f"📦 Loading model from: {model_path}")
    try:
        model_interface = create_model_interface(
            model_path=model_path,
            model_type="sml",  # Supervised ML model
            device=device
        )
        print(f"✅ Model loaded successfully: {model_interface.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Return empty results if model loading fails
        return np.array([]), np.array([]), {
            'model_path': model_path,
            'successful_seeds': [],
            'num_successful_rollouts': 0,
            'rollout_steps': rollout_steps,
            'final_mean_reward': 0.0,
            'final_std_reward': 0.0,
            'total_mean_return': 0.0,
            'total_std_return': 0.0,
            'output_dir': str(rollout_results_dir)
        }
    
    # Create rollout engine
    rollout_engine = UniversalRolloutEngine(
        model_interface=model_interface,
        env_args=env_args
    )
    
    # Run rollouts for each seed
    for seed in range(num_seeds):
        print(f"\n🔄 Running rollout {seed + 1}/{num_seeds} (seed={seed})")
        
        try:
            # Set random seed for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Run rollout directly using the engine
            episodic_returns, step_wise_rewards = rollout_engine.run_rollout(
                num_episodes=1,
                exploration_noise=0.0,
                save_path=None,
                save_episodes=False,
                random_policy=False
            )
            
            if len(episodic_returns) > 0 and len(step_wise_rewards) > 0:
                episode_rewards = step_wise_rewards[0]  # Get first (and only) episode
                
                # Pad or truncate to exact rollout_steps length
                if len(episode_rewards) < rollout_steps:
                    # Pad with the last reward value
                    last_reward = episode_rewards[-1] if episode_rewards else 0.0
                    episode_rewards.extend([last_reward] * (rollout_steps - len(episode_rewards)))
                elif len(episode_rewards) > rollout_steps:
                    # Truncate to exact length
                    episode_rewards = episode_rewards[:rollout_steps]
                
                all_episode_rewards.append(episode_rewards)
                successful_seeds.append(seed)
                
                print(f"  ✅ Rollout completed: {len(episode_rewards)} steps, "
                      f"total reward: {sum(episode_rewards):.4f}")
            else:
                print(f"  ⚠️  Rollout failed for seed {seed} - no rewards returned")
                
        except Exception as e:
            print(f"  ❌ Error running rollout for seed {seed}: {e}")
            continue
    
    if not all_episode_rewards:
        raise RuntimeError("No successful rollouts completed")
    
    # Convert to numpy array for analysis
    all_episode_rewards = np.array(all_episode_rewards)  # Shape: [num_successful, rollout_steps]
    
    print(f"\n📈 Analysis of {len(successful_seeds)} successful rollouts:")
    print(f"  Successful seeds: {successful_seeds}")
    
    # Calculate statistics across seeds for each timestep
    mean_rewards = np.mean(all_episode_rewards, axis=0)  # Shape: [rollout_steps]
    std_rewards = np.std(all_episode_rewards, axis=0)    # Shape: [rollout_steps]
    
    # Calculate cumulative rewards for additional analysis
    cumulative_rewards = np.cumsum(all_episode_rewards, axis=1)
    mean_cumulative = np.mean(cumulative_rewards, axis=0)
    std_cumulative = np.std(cumulative_rewards, axis=0)
    
    # Print summary statistics
    final_mean_reward = mean_rewards[-1]
    final_std_reward = std_rewards[-1]
    total_mean_return = mean_cumulative[-1]
    total_std_return = std_cumulative[-1]
    
    print(f"  Final step reward: {final_mean_reward:.4f} ± {final_std_reward:.4f}")
    print(f"  Total episode return: {total_mean_return:.4f} ± {total_std_return:.4f}")
    print(f"  Mean step reward: {np.mean(mean_rewards):.4f}")
    print(f"  Max step reward: {np.max(mean_rewards):.4f}")
    print(f"  Min step reward: {np.min(mean_rewards):.4f}")
    
    # Create comprehensive plots
    if save_results:
        print(f"\n📊 Creating rollout analysis plots...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        timesteps = np.arange(rollout_steps)
        
        # Plot 1: Step-wise rewards with error bars
        ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.fill_between(timesteps, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step-wise Reward Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        ax2.plot(timesteps, mean_cumulative, 'g-', linewidth=2, label='Mean Cumulative Return')
        ax2.fill_between(timesteps,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        alpha=0.3, color='green', label='±1 Std Dev')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative Return')
        ax2.set_title('Cumulative Return Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution heatmap (all seeds)
        im = ax3.imshow(all_episode_rewards, aspect='auto', cmap='viridis', interpolation='nearest')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Seed')
        ax3.set_title('Reward Heatmap Across All Seeds')
        plt.colorbar(im, ax=ax3, label='Reward')
        
        # Plot 4: Final return distribution
        final_returns = cumulative_rewards[:, -1]
        ax4.hist(final_returns, bins=min(20, len(final_returns)), alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(final_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_returns):.2f}')
        ax4.axvline(np.median(final_returns), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(final_returns):.2f}')
        ax4.set_xlabel('Total Episode Return')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Total Episode Returns')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = rollout_results_dir / "rollout_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  📈 Saved analysis plot: {plot_path}")
        
        # Save the raw data
        results_data = {
            'successful_seeds': successful_seeds,
            'all_episode_rewards': all_episode_rewards.tolist(),
            'mean_rewards': mean_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'mean_cumulative': mean_cumulative.tolist(),
            'std_cumulative': std_cumulative.tolist(),
            'metadata': {
                'model_path': model_path,
                'num_seeds': num_seeds,
                'rollout_steps': rollout_steps,
                'successful_rollouts': len(successful_seeds),
                'final_mean_reward': float(final_mean_reward),
                'final_std_reward': float(final_std_reward),
                'total_mean_return': float(total_mean_return),
                'total_std_return': float(total_std_return),
                'mean_step_reward': float(np.mean(mean_rewards)),
                'max_step_reward': float(np.max(mean_rewards)),
                'min_step_reward': float(np.min(mean_rewards))
            }
        }
        
        # Save results as JSON
        results_path = rollout_results_dir / "rollout_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  💾 Saved results data: {results_path}")
        
        # Save results as pickle for easy loading
        pickle_path = rollout_results_dir / "rollout_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"  💾 Saved results pickle: {pickle_path}")
        
        plt.close()
    
    # Prepare metadata for return
    metadata = {
        'model_path': model_path,
        'successful_seeds': successful_seeds,
        'num_successful_rollouts': len(successful_seeds),
        'rollout_steps': rollout_steps,
        'final_mean_reward': float(final_mean_reward),
        'final_std_reward': float(final_std_reward),
        'total_mean_return': float(total_mean_return),
        'total_std_return': float(total_std_return),
        'output_dir': str(rollout_results_dir)
    }
    
    print(f"\n🎉 Rollout instrumentation completed!")
    print(f"📁 Results saved to: {rollout_results_dir}")
    
    return mean_rewards, std_rewards, metadata


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
    parser.add_argument("--seed", type=int, default=random.randint(0, 999999), 
                       help="Random seed (default: random)")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--log_dir", type=str, default="runs",
                       help="Base directory for TensorBoard logs, plots, and saved models")
    parser.add_argument("--model_arch", type=str, default="sml_cnn", 
                       choices=["sml_cnn", "sml_resnet", "sml_resnet_gn", "sml_simple", "sml_hrnet", "sml_vanilla"],
                       help="Model architecture to use")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to load (disables cache)")
    parser.add_argument("--channel_scale", type=int, default=16,
                       help="Number of channels for VanillaConv architecture")
    parser.add_argument("--mlp_scale", type=int, default=128,
                       help="Hidden layer size for VanillaConv MLP")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Optional run name to append to UUID-based directory name")
    parser.add_argument("--center_crop_size", type=int, default=None,
                       help="Center crop images to n×n pixels (default: no cropping). Recommended: 128")
    
    # Rollout instrumentation arguments
    parser.add_argument("--enable_rollouts", action="store_true",
                       help="Enable rollout instrumentation after training")
    parser.add_argument("--rollout_seeds", type=int, default=10,
                       help="Number of random seeds for rollout evaluation")
    parser.add_argument("--rollout_steps", type=int, default=100,
                       help="Number of steps per rollout episode")
    parser.add_argument("--rollout_model_path", type=str, default=None,
                       help="Specific model path for rollouts (default: use most recent)")
    
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
    if args.run_name:
        # Sanitize run name for filesystem compatibility
        safe_run_name = "".join(c for c in args.run_name if c.isalnum() or c in ('-', '_')).strip()
        log_dir_name = f"run_{run_id}_{safe_run_name}"
    else:
        log_dir_name = f"run_{run_id}"
    
    log_dir = Path(args.log_dir) / log_dir_name
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
    print(f"Original observation shape: {sample_obs.shape}")
    
    # Update effective observation shape for model creation if center cropping is specified
    effective_obs_shape = sample_obs.shape
    if args.center_crop_size is not None:
        print(f"📏 Model will center crop images to {args.center_crop_size}×{args.center_crop_size} pixels")
        # Update effective observation shape for model creation
        if len(sample_obs.shape) == 3:
            effective_obs_shape = (sample_obs.shape[0], args.center_crop_size, args.center_crop_size)
        print(f"Effective observation shape: {effective_obs_shape}")
    
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, config.train_split, config.val_split, config.test_split, config.seed
    )
    
    # Create datasets and dataloaders (no transform needed - cropping is done in model)
    train_dataset = OptomechDataset(train_pairs)
    val_dataset = OptomechDataset(val_pairs)
    test_dataset = OptomechDataset(test_pairs)
    # DataLoader(ds, batch_size=..., shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    
    # Initialize model, optimizer, and criterion
    device_arg = "cpu" if args.force_cpu else config.device
    device, gpu_count = get_device(device_arg)
    
    input_channels = effective_obs_shape[0] if len(effective_obs_shape) == 3 else 1
    model = create_model(args.model_arch, input_channels=input_channels, action_dim=action_dim, 
                        channel_scale=args.channel_scale, mlp_scale=args.mlp_scale, 
                        input_crop_size=args.center_crop_size).to(device)
    
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
        
        # Train with detailed metrics
        train_loss, train_mae_metrics, train_error_distribution, train_abs_error_metrics = train_epoch_with_metrics(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate with detailed metrics
        val_loss, val_mae_metrics, val_error_distribution, val_abs_error_metrics = validate_epoch_with_metrics(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # TensorBoard logging
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Val', val_loss, epoch)
        
        # Training Instrumentation - Example-level MAE
        tb_writer.add_scalar('Training_Instrumentation/MAE_Mean', train_mae_metrics['mae_mean'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Median', train_mae_metrics['mae_median'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Min', train_mae_metrics['mae_min'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Max', train_mae_metrics['mae_max'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Std', train_mae_metrics['mae_std'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Q25', train_mae_metrics['mae_q25'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/MAE_Q75', train_mae_metrics['mae_q75'], epoch)
        tb_writer.add_histogram('Training_Instrumentation/Error_Distribution', train_error_distribution, epoch)
        
        # Training Instrumentation - Action-level absolute error statistics
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Mean', train_abs_error_metrics['abs_error_mean'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Median', train_abs_error_metrics['abs_error_median'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Min', train_abs_error_metrics['abs_error_min'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Max', train_abs_error_metrics['abs_error_max'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Std', train_abs_error_metrics['abs_error_std'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Q25', train_abs_error_metrics['abs_error_q25'], epoch)
        tb_writer.add_scalar('Training_Instrumentation/AbsError_Q75', train_abs_error_metrics['abs_error_q75'], epoch)
        
        # Validation Instrumentation - Example-level MAE
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Mean', val_mae_metrics['mae_mean'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Median', val_mae_metrics['mae_median'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Min', val_mae_metrics['mae_min'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Max', val_mae_metrics['mae_max'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Std', val_mae_metrics['mae_std'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Q25', val_mae_metrics['mae_q25'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/MAE_Q75', val_mae_metrics['mae_q75'], epoch)
        tb_writer.add_histogram('Validation_Instrumentation/Error_Distribution', val_error_distribution, epoch)
        
        # Validation Instrumentation - Action-level absolute error statistics
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Mean', val_abs_error_metrics['abs_error_mean'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Median', val_abs_error_metrics['abs_error_median'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Min', val_abs_error_metrics['abs_error_min'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Max', val_abs_error_metrics['abs_error_max'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Std', val_abs_error_metrics['abs_error_std'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Q25', val_abs_error_metrics['abs_error_q25'], epoch)
        tb_writer.add_scalar('Validation_Instrumentation/AbsError_Q75', val_abs_error_metrics['abs_error_q75'], epoch)

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

        # Log timing metrics to TensorBoard
        tb_writer.add_scalar('Timing/Epoch_Time', epoch_time, epoch)
        tb_writer.add_scalar('Timing/Average_Epoch_Time', avg_epoch_time, epoch)
        tb_writer.add_scalar('Timing/ETA_Hours', eta_hours, epoch)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Train MAE Mean:   {train_mae_metrics['mae_mean']:.6f}")
        print(f"  Train MAE Median: {train_mae_metrics['mae_median']:.6f}")
        print(f"  Train MAE Range:  [{train_mae_metrics['mae_min']:.6f}, {train_mae_metrics['mae_max']:.6f}]")
        print(f"  Val MAE Mean:     {val_mae_metrics['mae_mean']:.6f}")
        print(f"  Val MAE Median:   {val_mae_metrics['mae_median']:.6f}")
        print(f"  Val MAE Range:    [{val_mae_metrics['mae_min']:.6f}, {val_mae_metrics['mae_max']:.6f}]")
        print(f"  Train AbsErr Mean: {train_abs_error_metrics['abs_error_mean']:.6f}")
        print(f"  Val AbsErr Mean:   {val_abs_error_metrics['abs_error_mean']:.6f}")
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
        is_new_best = val_loss < best_val_loss
        if is_new_best:
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
                    'config': config,
                    'model_arch': args.model_arch,
                    'input_channels': input_channels,
                    'action_dim': action_dim
                }, config.model_save_path)
                print(f"  ✅ New best model saved at {config.model_save_path}!")
        
        # Perform rollout instrumentation if enabled
        if args.enable_rollouts:
            print(f"  🎯 Running epoch {epoch+1} rollout instrumentation...")
            try:
                # Use the current best model path if not specified
                rollout_model_path = args.rollout_model_path or config.model_save_path
                
                mean_rewards, std_rewards, rollout_metadata = perform_rollout_instrumentation(
                    model_path=rollout_model_path,
                    num_seeds=args.rollout_seeds,
                    rollout_steps=args.rollout_steps,
                    runs_dir=args.log_dir,
                    save_results=True,
                    output_dir=str(log_dir / f"rollouts_epoch_{epoch+1}")
                )
                
                # Log rollout results to TensorBoard for this epoch
                tb_writer.add_scalar('Rollout/Mean_Episode_Reward', rollout_metadata['total_mean_return'], epoch)
                tb_writer.add_scalar('Rollout/Std_Episode_Reward', rollout_metadata['total_std_return'], epoch)
                tb_writer.add_scalar('Rollout/Final_Step_Mean_Reward', rollout_metadata['final_mean_reward'], epoch)
                tb_writer.add_scalar('Rollout/Final_Step_Std_Reward', rollout_metadata['final_std_reward'], epoch)
                tb_writer.add_scalar('Rollout/Successful_Seeds', rollout_metadata['num_successful_rollouts'], epoch)
                
                # If this is the best model so far, save rollout plots separately
                if is_new_best:
                    best_rollout_dir = log_dir / "best_model_rollouts"
                    best_rollout_dir.mkdir(exist_ok=True)
                    
                    # Copy the rollout plots to the best model directory
                    rollout_dir = Path(rollout_metadata['output_dir'])
                    
                    for plot_file in rollout_dir.glob("*.png"):
                        shutil.copy2(plot_file, best_rollout_dir / f"best_{plot_file.name}")
                    
                    for data_file in rollout_dir.glob("*.json"):
                        shutil.copy2(data_file, best_rollout_dir / f"best_{data_file.name}")
                    
                    print(f"  📊 Best model rollout plots saved to: {best_rollout_dir}")
                
                print(f"  ✅ Rollout completed! Episode reward: {rollout_metadata['total_mean_return']:.4f} ± {rollout_metadata['total_std_return']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Rollout instrumentation failed: {e}")
                print("  Training will continue normally.")
        
        # Save training curves periodically (every 10 epochs or on best model)
        if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
            if config.plot_losses and len(train_losses) > 1:
                plot_save_path = str(log_dir / "training_curves.png")
                plot_training_curves(train_losses, val_losses, plot_save_path)
                # Don't print every time to avoid spam
                if (epoch + 1) % 100 == 0 or val_loss < best_val_loss:
                    print(f"  📈 Training curves updated at {plot_save_path}")
    
    total_training_time = time.time() - training_start_time
    
    # Test final model
    print(f"\n🧪 Testing final model...")
    test_loss, test_mae_metrics, test_error_distribution, test_abs_error_metrics = validate_epoch_with_metrics(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE Mean: {test_mae_metrics['mae_mean']:.6f}")
    print(f"Test MAE Median: {test_mae_metrics['mae_median']:.6f}")
    print(f"Test MAE Range: [{test_mae_metrics['mae_min']:.6f}, {test_mae_metrics['mae_max']:.6f}]")
    
    # Print action-level absolute error summary
    print(f"\nAction-level absolute error summary:")
    print(f"  AbsError Mean: {test_abs_error_metrics['abs_error_mean']:.6f}")
    print(f"  AbsError Median: {test_abs_error_metrics['abs_error_median']:.6f}")
    print(f"  AbsError Range: [{test_abs_error_metrics['abs_error_min']:.6f}, {test_abs_error_metrics['abs_error_max']:.6f}]")
    
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
        print(f"  Test MAE mean:      {test_mae_metrics['mae_mean']:.6f}")
        print(f"  Test MAE median:    {test_mae_metrics['mae_median']:.6f}")
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

    # Log final test metrics to TensorBoard
    tb_writer.add_scalar('Final/Test_Loss', test_loss, config.num_epochs)
    tb_writer.add_scalar('Final/Test_MAE_Mean', test_mae_metrics['mae_mean'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_MAE_Median', test_mae_metrics['mae_median'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_MAE_Min', test_mae_metrics['mae_min'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_MAE_Max', test_mae_metrics['mae_max'], config.num_epochs)
    tb_writer.add_histogram('Final/Test_Error_Distribution', test_error_distribution, config.num_epochs)
    
    # Log final absolute error test metrics
    tb_writer.add_scalar('Final/Test_AbsError_Mean', test_abs_error_metrics['abs_error_mean'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_AbsError_Median', test_abs_error_metrics['abs_error_median'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_AbsError_Min', test_abs_error_metrics['abs_error_min'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_AbsError_Max', test_abs_error_metrics['abs_error_max'], config.num_epochs)
    tb_writer.add_scalar('Final/Test_AbsError_Std', test_abs_error_metrics['abs_error_std'], config.num_epochs)

    tb_writer.close()
    
    print("\n🎉 Training complete!")
    
    # Perform rollout instrumentation if enabled
    if args.enable_rollouts:
        print("\n🎯 Starting post-training rollout instrumentation...")
        try:
            # Use the current model path if not specified
            rollout_model_path = args.rollout_model_path or config.model_save_path
            
            mean_rewards, std_rewards, rollout_metadata = perform_rollout_instrumentation(
                model_path=rollout_model_path,
                num_seeds=args.rollout_seeds,
                rollout_steps=args.rollout_steps,
                runs_dir=args.log_dir,
                save_results=True,
                output_dir=str(log_dir)
            )
            
            # Log rollout results to TensorBoard
            tb_writer = SummaryWriter(log_dir=str(log_dir))
            
            # Log rollout summary statistics
            tb_writer.add_scalar('Rollout/Final_Mean_Reward', rollout_metadata['final_mean_reward'], config.num_epochs)
            tb_writer.add_scalar('Rollout/Final_Std_Reward', rollout_metadata['final_std_reward'], config.num_epochs)
            tb_writer.add_scalar('Rollout/Total_Mean_Return', rollout_metadata['total_mean_return'], config.num_epochs)
            tb_writer.add_scalar('Rollout/Total_Std_Return', rollout_metadata['total_std_return'], config.num_epochs)
            tb_writer.add_scalar('Rollout/Successful_Seeds', rollout_metadata['num_successful_rollouts'], config.num_epochs)
            
            # Log the reward timeseries
            for step, (mean_r, std_r) in enumerate(zip(mean_rewards, std_rewards)):
                tb_writer.add_scalar('Rollout_Timeseries/Mean_Reward', mean_r, step)
                tb_writer.add_scalar('Rollout_Timeseries/Std_Reward', std_r, step)
            
            tb_writer.close()
            
            print(f"✅ Rollout instrumentation completed!")
            print(f"📊 Results: {rollout_metadata['total_mean_return']:.4f} ± {rollout_metadata['total_std_return']:.4f}")
            print(f"📁 Detailed results saved to: {rollout_metadata['output_dir']}")
            
        except Exception as e:
            print(f"❌ Rollout instrumentation failed: {e}")
            print("Training results are still valid and saved.")
    
    print(f"\n📁 All results saved to: {log_dir}")
    print("🎉 Script completed successfully!")


if __name__ == "__main__":
    main()
