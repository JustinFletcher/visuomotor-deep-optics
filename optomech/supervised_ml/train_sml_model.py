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
import time
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
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
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Path to checkpoint file to resume training from")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load dataset
    config = TrainingConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume_from
    )
    
    print("🚀 Starting SML Model Training")
    print("=" * 50)
    
    # Load and split dataset
    print("� Loading dataset...")
    pairs = load_dataset_pairs_sequential(config.dataset_path, use_cache=True)
    
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
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Initialize model, optimizer, and criterion
    device, gpu_count = get_device(config.device)
    
    input_channels = sample_obs.shape[0] if len(sample_obs.shape) == 3 else 1
    model = SMLModel(input_channels=input_channels, action_dim=action_dim).to(device)
    
    # Enable DataParallel for multi-GPU training
    if gpu_count > 1 and device.type == "cuda":
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
    print(f"  Input channels: {input_channels}")
    print(f"  Action dimension: {action_dim}")
    
    # Count parameters (handle DataParallel wrapper)
    param_model = model.module if isinstance(model, DataParallel) else model
    total_params = sum(p.numel() for p in param_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    if gpu_count > 1 and device.type == "cuda":
        print(f"  Parameters per GPU: ~{total_params // gpu_count:,}")
    
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
                os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
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
                print(f"  ✅ New best model saved!")
    
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
        plot_save_path = config.model_save_path.replace('.pth', '_training_curves.png')
        plot_training_curves(train_losses, val_losses, plot_save_path)
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
