#!/usr/bin/env python3
"""
Episode-Based Behavior Cloning Training Script with TBPTT

Trains a BC model using the same episode dataset and TBPTT approach as world model training.
The BC model predicts actions from observations using the same encoder + LSTM architecture,
but with an action head instead of a decoder.

Architecture:
- Uses pretrained autoencoder encoder
- LSTM for temporal modeling
- Fusion MLP to combine encoder and LSTM outputs
- Action head to predict actions
- TBPTT for sequence learning
- Pre-collated batches for efficiency

This script is designed to be compatible with world model training pipelines.
"""

import os
import sys
import json
import argparse
import time
import uuid
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import dataset and utilities
from optomech.world_models.episode_dataset import WorldModelEpisodeDataset, collate_episodes_padded
from utils.data_loading import DatasetDiscovery

# Import model
from models.bc_model import BCModel

# Import autoencoder architecture utilities
from models.autoencoders import (
    create_autoencoder_model,
    detect_autoencoder_architecture,
    load_autoencoder_weights
)


@dataclass
class BCConfig:
    """Configuration for behavior cloning training"""
    # Dataset settings
    dataset_path: str = "datasets/sa_dataset_1m"
    obs_key: str = "observations"
    action_key: str = "sa_incremental_actions"
    
    # Model architecture
    pretrained_autoencoder_path: str = None
    hidden_dim: int = 512
    num_lstm_layers: int = 1
    fusion_hidden_dim: int = 512
    action_head_hidden_dim: int = 128
    freeze_encoder: bool = True
    latent_dim: int = 256
    input_crop_size: int = 256
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    sequence_length: int = 32  # For TBPTT
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss_function: str = "mse"
    
    # Episode settings
    use_episodes: bool = True
    min_episode_length: int = 1
    max_episode_length: int = None
    
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05
    max_examples: int = None
    num_workers: int = 0
    prefetch_factor: int = None
    persistent_workers: bool = False
    pin_memory: bool = True
    load_in_memory: bool = True
    log_scale: bool = True
    
    # Training optimizations
    use_amp: bool = False
    
    # Scheduler settings
    use_scheduler: bool = False
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Output settings
    device: str = "auto"
    save_model: bool = True
    model_save_path: str = "saved_models/bc_model.pth"
    run_name: str = "bc_default"
    runs_dir: str = "runs"
    seed: int = 42
    checkpoint_interval: int = 50


class MaskedLoss(nn.Module):
    """Loss function that handles padded sequences with masking."""
    
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(self, predictions, targets, mask):
        """
        Compute masked loss.
        
        Args:
            predictions: [batch, seq_len, action_dim]
            targets: [batch, seq_len, action_dim]
            mask: [batch, seq_len] - 1 for valid, 0 for padding
        
        Returns:
            Scalar loss
        """
        # Compute element-wise loss
        if isinstance(self.base_criterion, nn.MSELoss):
            # MSE: (pred - target)^2
            element_loss = (predictions - targets) ** 2
            # Average over action dimension
            element_loss = element_loss.mean(dim=-1)  # [batch, seq_len]
        elif isinstance(self.base_criterion, nn.L1Loss):
            # MAE: |pred - target|
            element_loss = torch.abs(predictions - targets)
            element_loss = element_loss.mean(dim=-1)  # [batch, seq_len]
        else:
            # For other losses, compute on flattened tensors then reshape
            batch_size, seq_len, action_dim = predictions.shape
            pred_flat = predictions.reshape(-1, action_dim)
            target_flat = targets.reshape(-1, action_dim)
            loss_flat = self.base_criterion(pred_flat, target_flat)
            element_loss = loss_flat.reshape(batch_size, seq_len)
        
        # Apply mask
        masked_loss = element_loss * mask.float()
        
        # Compute mean over valid elements
        num_valid = mask.float().sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum()  # Return 0 if no valid elements


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔧 Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🔧 Using device: mps (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print(f"🔧 Using device: cpu")
    else:
        device = torch.device(device_str)
        print(f"🔧 Using device: {device_str}")
    
    return device


def train_epoch_batched(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    sequence_length: int,
    grad_clip: float = None,
    scaler=None
) -> float:
    """
    Train for one epoch with batched episode processing and TBPTT.
    
    Args:
        model: BC model
        dataloader: DataLoader (returns pre-collated batches if batch preloading enabled)
        optimizer: Optimizer
        criterion: Masked loss function
        device: Device
        sequence_length: TBPTT sequence length
        grad_clip: Gradient clipping value
        scaler: AMP gradient scaler (optional)
    
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_sequences = 0
    
    for batch_data in tqdm(dataloader, desc="Training", leave=False):
        # Unpack batch (pre-collated format)
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 5:
            obs_padded, actions_padded, next_obs_padded, lengths, mask = batch_data
        else:
            # Fallback: apply collation if not pre-collated
            obs_padded, actions_padded, next_obs_padded, lengths, mask = collate_episodes_padded(batch_data)
        
        obs_padded = obs_padded.to(device)
        actions_padded = actions_padded.to(device)  # These are the target actions
        mask = mask.to(device)
        
        batch_size, max_len = obs_padded.shape[0], obs_padded.shape[1]
        
        # Initialize hidden state
        hidden = model.get_zero_hidden(batch_size, device)
        
        # Process episode in chunks using TBPTT
        epoch_loss = 0.0
        num_chunks = 0
        
        for start_idx in range(0, max_len, sequence_length):
            end_idx = min(start_idx + sequence_length, max_len)
            
            # Get chunk
            obs_chunk = obs_padded[:, start_idx:end_idx]
            action_chunk = actions_padded[:, start_idx:end_idx]
            mask_chunk = mask[:, start_idx:end_idx]
            
            # Skip if chunk has no valid timesteps
            if mask_chunk.sum() == 0:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    action_pred, _, hidden = model(obs_chunk, hidden)
                    loss = criterion(action_pred, action_chunk, mask_chunk)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                action_pred, _, hidden = model(obs_chunk, hidden)
                loss = criterion(action_pred, action_chunk, mask_chunk)
                
                loss.backward()
                
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            # Detach hidden state for TBPTT
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            epoch_loss += loss.item()
            num_chunks += 1
        
        if num_chunks > 0:
            total_loss += epoch_loss / num_chunks
            num_sequences += 1
    
    return total_loss / max(num_sequences, 1)


def validate_epoch_batched(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate for one epoch with batched episode processing.
    
    Args:
        model: BC model
        dataloader: DataLoader
        criterion: Masked loss function
        device: Device
    
    Returns:
        Average loss for epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation", leave=False):
            # Unpack batch
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 5:
                obs_padded, actions_padded, next_obs_padded, lengths, mask = batch_data
            else:
                obs_padded, actions_padded, next_obs_padded, lengths, mask = collate_episodes_padded(batch_data)
            
            obs_padded = obs_padded.to(device)
            actions_padded = actions_padded.to(device)
            mask = mask.to(device)
            
            batch_size = obs_padded.shape[0]
            
            # Forward pass (full episode, no TBPTT for validation)
            hidden = model.get_zero_hidden(batch_size, device)
            action_pred, _, _ = model(obs_padded, hidden)
            
            # Compute loss
            loss = criterion(action_pred, actions_padded, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_bc(config: BCConfig):
    """Main BC training function."""
    print("=" * 80)
    print("🚀 Starting Episode-Based Behavior Cloning Training")
    print("=" * 80)
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
    
    # Get device
    device = get_device(config.device)
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.runs_dir) / f"{config.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    
    print(f"📁 Run directory: {run_dir}")
    print(f"📊 TensorBoard: tensorboard --logdir={config.runs_dir}")
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"💾 Config saved to {config_path}")
    
    # Discover dataset files
    print(f"\n📂 Discovering dataset files in: {config.dataset_path}")
    file_paths, dataset_type, metadata = DatasetDiscovery.discover_files(config.dataset_path)
    print(f"📊 Found {len(file_paths)} files of type '{dataset_type}'")
    
    # Setup transforms
    print(f"\n🎨 Setting up data transforms...")
    print(f"   Log scale: {config.log_scale}")
    
    transforms = None
    if config.log_scale:
        from optomech.world_models.transforms import create_log_scale_transform
        transforms = create_log_scale_transform(normalize=True)
        print(f"   ✅ Log-scale transform created")
    
    # Create episode dataset with batch preloading
    print(f"\n📂 Creating EPISODE-BASED dataset...")
    print(f"   Observation key: {config.obs_key}")
    print(f"   Action key: {config.action_key}")
    print(f"   Min episode length: {config.min_episode_length}")
    print(f"   Max episode length: {config.max_episode_length if config.max_episode_length else 'None (unlimited)'}")
    print(f"   Batch size: {config.batch_size}")
    
    dataset = WorldModelEpisodeDataset(
        file_paths=file_paths,
        dataset_type=dataset_type,
        transforms=transforms,
        obs_key=config.obs_key,
        action_key=config.action_key,
        min_episode_length=config.min_episode_length,
        max_episode_length=config.max_episode_length,
        load_in_memory=False,  # Will enable after filtering
        batch_size=None  # Will set after filtering
    )
    
    # Filter by max_examples
    if config.max_examples:
        original_count = len(dataset.episode_data)
        dataset.episode_data = dataset.episode_data[:config.max_examples]
        print(f"   Limited to {len(dataset.episode_data)} episodes (max_examples={config.max_examples}, was {original_count})")
    
    # Enable preloading with batch collation
    if config.load_in_memory:
        dataset.load_in_memory = True
        dataset.batch_size = config.batch_size
        if dataset.batch_size is not None:
            dataset._preload_batches()
        else:
            dataset._preload_episodes()
    
    # Collate function and batch_size
    if dataset.preloaded_batches is not None:
        collate_fn = lambda x: x[0]  # Just return the pre-collated batch
        batch_size = 1
        print(f"   ✅ Using pre-collated batches (batch_size={config.batch_size} episodes per batch)")
    else:
        collate_fn = collate_episodes_padded
        batch_size = config.batch_size
        print(f"   Using dynamic collation (batch_size={config.batch_size})")
    
    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val:   {len(val_dataset)}")
    print(f"   Test:  {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda',
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    # Infer action dimension from dataset
    print(f"\n🔍 Inferring dimensions from dataset...")
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 5:
        sample_obs, sample_actions, _, _, _ = sample_batch
    else:
        sample_obs, sample_actions, _, _, _ = collate_episodes_padded(sample_batch)
    
    action_dim = sample_actions.shape[-1]
    print(f"   Detected action_dim: {action_dim}")
    print(f"   Observation shape: {sample_obs.shape}")
    
    # Load pretrained autoencoder
    print(f"\n📦 Loading pretrained autoencoder from: {config.pretrained_autoencoder_path}")
    arch_type = detect_autoencoder_architecture(config.pretrained_autoencoder_path)
    print(f"  🔍 Detected architecture: {arch_type}")
    
    autoencoder, input_channels, detected_latent_dim = create_autoencoder_model(
        arch_type,
        input_channels=1,
        latent_dim=config.latent_dim,
        device=device
    )
    
    load_autoencoder_weights(autoencoder, config.pretrained_autoencoder_path, device)
    print(f"✅ Loaded pretrained autoencoder (latent_dim={detected_latent_dim})")
    
    # Create BC model
    print(f"\n🌍 Creating BC model...")
    print(f"   Action dim: {action_dim}")
    print(f"   LSTM hidden dim: {config.hidden_dim}")
    print(f"   LSTM layers: {config.num_lstm_layers}")
    print(f"   Fusion hidden dim: {config.fusion_hidden_dim}")
    print(f"   Action head hidden dim: {config.action_head_hidden_dim}")
    print(f"   Freeze encoder: {config.freeze_encoder}")
    
    model = BCModel(
        encoder=autoencoder.encoder,
        latent_dim=detected_latent_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_lstm_layers,
        fusion_hidden_dim=config.fusion_hidden_dim,
        action_head_hidden_dim=config.action_head_hidden_dim,
        freeze_encoder=config.freeze_encoder
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ BC model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create loss and optimizer
    base_criterion = nn.MSELoss() if config.loss_function == "mse" else nn.L1Loss()
    criterion = MaskedLoss(base_criterion)
    print(f"📉 Loss function: {config.loss_function} (wrapped with MaskedLoss)")
    
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"⚡ Optimizer: {config.optimizer} (lr={config.learning_rate})")
    
    # Scheduler
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
    
    # AMP scaler
    scaler = None
    if config.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print(f"⚡ Automatic Mixed Precision: Enabled")
    
    # Training loop
    print(f"\n🏋️  Starting training for {config.num_epochs} epochs...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch_batched(
            model, train_loader, optimizer, criterion, device,
            config.sequence_length, config.grad_clip, scaler
        )
        
        # Validate
        val_loss = validate_epoch_batched(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.save_model:
                best_path = run_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config.__dict__
                }, best_path)
                print(f"🏆 New best model saved (val_loss: {val_loss:.6f})")
        
        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0 and config.save_model:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Test final model
    print(f"\n🧪 Testing final model...")
    test_loss = validate_epoch_batched(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    writer.add_scalar('Final/Test_Loss', test_loss, config.num_epochs)
    
    # Save final model
    if config.save_model:
        final_path = run_dir / "final_model.pth"
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'config': config.__dict__
        }, final_path)
        print(f"💾 Final model saved: {final_path}")
    
    writer.close()
    print(f"\n🎉 Training complete!")
    print(f"📁 Results saved to: {run_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Episode-Based Behavior Cloning Training")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Dataset settings
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument("--obs-key", type=str, default="observations")
    parser.add_argument("--action-key", type=str, default="sa_incremental_actions")
    parser.add_argument("--max-examples", type=int, help="Limit dataset size")
    
    # Model settings
    parser.add_argument("--pretrained-autoencoder-path", type=str, help="Path to pretrained autoencoder")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-lstm-layers", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    
    # Training settings
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-episode-length", type=int, help="Maximum episode length")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    # Output
    parser.add_argument("--run-name", type=str, default="bc_default")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--no-save", action="store_true")
    
    args = parser.parse_args()
    
    # Load config from JSON if provided
    config_dict = {}
    if args.config:
        print(f"📄 Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        print(f"✅ Loaded config from {args.config}")
    
    # Helper to get value with priority: CLI > config file > default
    def get_value(arg_name, default=None):
        cli_value = getattr(args, arg_name, None)
        if cli_value is not None and (not isinstance(cli_value, bool) or cli_value):
            return cli_value
        if arg_name in config_dict:
            return config_dict[arg_name]
        return default
    
    # Create config
    config = BCConfig(
        dataset_path=get_value('dataset_path'),
        obs_key=get_value('obs_key', 'observations'),
        action_key=get_value('action_key', 'sa_incremental_actions'),
        pretrained_autoencoder_path=get_value('pretrained_autoencoder_path'),
        hidden_dim=get_value('hidden_dim', 512),
        num_lstm_layers=get_value('num_lstm_layers', 1),
        freeze_encoder=get_value('freeze_encoder', True),
        batch_size=get_value('batch_size', 32),
        learning_rate=get_value('learning_rate', 1e-4),
        num_epochs=get_value('num_epochs', 100),
        sequence_length=get_value('sequence_length', 32),
        max_episode_length=get_value('max_episode_length'),
        max_examples=get_value('max_examples'),
        device=get_value('device', 'auto'),
        run_name=get_value('run_name', 'bc_default'),
        runs_dir=get_value('runs_dir', 'runs'),
        save_model=not args.no_save if hasattr(args, 'no_save') else True,
        load_in_memory=get_value('load_in_memory', True),
        log_scale=get_value('log_scale', True),
        use_amp=get_value('use_amp', False)
    )
    
    # Print config
    print("🔧 BC Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Validate required args
    if not config.dataset_path:
        parser.error("--dataset-path is required")
    if not config.pretrained_autoencoder_path:
        parser.error("--pretrained-autoencoder-path is required")
    
    # Start training
    train_bc(config)


if __name__ == "__main__":
    main()
