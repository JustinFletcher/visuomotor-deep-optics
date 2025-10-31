#!/usr/bin/env python3
"""
Behavior Cloning Training Script with Unified Utilities

Trains a ResNet-18 based actor model to predict sa_incremental_actions from SA datasets.
Features:
- Uses unified dataset utilities with log-scaling support
- Center 256-pixel cropping
- TensorBoard logging
- Environment rollouts after validation epochs
- Forces incremental control mode for rollouts
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
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

# Add parent directories for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))

# Import unified dataset utilities
from utils.datasets import LazyDataset
from utils.transforms import log_scale_transform, center_crop_transform, normalize_transform

# Import model utilities
from models.models import ResNet18Actor

# Import rollout instrumentation
try:
    from optomech.eval.rollout_instrumentation import perform_rollout_instrumentation
except ImportError:
    print("⚠️  Rollout instrumentation not available")
    perform_rollout_instrumentation = None


@dataclass
class TrainingConfig:
    """Configuration for behavior cloning training"""
    dataset_path: str = "datasets/sa_dataset_100k"
    target_action_key: str = "sa_incremental_actions"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    device: str = "auto"
    save_model: bool = True
    model_save_path: str = "saved_models/bc_resnet18_model.pth"
    plot_losses: bool = True
    seed: int = 42
    
    # Model architecture settings
    input_crop_size: int = 256
    log_scale: bool = True
    
    # Training settings
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    grad_clip: float = None
    
    # Data loading settings
    num_workers: int = 4
    pin_memory: bool = True
    max_examples: int = None
    no_dataparallel: bool = False
    
    # Rollout settings
    enable_rollouts: bool = True
    rollout_seeds: int = 8
    rollout_steps: int = 250
    rollout_episode_steps: int = 100
    force_incremental_mode: bool = True
    
    # Pre-trained encoder settings
    pretrained_encoder_path: str = None
    freeze_encoder: bool = False


def get_device(device_str: str = "auto") -> Tuple[torch.device, int]:
    """
    Get torch device and GPU count.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Tuple of (device, gpu_count)
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            print(f"Using CUDA device with {gpu_count} GPU(s)")
            
            # Print GPU info
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Check memory
            try:
                memory_free, memory_total = torch.cuda.mem_get_info(device)
                print(f"  GPU Memory: {memory_free / 1e9:.1f}GB free / {memory_total / 1e9:.1f}GB total")
            except:
                pass
            
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
        gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
        print(f"Using specified device: {device}")
        return device, gpu_count


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, grad_clip: float = None) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for observations, actions in tqdm(dataloader, desc="Training", leave=False):
        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(observations)
        loss = criterion(predictions, actions)

        # Backward + step
        loss.backward()
        
        # Gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, Dict]:
    """
    Validate for one epoch and return loss + metrics.
    
    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_errors = []

    with torch.no_grad():
        for observations, actions in tqdm(dataloader, desc="Validation", leave=False):
            observations = observations.to(device)
            actions = actions.to(device)

            predictions = model(observations)
            loss = criterion(predictions, actions)

            # Calculate per-action MAE
            mae = torch.abs(predictions - actions)
            all_errors.append(mae.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    
    # Compute error statistics
    all_errors = np.concatenate(all_errors, axis=0)
    mae_mean = np.mean(all_errors)
    mae_median = np.median(all_errors)
    mae_std = np.std(all_errors)
    
    metrics = {
        'mae_mean': mae_mean,
        'mae_median': mae_median,
        'mae_std': mae_std,
        'mae_min': np.min(all_errors),
        'mae_max': np.max(all_errors)
    }
    
    return avg_loss, metrics


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   train_loss: float, val_loss: float, config: TrainingConfig,
                   save_path: str, is_best: bool = False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        shutil.copy2(save_path, best_path)
        print(f"💾 Saved best model to {best_path}")


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                         save_path: str = None):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Behavior Cloning Training Progress', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved training curves to {save_path}")
    
    plt.close()


def train_behavior_cloning(config: TrainingConfig):
    """Main training function for behavior cloning."""
    print("=" * 80)
    print("🚀 Starting Behavior Cloning Training with Unified Utilities")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
    
    # Get device
    device, gpu_count = get_device(config.device)
    
    # Setup log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / f"bc_run_{timestamp}_{uuid.uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📊 TensorBoard logs: {log_dir}")
    
    # Save configuration
    config_save_path = log_dir / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"💾 Saved config to {config_save_path}")
    
    # Load dataset using unified utilities
    print(f"\n📚 Loading dataset from {config.dataset_path}")
    print(f"  Target action: {config.target_action_key}")
    print(f"  Log-scaling: {config.log_scale}")
    print(f"  Crop size: {config.input_crop_size}px")
    
    full_dataset = LazyDataset(
        dataset_path=config.dataset_path,
        task_type='behavior_cloning',
        input_crop_size=config.input_crop_size,
        max_examples=config.max_examples,
        log_scale=config.log_scale,
        use_cache=True,
        target_action_key=config.target_action_key
    )
    
    print(f"✅ Loaded {len(full_dataset)} examples")
    
    # Split dataset
    train_size = int(config.train_split * len(full_dataset))
    val_size = int(config.val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
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
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == 'cuda'
    )
    
    # Get data dimensions
    sample_obs, sample_action = full_dataset[0]
    input_channels = sample_obs.shape[0]
    action_dim = sample_action.shape[0]
    
    print(f"\n📐 Data Information:")
    print(f"  Observation shape: {sample_obs.shape}")
    print(f"  Input channels: {input_channels}")
    print(f"  Action dimension: {action_dim}")
    
    # Create ResNet-18 actor model
    print(f"\n🏗️  Creating ResNet-18 actor model...")
    if config.pretrained_encoder_path:
        print(f"  Loading pre-trained encoder from: {config.pretrained_encoder_path}")
        print(f"  Freeze encoder: {config.freeze_encoder}")
    
    model = ResNet18Actor(
        input_channels=input_channels, 
        action_dim=action_dim,
        pretrained_encoder_path=config.pretrained_encoder_path,
        freeze_encoder=config.freeze_encoder
    )
    
    # Print model summary
    try:
        sample_input = torch.randn(1, input_channels, config.input_crop_size, config.input_crop_size)
        summary(model, input_data=sample_input, verbose=0)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        if config.freeze_encoder:
            frozen_params = total_params - trainable_params
            print(f"  Frozen parameters: {frozen_params:,}")
    except Exception as e:
        print(f"  Model summary failed: {e}")
    
    # Multi-GPU setup
    if gpu_count > 1 and not config.no_dataparallel:
        print(f"🔧 Using DataParallel with {gpu_count} GPUs")
        model = nn.DataParallel(model)
    elif gpu_count > 1 and config.no_dataparallel:
        print(f"⚠️  DataParallel disabled (found {gpu_count} GPUs but using only 1)")
    
    model.to(device)
    
    # Create loss function and optimizer
    criterion = nn.MSELoss()
    
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Loss function: MSE")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    if config.grad_clip:
        print(f"  Gradient clipping: {config.grad_clip}")
    
    # Training loop
    print("\n🎯 Starting Training...")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config.grad_clip)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Val_MAE_Mean', val_metrics['mae_mean'], epoch)
        writer.add_scalar('Metrics/Val_MAE_Median', val_metrics['mae_median'], epoch)
        writer.add_scalar('Metrics/Val_MAE_Std', val_metrics['mae_std'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val MAE: {val_metrics['mae_mean']:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if config.save_model:
                checkpoint_path = log_dir / "bc_model_best.pth"
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss,
                              config, str(checkpoint_path), is_best=True)
        
        # Perform rollout instrumentation after validation
        if config.enable_rollouts and perform_rollout_instrumentation is not None:
            print(f"  🎯 Running rollout instrumentation...")
            try:
                # Get the model to use for rollouts (unwrap DataParallel if needed)
                rollout_model = model.module if (gpu_count > 1 and not config.no_dataparallel) else model
                
                # Save temporary model for rollout
                temp_model_path = log_dir / f"temp_model_epoch_{epoch+1}.pth"
                torch.save({
                    'model_state_dict': rollout_model.state_dict(),
                    'epoch': epoch + 1,
                    'config': config
                }, temp_model_path)
                
                # Prepare environment config matching dataset generation settings
                # These MUST match the settings used in optomech/optimization/sa_dataset_config.json
                # environment_flags should contain ALL command-line arguments exactly as used in dataset generation
                env_config = {
                    "env_id": "optomech-v1",
                    "object_type": "single",
                    "aperture_type": "elf",
                    "reward_function": "align",
                    "observation_mode": "image_only",
                    "focal_plane_image_size_pixels": 512,
                    "ao_interval_ms": 5.0,
                    "control_interval_ms": 5.0,
                    "frame_interval_ms": 5.0,
                    "decision_interval_ms": 5.0,
                    "num_atmosphere_layers": 0,
                    "environment_flags": [
                        "--object_type=single",
                        "--ao_interval_ms=5.0",
                        "--control_interval_ms=5.0",
                        "--frame_interval_ms=5.0",
                        "--decision_interval_ms=5.0",
                        "--num_atmosphere_layers=0",
                        "--aperture_type=elf",
                        "--focal_plane_image_size_pixels=512",
                        "--observation_mode=image_only",
                        "--command_secondaries",
                        "--init_differential_motion",
                        "--model_wind_diff_motion",
                        "--num_envs=1",
                        "--reward_function=align",
                        "--max_episode_steps=10_000",
                        "--init_temperature=10.0",
                        "--std_dev_patience=10",
                        "--sparsity_patience=10000000",
                        "--temperature_patience=10000000",
                        "--init_std_dev=1.0",
                        "--scale_patience=10",
                        "--scale_stickiness=1.0"
                    ],
                    "log_scale": config.log_scale,  # Track preprocessing used during training
                    "input_crop_size": config.input_crop_size,  # Track crop size used during training
                    "rollout_episode_steps": config.rollout_episode_steps  # Override max_episode_steps for rollouts
                }
                
                # Force incremental control mode (dataset was collected with SA = incremental actions)
                if config.force_incremental_mode:
                    if "--incremental_action" not in env_config["environment_flags"]:
                        env_config["environment_flags"].append("--incremental_action")
                
                env_config_path = log_dir / "rollout_env_config.json"
                with open(env_config_path, 'w') as f:
                    json.dump(env_config, f, indent=2)
                
                # Perform rollout with BC model type
                mean_rewards, std_rewards, rollout_metadata = perform_rollout_instrumentation(
                    model_path=str(temp_model_path),
                    num_seeds=config.rollout_seeds,
                    rollout_steps=config.rollout_steps,
                    runs_dir=str(log_dir.parent),
                    save_results=True,
                    output_dir=str(log_dir / f"rollouts_epoch_{epoch+1}"),
                    env_config_path=str(env_config_path),
                    model_type="bc"  # Specify BC model type
                )
                
                # Log rollout results to TensorBoard
                if 'total_mean_return' in rollout_metadata:
                    writer.add_scalar('Rollout/Mean_Episode_Reward', 
                                    rollout_metadata['total_mean_return'], epoch)
                if 'total_std_return' in rollout_metadata:
                    writer.add_scalar('Rollout/Std_Episode_Reward',
                                    rollout_metadata['total_std_return'], epoch)
                if 'num_successful_rollouts' in rollout_metadata:
                    writer.add_scalar('Rollout/Successful_Seeds',
                                    rollout_metadata['num_successful_rollouts'], epoch)
                
                print(f"  ✅ Rollout complete: {rollout_metadata}")
                
                if 'total_mean_return' in rollout_metadata and 'total_std_return' in rollout_metadata:
                    print(f"  ✅ Rollout completed! Episode reward: "
                          f"{rollout_metadata['total_mean_return']:.4f} ± "
                          f"{rollout_metadata['total_std_return']:.4f}")
                else:
                    print(f"  ⚠️  Rollout completed but metrics unavailable")
                
                # Clean up temporary model
                temp_model_path.unlink()
                
            except Exception as e:
                print(f"  ❌ Rollout failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save training curves periodically
        if (epoch + 1) % 10 == 0 or is_best:
            plot_save_path = log_dir / "training_curves.png"
            plot_training_curves(train_losses, val_losses, str(plot_save_path))
    
    total_training_time = time.time() - training_start_time
    
    # Test final model
    print(f"\n🧪 Testing final model...")
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE Mean: {test_metrics['mae_mean']:.6f}")
    print(f"Test MAE Median: {test_metrics['mae_median']:.6f}")
    print(f"Test MAE Range: [{test_metrics['mae_min']:.6f}, {test_metrics['mae_max']:.6f}]")
    
    # Print training summary
    if len(train_losses) > 0:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n📊 Training Summary:")
        print(f"  Initial train loss: {initial_loss:.6f}")
        print(f"  Final train loss:   {final_loss:.6f}")
        print(f"  Improvement:        {improvement:.1f}%")
        print(f"  Best val loss:      {best_val_loss:.6f}")
        print(f"  Final test loss:    {test_loss:.6f}")
        print(f"  Test MAE mean:      {test_metrics['mae_mean']:.6f}")
        print(f"  Total time:         {total_training_time/60:.1f} minutes")
        
        if improvement > 5:
            print("✅ Training loss decreased significantly - model is learning!")
        elif improvement > 0:
            print("⚠️  Training loss decreased slightly - consider longer training")
        else:
            print("❌ Training loss did not decrease - check hyperparameters")
    
    # Save final model
    if config.save_model:
        final_model = model.module if (gpu_count > 1 and not config.no_dataparallel) else model
        final_model_path = log_dir / "bc_model_final.pth"
        
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': config.num_epochs,
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'config': config,
            'input_channels': input_channels,
            'action_dim': action_dim
        }, final_model_path)
        
        print(f"\n💾 Final model saved to {final_model_path}")
    
    # Plot final training curves
    if config.plot_losses:
        plot_save_path = log_dir / "training_curves_final.png"
        plot_training_curves(train_losses, val_losses, str(plot_save_path))
    
    # Log final test metrics to TensorBoard
    writer.add_scalar('Final/Test_Loss', test_loss, config.num_epochs)
    writer.add_scalar('Final/Test_MAE_Mean', test_metrics['mae_mean'], config.num_epochs)
    writer.add_scalar('Final/Test_MAE_Median', test_metrics['mae_median'], config.num_epochs)
    
    writer.close()
    
    print(f"\n📁 All results saved to: {log_dir}")
    print("🎉 Training complete!")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Behavior Cloning Training with Unified Utilities"
    )
    
    # Dataset settings
    parser.add_argument("--dataset-path", type=str, default="datasets/sa_dataset_100k",
                       help="Path to SA dataset")
    parser.add_argument("--target-action", type=str, default="sa_incremental_action",
                       help="Target action key in dataset")
    parser.add_argument("--max-examples", type=int, help="Limit dataset size for debugging")
    
    # Model settings
    parser.add_argument("--input-crop-size", type=int, default=256,
                       help="Center crop size for observations")
    parser.add_argument("--log-scale", action="store_true", default=True,
                       help="Apply log-scaling to observations")
    parser.add_argument("--no-log-scale", action="store_false", dest="log_scale",
                       help="Disable log-scaling")
    parser.add_argument("--pretrained-encoder", type=str, 
                       help="Path to pre-trained autoencoder model to load encoder from")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze encoder weights (fine-tune action head only)")
    
    # Training settings
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=['adam', 'adamw'],
                       help="Optimizer type")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--grad-clip", type=float, help="Gradient clipping threshold")
    
    # Device settings
    parser.add_argument("--device", type=str, default="auto",
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help="Device to use for training")
    parser.add_argument("--no-dataparallel", action="store_true",
                       help="Disable DataParallel for multi-GPU training")
    
    # Output settings
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save model")
    
    # Rollout settings
    parser.add_argument("--enable-rollouts", action="store_true", default=True,
                       help="Enable environment rollouts after validation")
    parser.add_argument("--no-rollouts", action="store_false", dest="enable_rollouts",
                       help="Disable environment rollouts")
    parser.add_argument("--rollout-seeds", type=int, default=8,
                       help="Number of rollout seeds")
    parser.add_argument("--rollout-steps", type=int, default=250,
                       help="Number of steps per rollout")
    parser.add_argument("--rollout-episode-steps", type=int, default=100,
                       help="Maximum episode length for rollouts (overrides dataset config)")
    parser.add_argument("--force-incremental-mode", action="store_true", default=True,
                       help="Force incremental control mode for rollouts")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        dataset_path=args.dataset_path,
        target_action_key=args.target_action,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        input_crop_size=args.input_crop_size,
        log_scale=args.log_scale,
        save_model=not args.no_save,
        max_examples=args.max_examples,
        seed=args.seed,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        no_dataparallel=args.no_dataparallel,
        enable_rollouts=args.enable_rollouts,
        rollout_seeds=args.rollout_seeds,
        rollout_steps=args.rollout_steps,
        rollout_episode_steps=args.rollout_episode_steps,
        force_incremental_mode=args.force_incremental_mode,
        pretrained_encoder_path=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder
    )
    
    # Print configuration
    print("🔧 Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_behavior_cloning(config)


if __name__ == "__main__":
    main()
