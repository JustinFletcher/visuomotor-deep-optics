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
- MAE vs Target L2 candlestick plots for error analysis
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
import matplotlib.patches as mpatches
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
    loss_function: str = "mse"  # Options: mse, huber, log_cosh, adaptive_mse
    huber_delta: float = 0.01  # Delta for Huber loss (threshold for small values)
    adaptive_scale: float = 0.01  # Scale threshold for adaptive MSE loss
    
    # Learning rate scheduler settings
    use_scheduler: bool = True
    scheduler_patience: int = 10  # Epochs to wait before reducing LR
    scheduler_factor: float = 0.5  # Factor to reduce LR by
    scheduler_min_lr: float = 1e-7  # Minimum learning rate
    
    # Data loading settings
    num_workers: int = 4
    pin_memory: bool = True
    max_examples: int = None
    no_dataparallel: bool = False
    lazy_loading: bool = False  # Use lazy (on-demand) loading instead of preloading into memory
    
    # Rollout settings
    enable_rollouts: bool = True
    rollout_interval: int = 1  # Perform rollouts every N epochs (default: every epoch)
    rollout_seeds: int = 8
    rollout_steps: int = 250
    rollout_episode_steps: int = 100
    force_incremental_mode: bool = True
    
    # Run naming
    run_name: str = None  # Optional name to append to run directory
    
    # Pre-trained encoder settings
    pretrained_encoder_path: str = None
    freeze_encoder: bool = False


def get_device(device_str: str = "auto") -> Tuple[torch.device, int]:
    """
    Get torch device and GPU count.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu')
        
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
        
        # Determine GPU count based on device type
        if device.type == "cuda":
            gpu_count = torch.cuda.device_count()
            
            # Print info about the specific GPU if specified (e.g., cuda:0)
            if device.index is not None:
                if device.index >= gpu_count:
                    raise ValueError(f"Invalid device {device_str}: only {gpu_count} GPU(s) available")
                print(f"Using specified device: {device} ({torch.cuda.get_device_name(device.index)})")
                
                # Check memory for specific GPU
                try:
                    memory_free, memory_total = torch.cuda.mem_get_info(device)
                    print(f"  GPU Memory: {memory_free / 1e9:.1f}GB free / {memory_total / 1e9:.1f}GB total")
                except:
                    pass
            else:
                print(f"Using specified device: {device} with {gpu_count} GPU(s) available")
        else:
            gpu_count = 1
            print(f"Using specified device: {device}")
        
        return device, gpu_count


def get_loss_function(config: TrainingConfig):
    """
    Get loss function based on config.
    
    Loss functions that emphasize small values (<0.01):
    - mse: Standard MSE loss
    - huber: Huber loss (linear for large errors, quadratic for small)
    - log_cosh: log(cosh(x)) - smooth approximation to absolute error
    - adaptive_mse: MSE with higher weight on small values
    
    Args:
        config: Training configuration
        
    Returns:
        Loss function
    """
    if config.loss_function == "mse":
        print(f"Using MSE loss")
        return nn.MSELoss()
    
    elif config.loss_function == "huber":
        # Huber loss: quadratic for small errors (|error| < delta), linear beyond
        # This emphasizes accurate prediction of small values
        print(f"Using Huber loss (delta={config.huber_delta})")
        return nn.HuberLoss(delta=config.huber_delta)
    
    elif config.loss_function == "log_cosh":
        # log(cosh(x)) loss - smooth approximation to absolute error
        # More sensitive to small errors than MSE
        print(f"Using Log-Cosh loss")
        def log_cosh_loss(pred, target):
            diff = pred - target
            return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
        return log_cosh_loss
    
    elif config.loss_function == "adaptive_mse":
        # Adaptive MSE: higher weight on predictions where |target| < scale
        # This directly emphasizes small values
        print(f"Using Adaptive MSE loss (scale={config.adaptive_scale})")
        def adaptive_mse_loss(pred, target):
            diff_sq = (pred - target) ** 2
            # Weight: higher for small target values
            weight = torch.where(
                torch.abs(target) < config.adaptive_scale,
                torch.tensor(2.0, device=target.device),  # 2x weight for small values
                torch.tensor(1.0, device=target.device)
            )
            return torch.mean(weight * diff_sq)
        return adaptive_mse_loss
    
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}. "
                        f"Choose from: mse, huber, log_cosh, adaptive_mse")


def create_mae_vs_l2_candlestick(target_l2_norms: np.ndarray, 
                                  mae_per_sample: np.ndarray,
                                  title: str,
                                  num_bins: int = 10) -> plt.Figure:
    """
    Create a candlestick plot showing MAE distribution across target L2 norm bins.
    
    Args:
        target_l2_norms: Array of L2 norms of target actions [num_samples]
        mae_per_sample: Array of MAE values per sample [num_samples]
        title: Plot title
        num_bins: Number of L2 bins to create
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bins based on quantiles for more even distribution
    try:
        bin_edges = np.percentile(target_l2_norms, np.linspace(0, 100, num_bins + 1))
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Fallback: if all values are the same, create artificial bins
            bin_edges = np.linspace(target_l2_norms.min() - 0.001, 
                                   target_l2_norms.max() + 0.001, num_bins + 1)
    except Exception as e:
        # Fallback to linear bins if quantile fails
        bin_edges = np.linspace(target_l2_norms.min(), target_l2_norms.max(), num_bins + 1)
    
    bin_centers = []
    bin_widths = []
    candlestick_data = []
    
    for i in range(len(bin_edges) - 1):
        # Find samples in this bin
        mask = (target_l2_norms >= bin_edges[i]) & (target_l2_norms < bin_edges[i+1])
        if i == len(bin_edges) - 2:  # Include upper edge in last bin
            mask = (target_l2_norms >= bin_edges[i]) & (target_l2_norms <= bin_edges[i+1])
        
        bin_mae = mae_per_sample[mask]
        
        if len(bin_mae) > 0:
            # Compute candlestick statistics
            stats = {
                'min': np.min(bin_mae),
                'q25': np.percentile(bin_mae, 25),
                'median': np.median(bin_mae),
                'q75': np.percentile(bin_mae, 75),
                'max': np.max(bin_mae),
                'count': len(bin_mae)
            }
            candlestick_data.append(stats)
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_widths.append(bin_edges[i+1] - bin_edges[i])
    
    if not candlestick_data:
        # No data to plot
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Plot candlesticks
    for i, (center, width, stats) in enumerate(zip(bin_centers, bin_widths, candlestick_data)):
        # Box from Q25 to Q75
        box_height = stats['q75'] - stats['q25']
        box = mpatches.Rectangle(
            (center - width*0.3, stats['q25']),
            width*0.6, box_height,
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax.add_patch(box)
        
        # Median line
        ax.plot([center - width*0.3, center + width*0.3], 
               [stats['median'], stats['median']], 
               'r-', linewidth=2)
        
        # Whiskers (min to Q25, Q75 to max)
        ax.plot([center, center], [stats['min'], stats['q25']], 'k-', linewidth=1)
        ax.plot([center, center], [stats['q75'], stats['max']], 'k-', linewidth=1)
        
        # End caps
        ax.plot([center - width*0.15, center + width*0.15], 
               [stats['min'], stats['min']], 'k-', linewidth=1)
        ax.plot([center - width*0.15, center + width*0.15], 
               [stats['max'], stats['max']], 'k-', linewidth=1)
        
        # Add sample count as text
        ax.text(center, stats['max'], f"n={stats['count']}", 
               ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Target Action L2 Norm', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Q25-Q75 (IQR)'),
        mpatches.Patch(facecolor='red', label='Median'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='Min-Max Whiskers')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig



def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, grad_clip: float = None) -> Tuple[float, Dict, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Train for one epoch and return loss, MAE metrics, raw error distribution, absolute error metrics,
    and data for MAE vs target L2 analysis.
    
    Returns:
        Tuple of (avg_loss, mae_metrics_dict, raw_errors_array, abs_error_metrics_dict, 
                  target_l2_norms_array, mae_per_sample_array)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_mae_errors = []  # Example-level MAE (mean across actions per sample)
    all_raw_errors = []  # All raw errors (predictions - targets) flattened
    all_abs_errors = []  # All absolute errors flattened
    all_target_l2_norms = []  # L2 norm of target action per sample
    all_mae_per_sample = []  # MAE per sample (for MAE vs L2 plot)

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

        # Calculate raw errors for histogram (predictions - actions)
        with torch.no_grad():
            raw_errors = predictions - actions  # [batch_size, action_dim]
            all_raw_errors.extend(raw_errors.cpu().numpy().flatten())
            
            # Calculate absolute errors
            abs_errors = torch.abs(raw_errors)  # [batch_size, action_dim]
            all_abs_errors.extend(abs_errors.cpu().numpy().flatten())
            
            # Calculate MAE for each sample in the batch (example-level)
            mae_per_sample = torch.mean(abs_errors, dim=1)  # [batch_size]
            all_mae_errors.extend(mae_per_sample.cpu().numpy())
            all_mae_per_sample.extend(mae_per_sample.cpu().numpy())
            
            # Calculate L2 norm of target actions per sample
            target_l2 = torch.norm(actions, p=2, dim=1)  # [batch_size]
            all_target_l2_norms.extend(target_l2.cpu().numpy())

        # Track loss
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    
    # Convert to numpy arrays
    mae_errors_array = np.array(all_mae_errors)
    raw_errors_array = np.array(all_raw_errors)
    abs_errors_array = np.array(all_abs_errors)
    target_l2_norms_array = np.array(all_target_l2_norms)
    mae_per_sample_array = np.array(all_mae_per_sample)
    
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
    
    return avg_loss, mae_metrics, raw_errors_array, abs_error_metrics, target_l2_norms_array, mae_per_sample_array


def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, Dict, np.ndarray, Dict]:
    """
    Validate for one epoch and return loss, MAE metrics, raw error distribution, and absolute error metrics.
    
    Returns:
        Tuple of (avg_loss, mae_metrics_dict, raw_errors_array, abs_error_metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_mae_errors = []  # Example-level MAE (mean across actions per sample)
    all_raw_errors = []  # All raw errors (predictions - targets) flattened
    all_abs_errors = []  # All absolute errors flattened
    all_target_l2_norms = []  # L2 norm of target actions for each sample
    all_mae_per_sample = []  # MAE per sample for candlestick plot

    with torch.no_grad():
        for observations, actions in tqdm(dataloader, desc="Validation", leave=False):
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
            
            # Calculate MAE for each sample in the batch (example-level)
            mae_per_sample = torch.mean(abs_errors, dim=1)  # [batch_size]
            all_mae_errors.extend(mae_per_sample.cpu().numpy())
            
            # Track L2 norm of target actions and MAE per sample for candlestick plot
            target_l2 = torch.norm(actions, p=2, dim=1)  # [batch_size]
            all_target_l2_norms.extend(target_l2.cpu().numpy())
            all_mae_per_sample.extend(mae_per_sample.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    
    # Convert to numpy arrays
    mae_errors_array = np.array(all_mae_errors)
    raw_errors_array = np.array(all_raw_errors)
    abs_errors_array = np.array(all_abs_errors)
    target_l2_norms_array = np.array(all_target_l2_norms)
    mae_per_sample_array = np.array(all_mae_per_sample)
    
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
    
    return avg_loss, mae_metrics, raw_errors_array, abs_error_metrics, target_l2_norms_array, mae_per_sample_array


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
    
    # Setup log directory with optional run name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"bc_run_{timestamp}_{uuid.uuid4().hex[:8]}"
    if config.run_name:
        run_dir_name += f"_{config.run_name}"
    log_dir = Path("runs") / run_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📊 TensorBoard logs: {log_dir}")
    
    # Save configuration
    config_save_path = log_dir / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"💾 Saved config to {config_save_path}")
    
    # Save rollout environment config (always, even if rollouts are disabled)
    # This ensures the config is available for manual rollouts later
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
        "log_scale": config.log_scale,
        "input_crop_size": config.input_crop_size,
        "rollout_episode_steps": config.rollout_episode_steps
    }
    
    # Force incremental control mode if configured
    if config.force_incremental_mode:
        if "--incremental_control" not in env_config["environment_flags"]:
            env_config["environment_flags"].append("--incremental_control")
    
    env_config_path = log_dir / "rollout_env_config.json"
    with open(env_config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    print(f"💾 Saved rollout environment config to {env_config_path}")
    
    # Load dataset using unified utilities
    print(f"\n📚 Loading dataset from {config.dataset_path}")
    print(f"  Dataset mode: {'Lazy (on-demand loading)' if config.lazy_loading else 'In-Memory (preload all)'}")
    print(f"  Target action: {config.target_action_key}")
    print(f"  Log-scaling: {config.log_scale}")
    print(f"  Crop size: {config.input_crop_size}px")
    
    if config.lazy_loading:
        # Use LazyDataset for memory-efficient on-demand loading
        print("  🔄 Using lazy loading (on-demand from disk)")
        full_dataset = LazyDataset(
            dataset_path=config.dataset_path,
            task_type='behavior_cloning',
            input_crop_size=config.input_crop_size,
            max_examples=config.max_examples,
            log_scale=config.log_scale,
            use_cache=True,
            target_action_key=config.target_action_key
        )
    else:
        # Use InMemoryDataset for fastest training (loads entire dataset into RAM)
        print("  💾 Using in-memory loading (preload entire dataset)")
        from utils.datasets import InMemoryDataset
        full_dataset = InMemoryDataset(
            dataset_path=config.dataset_path,
            task_type='behavior_cloning',
            input_crop_size=config.input_crop_size,
            max_examples=config.max_examples,
            log_scale=config.log_scale,
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
    
    # Print model summary and capture for TensorBoard
    model_summary_str = None
    try:
        sample_input = torch.randn(1, input_channels, config.input_crop_size, config.input_crop_size)
        
        # Get detailed model summary as string
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        summary(model, input_data=sample_input, verbose=1)
        sys.stdout = old_stdout
        model_summary_str = summary_buffer.getvalue()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        if config.freeze_encoder:
            frozen_params = total_params - trainable_params
            print(f"  Frozen parameters: {frozen_params:,}")
    except Exception as e:
        print(f"  Model summary failed: {e}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Multi-GPU setup
    if gpu_count > 1 and not config.no_dataparallel:
        print(f"🔧 Using DataParallel with {gpu_count} GPUs")
        model = nn.DataParallel(model)
    elif gpu_count > 1 and config.no_dataparallel:
        print(f"⚠️  DataParallel disabled (found {gpu_count} GPUs but using only 1)")
    
    model.to(device)
    
    # Create loss function and optimizer
    criterion = get_loss_function(config)
    
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            threshold=1e-5,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )
        print(f"\n📉 Learning Rate Scheduler:")
        print(f"  Type: ReduceLROnPlateau")
        print(f"  Patience: {config.scheduler_patience} epochs")
        print(f"  Factor: {config.scheduler_factor}")
        print(f"  Min LR: {config.scheduler_min_lr}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Loss function: MSE")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    if config.grad_clip:
        print(f"  Gradient clipping: {config.grad_clip}")
    
    # Log hyperparameters and model architecture to TensorBoard
    print("\n📊 Logging hyperparameters and model architecture to TensorBoard...")
    
    # Create hyperparameters text
    hparams_text = f"""
# Behavior Cloning Training Configuration

## Dataset Settings
- Dataset path: {config.dataset_path}
- Target action key: {config.target_action_key}
- Total samples: {len(full_dataset)}
- Train samples: {len(train_dataset)}
- Validation samples: {len(val_dataset)}
- Test samples: {len(test_dataset)}
- Input crop size: {config.input_crop_size}
- Log-scale preprocessing: {config.log_scale}

## Model Architecture
- Architecture: ResNet-18 Actor
- Input channels: {input_channels}
- Action dimension: {action_dim}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Pretrained encoder: {config.pretrained_encoder_path if config.pretrained_encoder_path else 'None'}
- Freeze encoder: {config.freeze_encoder}

## Training Settings
- Optimizer: {config.optimizer}
- Learning rate: {config.learning_rate}
- Weight decay: {config.weight_decay}
- Batch size: {config.batch_size}
- Number of epochs: {config.num_epochs}
- Gradient clipping: {config.grad_clip if config.grad_clip else 'None'}
- Loss function: {config.loss_function}
{f"- Huber delta: {config.huber_delta}" if config.loss_function == "huber" else ""}
{f"- Adaptive scale: {config.adaptive_scale}" if config.loss_function == "adaptive_mse" else ""}
- LR Scheduler: {config.use_scheduler}
{f"- Scheduler patience: {config.scheduler_patience}" if config.use_scheduler else ""}
{f"- Scheduler factor: {config.scheduler_factor}" if config.use_scheduler else ""}
{f"- Scheduler min LR: {config.scheduler_min_lr}" if config.use_scheduler else ""}
- Device: {device}
- GPU count: {gpu_count}
- DataParallel: {gpu_count > 1 and not config.no_dataparallel}

## Data Loading
- Number of workers: {config.num_workers}
- Pin memory: {config.pin_memory}
- Train split: {config.train_split}
- Validation split: {config.val_split}
- Test split: {config.test_split}

## Rollout Settings
- Enable rollouts: {config.enable_rollouts}
- Rollout interval: every {config.rollout_interval} epoch(s)
- Rollout seeds: {config.rollout_seeds}
- Rollout steps: {config.rollout_steps}
- Rollout episode steps: {config.rollout_episode_steps}
- Force incremental mode: {config.force_incremental_mode}

## Other Settings
- Random seed: {config.seed}
- Run name: {config.run_name if config.run_name else 'None'}
"""
    
    writer.add_text("Configuration/Hyperparameters", hparams_text, 0)
    
    # Log model architecture if available
    if model_summary_str:
        writer.add_text("Configuration/Model_Architecture", f"```\n{model_summary_str}\n```", 0)
    
    print("✅ Hyperparameters and architecture logged to TensorBoard")
    
    # Training loop
    print("\n🎯 Starting Training...")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epoch_times = []
    
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train with detailed metrics
        train_loss, train_mae_metrics, train_error_distribution, train_abs_error_metrics, train_l2_norms, train_mae_per_sample = train_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )
        
        # Validate with detailed metrics
        val_loss, val_mae_metrics, val_error_distribution, val_abs_error_metrics, val_l2_norms, val_mae_per_sample = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # TensorBoard logging - Basic losses
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Training Instrumentation - Example-level MAE
        writer.add_scalar('Training_Instrumentation/MAE_Mean', train_mae_metrics['mae_mean'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Median', train_mae_metrics['mae_median'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Min', train_mae_metrics['mae_min'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Max', train_mae_metrics['mae_max'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Std', train_mae_metrics['mae_std'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Q25', train_mae_metrics['mae_q25'], epoch)
        writer.add_scalar('Training_Instrumentation/MAE_Q75', train_mae_metrics['mae_q75'], epoch)
        writer.add_histogram('Training_Instrumentation/Error_Distribution', train_error_distribution, epoch)
        
        # Training Instrumentation - Action-level absolute error statistics
        writer.add_scalar('Training_Instrumentation/AbsError_Mean', train_abs_error_metrics['abs_error_mean'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Median', train_abs_error_metrics['abs_error_median'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Min', train_abs_error_metrics['abs_error_min'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Max', train_abs_error_metrics['abs_error_max'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Std', train_abs_error_metrics['abs_error_std'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Q25', train_abs_error_metrics['abs_error_q25'], epoch)
        writer.add_scalar('Training_Instrumentation/AbsError_Q75', train_abs_error_metrics['abs_error_q75'], epoch)
        
        # MAE vs Target L2 Candlestick Plots
        train_candlestick_fig = create_mae_vs_l2_candlestick(
            train_l2_norms, train_mae_per_sample, 
            f'Training MAE vs Target Action L2 Norm (Epoch {epoch})'
        )
        writer.add_figure('Training_Instrumentation/MAE_vs_L2_Candlestick', train_candlestick_fig, epoch)
        plt.close(train_candlestick_fig)
        
        # Validation Instrumentation - Example-level MAE
        writer.add_scalar('Validation_Instrumentation/MAE_Mean', val_mae_metrics['mae_mean'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Median', val_mae_metrics['mae_median'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Min', val_mae_metrics['mae_min'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Max', val_mae_metrics['mae_max'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Std', val_mae_metrics['mae_std'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Q25', val_mae_metrics['mae_q25'], epoch)
        writer.add_scalar('Validation_Instrumentation/MAE_Q75', val_mae_metrics['mae_q75'], epoch)
        writer.add_histogram('Validation_Instrumentation/Error_Distribution', val_error_distribution, epoch)
        
        # MAE vs Target L2 Candlestick Plots
        val_candlestick_fig = create_mae_vs_l2_candlestick(
            val_l2_norms, val_mae_per_sample,
            f'Validation MAE vs Target Action L2 Norm (Epoch {epoch})'
        )
        writer.add_figure('Validation_Instrumentation/MAE_vs_L2_Candlestick', val_candlestick_fig, epoch)
        plt.close(val_candlestick_fig)
        
        # Validation Instrumentation - Action-level absolute error statistics
        writer.add_scalar('Validation_Instrumentation/AbsError_Mean', val_abs_error_metrics['abs_error_mean'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Median', val_abs_error_metrics['abs_error_median'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Min', val_abs_error_metrics['abs_error_min'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Max', val_abs_error_metrics['abs_error_max'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Std', val_abs_error_metrics['abs_error_std'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Q25', val_abs_error_metrics['abs_error_q25'], epoch)
        writer.add_scalar('Validation_Instrumentation/AbsError_Q75', val_abs_error_metrics['abs_error_q75'], epoch)
        
        # Calculate timing and ETA
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
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
        writer.add_scalar('Timing/Epoch_Time', epoch_time, epoch)
        writer.add_scalar('Timing/Average_Epoch_Time', avg_epoch_time, epoch)
        writer.add_scalar('Timing/ETA_Hours', eta_hours, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate/Current', current_lr, epoch)
        
        # Print progress with detailed metrics
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta_str}")
        print(f"  Train MAE Mean:   {train_mae_metrics['mae_mean']:.6f} | "
              f"Median: {train_mae_metrics['mae_median']:.6f} | "
              f"Range: [{train_mae_metrics['mae_min']:.6f}, {train_mae_metrics['mae_max']:.6f}]")
        print(f"  Val MAE Mean:     {val_mae_metrics['mae_mean']:.6f} | "
              f"Median: {val_mae_metrics['mae_median']:.6f} | "
              f"Range: [{val_mae_metrics['mae_min']:.6f}, {val_mae_metrics['mae_max']:.6f}]")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎉 New best validation loss: {best_val_loss:.6f}")
            if config.save_model:
                checkpoint_path = log_dir / "bc_model_best.pth"
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss,
                              config, str(checkpoint_path), is_best=True)
        
        # Perform rollout instrumentation after validation (check interval)
        should_rollout = (epoch + 1) % config.rollout_interval == 0
        if config.enable_rollouts and should_rollout and perform_rollout_instrumentation is not None:
            print(f"  🎯 Running rollout instrumentation (interval: every {config.rollout_interval} epoch(s))...")
            try:
                # Get the model to use for rollouts (unwrap DataParallel if needed)
                rollout_model = model.module if (gpu_count > 1 and not config.no_dataparallel) else model
                
                # Use a single temporary model file that gets overwritten each epoch
                temp_model_path = log_dir / "temp_model_for_rollout.pth"
                torch.save({
                    'model_state_dict': rollout_model.state_dict(),
                    'epoch': epoch + 1,
                    'config': vars(config) if hasattr(config, '__dict__') else config
                }, temp_model_path)
                
                # Use the rollout environment config that was saved at the start of training
                env_config_path = log_dir / "rollout_env_config.json"
                
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
                
            except Exception as e:
                print(f"  ❌ Rollout failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save training curves periodically
        if (epoch + 1) % 10 == 0 or is_best:
            plot_save_path = log_dir / "training_curves.png"
            plot_training_curves(train_losses, val_losses, str(plot_save_path))
    
    total_training_time = time.time() - training_start_time
    
    # Test final model with detailed metrics
    print(f"\n🧪 Testing final model...")
    test_loss, test_mae_metrics, test_error_distribution, test_abs_error_metrics = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE Mean: {test_mae_metrics['mae_mean']:.6f}")
    print(f"Test MAE Median: {test_mae_metrics['mae_median']:.6f}")
    print(f"Test MAE Range: [{test_mae_metrics['mae_min']:.6f}, {test_mae_metrics['mae_max']:.6f}]")
    print(f"Test MAE Q25-Q75: [{test_mae_metrics['mae_q25']:.6f}, {test_mae_metrics['mae_q75']:.6f}]")
    print(f"Test AbsError Mean: {test_abs_error_metrics['abs_error_mean']:.6f}")
    print(f"Test AbsError Median: {test_abs_error_metrics['abs_error_median']:.6f}")
    
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
        print(f"  Test MAE mean:      {test_mae_metrics['mae_mean']:.6f}")
        print(f"  Test MAE median:    {test_mae_metrics['mae_median']:.6f}")
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
            'test_mae_metrics': test_mae_metrics,
            'test_abs_error_metrics': test_abs_error_metrics,
            'config': vars(config) if hasattr(config, '__dict__') else config,
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
    writer.add_scalar('Final/Test_MAE_Mean', test_mae_metrics['mae_mean'], config.num_epochs)
    writer.add_scalar('Final/Test_MAE_Median', test_mae_metrics['mae_median'], config.num_epochs)
    writer.add_scalar('Final/Test_MAE_Q25', test_mae_metrics['mae_q25'], config.num_epochs)
    writer.add_scalar('Final/Test_MAE_Q75', test_mae_metrics['mae_q75'], config.num_epochs)
    writer.add_scalar('Final/Test_AbsError_Mean', test_abs_error_metrics['abs_error_mean'], config.num_epochs)
    writer.add_scalar('Final/Test_AbsError_Median', test_abs_error_metrics['abs_error_median'], config.num_epochs)
    writer.add_histogram('Final/Test_Error_Distribution', test_error_distribution, config.num_epochs)
    
    writer.close()
    
    # Clean up temporary rollout model if it exists
    temp_model_path = log_dir / "temp_model_for_rollout.pth"
    if temp_model_path.exists():
        temp_model_path.unlink()
        print(f"🧹 Cleaned up temporary rollout model")
    
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
    parser.add_argument("--loss-function", type=str, default="mse",
                       choices=['mse', 'huber', 'log_cosh', 'adaptive_mse'],
                       help="Loss function (huber, log_cosh, adaptive_mse emphasize small values)")
    parser.add_argument("--huber-delta", type=float, default=0.01,
                       help="Delta threshold for Huber loss (default: 0.01)")
    parser.add_argument("--adaptive-scale", type=float, default=0.01,
                       help="Scale threshold for adaptive MSE loss (default: 0.01)")
    
    # Learning rate scheduler settings
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                       help="Use learning rate scheduler (ReduceLROnPlateau)")
    parser.add_argument("--no-scheduler", action="store_false", dest="use_scheduler",
                       help="Disable learning rate scheduler")
    parser.add_argument("--scheduler-patience", type=int, default=50,
                       help="Epochs to wait before reducing LR (default: 10)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                       help="Factor to reduce LR by (default: 0.5)")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-7,
                       help="Minimum learning rate (default: 1e-7)")
    
    # Device settings
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for training (auto, cuda, cuda:0, cuda:1, mps, cpu)")
    parser.add_argument("--no-dataparallel", action="store_true",
                       help="Disable DataParallel for multi-GPU training")
    
    # Data loading settings
    parser.add_argument("--lazy", action="store_true", default=False,
                       help="Use lazy (on-demand) data loading instead of preloading into memory")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Output settings
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save model")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Optional name to append to run directory")
    
    # Rollout settings
    parser.add_argument("--enable-rollouts", action="store_true", default=True,
                       help="Enable environment rollouts after validation")
    parser.add_argument("--no-rollouts", action="store_false", dest="enable_rollouts",
                       help="Disable environment rollouts")
    parser.add_argument("--rollout-interval", type=int, default=1,
                       help="Perform rollouts every N epochs (default: 1, every epoch)")
    parser.add_argument("--rollout-seeds", type=int, default=8,
                       help="Number of rollout seeds")
    parser.add_argument("--rollout-steps", type=int, default=250,
                       help="Number of steps per rollout")
    parser.add_argument("--rollout-episode-steps", type=int, default=100,
                       help="Maximum episode length for rollouts (overrides dataset config)")
    parser.add_argument("--force-incremental-mode", action="store_true", default=True,
                       help="Force incremental control mode for rollouts")
    parser.add_argument("--no-force-incremental-mode", action="store_false", dest="force_incremental_mode",
                       help="Don't force incremental control mode (use absolute actions)")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
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
        loss_function=args.loss_function,
        huber_delta=args.huber_delta,
        adaptive_scale=args.adaptive_scale,
        use_scheduler=args.use_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_min_lr=args.scheduler_min_lr,
        num_workers=args.num_workers,
        no_dataparallel=args.no_dataparallel,
        lazy_loading=args.lazy,
        enable_rollouts=args.enable_rollouts,
        rollout_interval=args.rollout_interval,
        rollout_seeds=args.rollout_seeds,
        rollout_steps=args.rollout_steps,
        rollout_episode_steps=args.rollout_episode_steps,
        force_incremental_mode=args.force_incremental_mode,
        pretrained_encoder_path=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        run_name=args.run_name
    )
    
    # Print configuration
    print("🔧 Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_behavior_cloning(config)


if __name__ == "__main__":
    main()
