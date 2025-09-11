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
    plot_per_epoch: bool = True  # Plot after each epoch
    plots_dir: str = "saved_models/training_plots"  # Directory for plots
    seed: int = 42
    resume_from: str = None  # Path to checkpoint to resume from
    start_epoch: int = 0  # Starting epoch (for resumed training)
    max_examples: int = None  # Limit dataset size for testing
    # Rollout evaluation settings
    enable_rollouts: bool = False  # Whether to perform rollouts during training
    rollouts_per_improvement: int = 5  # Number of rollouts to perform when validation improves
    rollout_results_dir: str = "saved_models/rollout_results"  # Directory for rollout results
    render_rollouts: bool = False  # Whether to render rollout results
    render_epoch_interval: int = 20  # Interval of epochs at which to perform rendering (0 = every epoch)
    render_step_interval: int = 1  # Step interval for rendering (passed to render_history.py)
    render_dpi: int = 100  # DPI for rendered images (lower = faster rendering)


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
    Load dataset with sequential processing (most reliable)
    
    Args:
        dataset_path: Path to dataset directory
        use_cache: Whether to use cached data if available
        max_examples: Maximum number of examples to load (stops early if specified)
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    dataset_path = Path(dataset_path)
    cache_file = dataset_path / "processed_pairs_cache.pkl"
    
    print(f"Loading dataset from {dataset_path}...")
    if max_examples is not None:
        print(f"🎯 Target: {max_examples} examples (will stop early)")
    
    # Find all completed episode files
    all_files = list(dataset_path.glob("*.json"))
    episode_files = [f for f in all_files 
                    if f.name.startswith('episode_') and not f.name.startswith('.tmp_')]
    
    print(f"Found {len(episode_files)} completed episode files")
    
    if len(episode_files) == 0:
        print("❌ No episode files found!")
        return []
    
    # Check cache only if we're loading the full dataset
    if use_cache and cache_file.exists() and max_examples is None:
        try:
            cache_mtime = cache_file.stat().st_mtime
            latest_file_mtime = max(f.stat().st_mtime for f in episode_files) if episode_files else 0
            
            if cache_mtime > latest_file_mtime:
                print("📦 Loading from cache...")
                start_time = time.time()
                with open(cache_file, 'rb') as f:
                    pairs = pickle.load(f)
                load_time = time.time() - start_time
                print(f"✅ Loaded {len(pairs)} pairs from cache in {load_time:.1f}s")
                return pairs
        except Exception as e:
            print(f"Cache error: {e}, reprocessing...")
    
    # Sequential loading
    pairs = []
    start_time = time.time()
    
    print("🔄 Loading files sequentially...")
    for episode_file in tqdm(episode_files, desc="Loading episodes"):
        episode_pairs = load_single_episode(episode_file)
        pairs.extend(episode_pairs)
        
        # Stop early if we've reached the target
        if max_examples is not None and len(pairs) >= max_examples:
            print(f"🎯 Reached target of {max_examples} examples, stopping early...")
            pairs = pairs[:max_examples]  # Trim to exact count
            break
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(pairs)} pairs in {load_time:.1f}s")
    if load_time > 0:
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


def load_dataset_pairs(dataset_path: str, max_examples: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all observation-action pairs from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        max_examples: Maximum number of examples to load (for testing)
        
    Returns:
        List of (observation, perfect_action) tuples
    """
    # Use the sequential loading (simple and reliable)
    pairs = load_dataset_pairs_sequential(dataset_path, use_cache=True, max_examples=max_examples)
    
    # If we used max_examples, we already have the right amount
    if max_examples is not None:
        print(f"✅ Loaded exactly {len(pairs)} examples as requested")
        return pairs
    
    # Only do random sampling if we loaded the full dataset and want to limit it
    # (This path shouldn't be reached anymore, but keeping for safety)
    if max_examples is not None and len(pairs) > max_examples:
        print(f"📊 Limiting dataset to {max_examples} examples (from {len(pairs)})")
        # Use random sampling to get diverse examples
        import random
        random.seed(42)  # For reproducible sampling
        pairs = random.sample(pairs, max_examples)
        print(f"✅ Dataset limited to {len(pairs)} examples")
    
    return pairs


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
    
    plt.close()  # Close figure to free memory


def plot_epoch_times(epoch_times: List[float], save_path: str = None):
    """Plot epoch timing information"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(epoch_times) + 1)
    
    plt.plot(epochs, epoch_times, 'g-', marker='o', linewidth=2, markersize=4)
    
    plt.title('Training Time per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add average line
    if len(epoch_times) > 1:
        avg_time = np.mean(epoch_times)
        plt.axhline(y=avg_time, color='r', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_time:.1f}s')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Epoch times plot saved to {save_path}")
    
    plt.close()  # Close figure to free memory


def update_training_plots(train_losses: List[float], val_losses: List[float], 
                         epoch_times: List[float], plots_dir: str):
    """Update training plots after each epoch"""
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training curves
    loss_plot_path = os.path.join(plots_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, loss_plot_path)
    
    # Plot epoch times
    if len(epoch_times) > 0:
        time_plot_path = os.path.join(plots_dir, "epoch_times.png")
        plot_epoch_times(epoch_times, time_plot_path)


def load_environment_flags(dataset_path: str) -> List[str]:
    """
    Load environment flags from the job config file in the dataset directory
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of environment flags for rollout evaluation
    """
    dataset_path = Path(dataset_path)
    
    # Look for job config file
    job_config_files = list(dataset_path.glob("*_job_config.json"))
    
    if not job_config_files:
        print(f"⚠️  No job config file found in {dataset_path}")
        return []
    
    job_config_file = job_config_files[0]
    print(f"📋 Loading environment flags from {job_config_file.name}")
    
    try:
        with open(job_config_file, 'r') as f:
            config = json.load(f)
        
        environment_flags = config.get('environment_flags', [])
        print(f"✅ Loaded {len(environment_flags)} environment flags")
        return environment_flags
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Failed to load environment flags: {e}")
        return []


def perform_rollout_evaluation(model_path: str, environment_flags: List[str], 
                              num_rollouts: int, results_dir: str, epoch: int, 
                              render: bool = False, render_step_interval: int = 1, render_dpi: int = 100) -> Dict:
    """
    Perform rollout evaluation of the trained model using the universal rollout system
    
    Args:
        model_path: Path to the saved model file
        environment_flags: List of environment configuration flags
        num_rollouts: Number of rollouts to perform
        results_dir: Directory to save rollout results
        epoch: Current training epoch (for naming/tracking)
        render: Whether to render rollout results after completion
        render_step_interval: Step interval for rendering (passed to render_history.py)
        render_dpi: DPI for rendered images (lower = faster rendering)
        
    Returns:
        rollout_results: Dictionary containing evaluation metrics
    """
    print(f"\n🎮 Starting rollout evaluation (epoch {epoch+1})")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Rollouts: {num_rollouts}")
    print(f"   Environment flags: {len(environment_flags)} flags")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create state info directory for this rollout
    state_info_dir = os.path.join(results_dir, f"state_info_epoch_{epoch}")
    os.makedirs(state_info_dir, exist_ok=True)
    
    # Import the universal rollout system and new environment config
    sys.path.insert(0, str(Path(__file__).parent.parent))  # Add optomech to path
    from optomech_rollout import UniversalRolloutEngine, SMLModelInterface
    from optomech.env_config import merge_config_with_flags
    
    # Create environment arguments using the same approach as standalone rollout
    # Import configuration utilities
    from optomech_rollout import load_environment_config
    
    # Create a temporary job config file path for the config loader
    dataset_path = Path(model_path).parent.parent / "datasets" / "sml_100k_dataset"
    job_config_files = list(dataset_path.glob("*_job_config.json"))
    
    if job_config_files:
        job_config_path = job_config_files[0]
        print(f"📋 Using environment config from: {job_config_path}")
        
        # Use the same environment configuration system as standalone rollout
        env_args = load_environment_config(str(job_config_path))
        
        # Override specific values for rollout evaluation
        env_args.max_episode_steps = 250
        env_args.save_episodes = True
        env_args.render = render
        env_args.render_interval = render_step_interval  # Set step interval for rendering
        env_args.render_dpi = render_dpi  # Set DPI for rendering
        
        # Only enable state writing/recording if we're rendering (expensive disk I/O)
        env_args.record_env_state_info = render
        env_args.write_env_state_info = render
        
        print(f"✅ Environment configuration loaded successfully")
        # DEBUG: print(f"State recording: {getattr(env_args, 'record_env_state_info', 'unknown')}")
        # DEBUG: print(f"State writing: {getattr(env_args, 'write_env_state_info', 'unknown')}")
    else:
        print("❌ No job config file found - using fallback configuration")
        # Fallback to merge_config_with_flags approach
        env_args = merge_config_with_flags(
            config=None,  # Use default config
            flags_list=environment_flags,
            # Override specific values for rollout evaluation
            max_episode_steps=250,
            save_episodes=True,  # Enable episode saving for render_history compatibility
            record_env_state_info=render,  # Only enable state recording if rendering (expensive disk I/O)
            write_env_state_info=render,   # Only enable state writing if rendering (expensive disk I/O)
            state_info_save_dir=state_info_dir,  # Use epoch-specific directory
            render=render,  # Enable rendering if requested
            render_interval=render_step_interval,  # Set step interval for rendering
            render_dpi=render_dpi,  # Set DPI for rendering
        )
    
    print(f"📋 Environment configuration created")
    print(f"📁 State info will be saved to: {state_info_dir}")
    print(f"🎨 Rendering enabled: {getattr(env_args, 'render', False)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Create model interface
    model_interface = SMLModelInterface(
        model_path=model_path,
        device=device
    )
    model_interface.load_model()
    
    # Create rollout engine
    engine = UniversalRolloutEngine(
        model_interface=model_interface,
        env_args=env_args,
        device=device
    )
    
    # Generate unique seed for this rollout evaluation
    rollout_seed = random.randint(0, 2**16) + epoch * 1000  # Unique seed per epoch
    
    # Run rollouts with randomized seed
    episode_save_path = os.path.join(results_dir, f"epoch_{epoch+1:03d}")
    returns, step_wise_rewards = engine.run_rollout(
        num_episodes=num_rollouts,
        exploration_noise=0.0,  # No exploration noise for SML evaluation
        save_path=episode_save_path,
        save_episodes=True,  # Save episodes for render_history compatibility
        random_policy=False
    )
    
    # Calculate statistics
    rollout_results = {
        'epoch': epoch,
        'num_rollouts': num_rollouts,
        'cumulative_rewards': returns,
        'step_wise_rewards': step_wise_rewards,
        'episode_lengths': [250] * len(returns),  # Placeholder - would need to get from engine
        'success_rate': np.mean([r > 0 for r in returns]) if returns else 0.0,
        'mean_reward': np.mean(returns) if returns else 0.0,
        'std_reward': np.std(returns) if returns else 0.0,
        'model_path': model_path,
        'environment_flags': environment_flags,
        'timestamp': time.time(),
        'save_path': episode_save_path
    }
    
    print(f"\n✅ Rollout evaluation complete!")
    print(f"   Mean reward: {rollout_results['mean_reward']:.3f} ± {rollout_results['std_reward']:.3f}")
    print(f"   Success rate: {rollout_results['success_rate']:.1%}")
    print(f"   Episodes saved to: {os.path.basename(episode_save_path)}")
    
    # Create stepwise analysis plot for this epoch's rollouts
    if step_wise_rewards:
        plots_dir = os.path.join(os.path.dirname(results_dir), "training_plots")
        create_stepwise_analysis(step_wise_rewards, epoch, plots_dir)
    
    # Save results to JSON file as before
    results_file = os.path.join(results_dir, f"rollout_epoch_{epoch+1:03d}.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {k: (v.item() if hasattr(v, 'item') else v) 
                       for k, v in rollout_results.items()}
        json.dump(json_results, f, indent=2)
    print(f"   Results saved: {os.path.basename(results_file)}")
    
    return rollout_results


def create_stepwise_analysis(step_wise_rewards: List[List[float]], epoch: int, plots_dir: str):
    """
    Create stepwise reward analysis for a single epoch's rollouts
    
    Args:
        step_wise_rewards: List of episodes, each containing step-by-step rewards
        epoch: Current epoch number (for plot title and filename)
        plots_dir: Directory to save the plot
    """
    if not step_wise_rewards:
        print("⚠️  No stepwise data available for analysis")
        return
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert to numpy array for easier analysis
    # Find the maximum episode length
    max_length = max(len(episode) for episode in step_wise_rewards)
    
    # Pad shorter episodes with NaN for proper averaging
    padded_rewards = []
    for episode in step_wise_rewards:
        padded = list(episode) + [np.nan] * (max_length - len(episode))
        padded_rewards.append(padded)
    
    rewards_array = np.array(padded_rewards)  # Shape: (num_episodes, max_steps)
    
    # Calculate statistics across episodes for each step
    step_means = np.nanmean(rewards_array, axis=0)
    step_stds = np.nanstd(rewards_array, axis=0)
    step_numbers = np.arange(len(step_means))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual episodes (lighter lines)
    for i, episode_rewards in enumerate(step_wise_rewards):
        episode_steps = np.arange(len(episode_rewards))
        plt.plot(episode_steps, episode_rewards, 'lightblue', alpha=0.3, linewidth=1)
    
    # Plot mean with error bars
    plt.errorbar(step_numbers, step_means, yerr=step_stds, 
                color='darkblue', linewidth=2, capsize=3, 
                label=f'Mean ± Std (n={len(step_wise_rewards)} episodes)')
    
    # Plot mean line only (for clarity)
    plt.plot(step_numbers, step_means, 'red', linewidth=2, label='Mean reward per step')
    
    plt.title(f'Stepwise Reward Analysis - Epoch {epoch + 1}', fontsize=16)
    plt.xlabel('Step Number', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text with summary statistics
    total_episodes = len(step_wise_rewards)
    mean_episode_length = np.mean([len(ep) for ep in step_wise_rewards])
    plt.text(0.02, 0.98, f'Episodes: {total_episodes}\nMean length: {mean_episode_length:.1f} steps', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, f"stepwise_analysis_epoch_{epoch+1:03d}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Stepwise analysis plot saved: {os.path.basename(plot_path)}")


def update_rollout_plots(all_rollout_results: List[Dict], plots_dir: str):
    """
    Update rollout evaluation plots
    
    Args:
        all_rollout_results: List of rollout results from all epochs
        plots_dir: Directory to save plots
    """
    if not all_rollout_results:
        return
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for plotting
    epochs = [r['epoch'] + 1 for r in all_rollout_results]
    mean_rewards = [r['mean_reward'] for r in all_rollout_results]
    std_rewards = [r['std_reward'] for r in all_rollout_results]
    success_rates = [r['success_rate'] for r in all_rollout_results]
    
    # Plot mean rewards over epochs
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Mean rewards with error bars
    plt.subplot(1, 2, 1)
    plt.errorbar(epochs, mean_rewards, yerr=std_rewards, marker='o', linewidth=2, capsize=5)
    plt.title('Rollout Mean Rewards', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Cumulative Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Success rates
    plt.subplot(1, 2, 2)
    plt.plot(epochs, success_rates, 'g-o', linewidth=2, markersize=6)
    plt.title('Rollout Success Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, "rollout_evaluation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Rollout evaluation plot saved to {plot_path}")
    plt.close()


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
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to use (for testing)")
    parser.add_argument("--enable_rollouts", action="store_true",
                       help="Enable rollout evaluation during training")
    parser.add_argument("--rollouts_per_improvement", type=int, default=5,
                       help="Number of rollouts to perform when validation improves")
    parser.add_argument("--render_rollouts", action="store_true",
                       help="Enable rendering of rollout results")
    parser.add_argument("--render_epoch_interval", type=int, default=20,
                       help="Interval of epochs at which to perform rendering (0 = every epoch)")
    parser.add_argument("--render_step_interval", type=int, default=1,
                       help="Step interval for rendering (passed to render_history.py)")
    parser.add_argument("--render_dpi", type=int, default=100,
                       help="DPI for rendered images (lower = faster rendering)")
    
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
        resume_from=args.resume_from,
        max_examples=args.max_examples,
        enable_rollouts=args.enable_rollouts,
        rollouts_per_improvement=args.rollouts_per_improvement,
        render_rollouts=args.render_rollouts,
        render_epoch_interval=args.render_epoch_interval,
        render_step_interval=args.render_step_interval,
        render_dpi=args.render_dpi
    )
    
    print("🚀 Starting SML Model Training")
    print("=" * 50)
    
    # Load and split dataset
    print("� Loading dataset...")
    pairs = load_dataset_pairs(config.dataset_path, config.max_examples)
    
    if len(pairs) == 0:
        print("❌ No valid observation-action pairs found!")
        return
    
    # Check action dimensions
    sample_action = pairs[0][1]
    action_dim = len(sample_action) if sample_action.size > 0 else 15
    print(f"Action dimension: {action_dim}")
    
    # Check observation shape
    sample_obs = pairs[0][0]
    # DEBUG: print(f"Observation shape: {sample_obs.shape}")
    
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, config.train_split, config.val_split, config.test_split, config.seed
    )
    
    # Load environment flags for rollout evaluation
    environment_flags = []
    all_rollout_results = []
    if config.enable_rollouts:
        print("\n🎮 Rollout evaluation enabled")
        environment_flags = load_environment_flags(config.dataset_path)
        print(f"  Rollouts per improvement: {config.rollouts_per_improvement}")
        print(f"  Results directory: {config.rollout_results_dir}")
    
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
                
                # Perform rollout evaluation if enabled
                if config.enable_rollouts:
                    # Determine if we should render this epoch based on the interval
                    should_render = False
                    if config.render_rollouts:
                        if config.render_epoch_interval == 0:
                            should_render = True  # Render every epoch
                        else:
                            should_render = (epoch + 1) % config.render_epoch_interval == 0
                    
                    rollout_results = perform_rollout_evaluation(
                        model_path=config.model_save_path,
                        environment_flags=environment_flags,
                        num_rollouts=config.rollouts_per_improvement,
                        results_dir=config.rollout_results_dir,
                        epoch=epoch,
                        render=should_render,
                        render_step_interval=config.render_step_interval,
                        render_dpi=config.render_dpi
                    )
                    all_rollout_results.append(rollout_results)
                    
                    # Update rollout plots
                    update_rollout_plots(all_rollout_results, config.plots_dir)
        
        # Update plots after each epoch if enabled
        if config.plot_per_epoch:
            update_training_plots(train_losses, val_losses, epoch_times, config.plots_dir)
    
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
    
    # Rollout evaluation summary
    if config.enable_rollouts and all_rollout_results:
        print(f"\n🎮 Rollout Evaluation Summary:")
        print(f"  Total rollout evaluations: {len(all_rollout_results)}")
        final_rollout = all_rollout_results[-1]
        print(f"  Final mean reward: {final_rollout['mean_reward']:.3f} ± {final_rollout['std_reward']:.3f}")
        print(f"  Final success rate: {final_rollout['success_rate']:.1%}")
        
        # Best rollout performance
        best_rollout = max(all_rollout_results, key=lambda x: x['mean_reward'])
        print(f"  Best mean reward: {best_rollout['mean_reward']:.3f} (epoch {best_rollout['epoch']+1})")
        
        print(f"  Rollout results saved in: {config.rollout_results_dir}")
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
