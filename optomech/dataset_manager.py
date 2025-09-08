"""
Simple, thread-safe dataset manager for RL episodes.
Allows multiple processes to write independently to a common store.
"""

import json
import time
import uuid
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import fcntl  # For file locking on Unix systems


class EpisodeDataset(Dataset):
    """PyTorch Dataset for loading RL episodes."""
    
    def __init__(self, dataset_dir: str, transform=None, max_episodes=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.max_episodes = max_episodes
        self.episodes = self._load_episode_list()
        
    def _load_episode_list(self) -> List[Dict]:
        """Load all available episodes."""
        episodes = []
        episode_files = list(self.dataset_dir.glob("episode_*.json"))
        
        # Limit files to scan if max_episodes is set
        if self.max_episodes:
            episode_files = sorted(episode_files)[:self.max_episodes]
            print(f"Limiting scan to first {self.max_episodes} episode files for faster loading")
        
        print(f"Found {len(episode_files)} episode files in {self.dataset_dir}")
        print("Scanning episode files for metadata (this may take a while for large files)...")
        
        import time
        start_time = time.time()
        
        for i, episode_file in enumerate(sorted(episode_files)):
            file_start = time.time()
            file_size_mb = episode_file.stat().st_size / (1024 * 1024)
            
            print(f"  [{i+1:3d}/{len(episode_files)}] Loading {episode_file.name} ({file_size_mb:.1f}MB)...", end="", flush=True)
            
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                    episodes.append({
                        'file_path': episode_file,
                        'metadata': episode_data.get('metadata', {}),
                        'length': len(episode_data.get('observations', []))
                    })
                
                file_time = time.time() - file_start
                print(f" ✓ ({file_time:.1f}s)")
                
                # Time estimation
                if i >= 2:  # After loading a few files, estimate remaining time
                    avg_time_per_file = (time.time() - start_time) / (i + 1)
                    remaining_files = len(episode_files) - (i + 1)
                    est_remaining_time = avg_time_per_file * remaining_files
                    print(f"      Estimated time remaining: {est_remaining_time/60:.1f} minutes")
                    
            except (json.JSONDecodeError, KeyError) as e:
                file_time = time.time() - file_start
                print(f" ✗ CORRUPTED ({file_time:.1f}s) - {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\nLoaded {len(episodes)} valid episodes in {total_time/60:.1f} minutes")
        return episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        """Load and return an episode."""
        episode_info = self.episodes[idx]
        
        import time
        start_time = time.time()
        file_size_mb = episode_info['file_path'].stat().st_size / (1024 * 1024)
        
        print(f"Loading episode {idx+1}/{len(self.episodes)}: {episode_info['file_path'].name} ({file_size_mb:.1f}MB)...", end="", flush=True)
        
        with open(episode_info['file_path'], 'r') as f:
            episode_data = json.load(f)
        
        load_time = time.time() - start_time
        print(f" ✓ ({load_time:.1f}s)")
        
        # Convert to numpy arrays
        print(f"  Converting to tensors...", end="", flush=True)
        convert_start = time.time()
        
        observations = np.array(episode_data['observations'])
        next_observations = np.array(episode_data.get('next_observations', episode_data['observations']))
        actions = np.array(episode_data['actions'])
        rewards = np.array(episode_data['rewards'])
        dones = np.array(episode_data['dones'])
        
        # Convert to torch tensors
        data = {
            'observations': torch.from_numpy(observations).float(),
            'next_observations': torch.from_numpy(next_observations).float(),
            'actions': torch.from_numpy(actions).float(),
            'rewards': torch.from_numpy(rewards).float(),
            'dones': torch.from_numpy(dones).bool(),
            'metadata': episode_info['metadata']
        }
        
        # Add perfect_actions and best_actions if available
        if 'perfect_actions' in episode_data:
            perfect_actions = np.array(episode_data['perfect_actions'])
            data['perfect_actions'] = torch.from_numpy(perfect_actions).float()
        
        if 'best_actions' in episode_data:
            best_actions = np.array(episode_data['best_actions'])
            data['best_actions'] = torch.from_numpy(best_actions).float()
        
        convert_time = time.time() - convert_start
        total_time = time.time() - start_time
        print(f" ✓ ({convert_time:.1f}s) | Total: {total_time:.1f}s")
        
        if self.transform:
            data = self.transform(data)
            
        return data


class DatasetManager:
    """Simple, process-safe dataset manager."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = self.dataset_dir / "dataset_stats.json"
        
    def save_episode(self, episode_data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        """
        Save an episode to the dataset. Returns the episode ID.
        Thread-safe: multiple processes can call this simultaneously.
        """
        # Generate unique episode ID
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Prepare episode data with metadata
        full_episode_data = {
            'episode_id': episode_id,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'observations': episode_data['observations'],
            'next_observations': episode_data.get('next_observations', episode_data['observations']),
            'actions': episode_data['actions'],
            'rewards': episode_data['rewards'],
            'dones': episode_data['dones']
        }
        
        # Add perfect_actions and best_actions if provided
        if 'perfect_actions' in episode_data:
            full_episode_data['perfect_actions'] = episode_data['perfect_actions']
        
        if 'best_actions' in episode_data:
            full_episode_data['best_actions'] = episode_data['best_actions']
        
        # Add computed metrics
        full_episode_data['metrics'] = {
            'total_reward': float(np.sum(episode_data['rewards'])),
            'episode_length': len(episode_data['observations']),
            'mean_reward': float(np.mean(episode_data['rewards'])),
            'success': bool(episode_data.get('success', False))
        }
        
        # Save episode file atomically to prevent corruption during concurrent access
        episode_file = self.dataset_dir / f"episode_{episode_id}.json"
        
        print(f"Saving episode {episode_id[:8]}...", end="")
        
        # Use atomic file write: write to temp file then rename
        # This prevents corruption if another process reads while writing
        temp_file = self.dataset_dir / f".tmp_episode_{episode_id}.json"
        
        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(full_episode_data, f, indent=2)
            
            # Atomically move temp file to final location
            # This operation is atomic on most filesystems
            temp_file.rename(episode_file)
            print(" saved!")
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            print(f" failed! {e}")
            raise
        
        # Update stats (with file locking for thread safety)
        self._update_stats(full_episode_data)
        
        return episode_id
    
    def _update_stats(self, episode_data: Dict):
        """Update dataset statistics in a thread-safe manner."""
        stats_lock_file = self.dataset_dir / ".stats_lock"
        
        # Use file locking to ensure thread safety
        with open(stats_lock_file, 'w') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                # Load existing stats
                if self.stats_file.exists():
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                else:
                    stats = {
                        'total_episodes': 0,
                        'total_steps': 0,
                        'total_reward': 0.0,
                        'created_at': datetime.now().isoformat(),
                        'last_updated': None,
                        'reward_functions': {},
                        'env_configs': {}
                    }
                
                # Update stats
                stats['total_episodes'] += 1
                stats['total_steps'] += episode_data['metrics']['episode_length']
                stats['total_reward'] += episode_data['metrics']['total_reward']
                stats['last_updated'] = datetime.now().isoformat()
                
                # Track reward functions and env configs
                reward_func = episode_data['metadata'].get('reward_function', 'unknown')
                stats['reward_functions'][reward_func] = stats['reward_functions'].get(reward_func, 0) + 1
                
                # Save updated stats
                with open(self.stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                    
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.stats_file.exists():
            return {'total_episodes': 0, 'total_steps': 0}
        
        with open(self.stats_file, 'r') as f:
            return json.load(f)
    
    def list_episodes(self) -> List[Dict]:
        """List all episodes with basic info."""
        episodes = []
        for episode_file in sorted(self.dataset_dir.glob("episode_*.json")):
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                    episodes.append({
                        'episode_id': episode_data['episode_id'],
                        'timestamp': episode_data['timestamp'],
                        'total_reward': episode_data['metrics']['total_reward'],
                        'episode_length': episode_data['metrics']['episode_length'],
                        'reward_function': episode_data['metadata'].get('reward_function', 'unknown')
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        return episodes
    
    def create_pytorch_dataset(self, transform=None, max_episodes=None) -> EpisodeDataset:
        """Create a PyTorch Dataset for this data."""
        return EpisodeDataset(str(self.dataset_dir), transform=transform, max_episodes=max_episodes)
    
    def cleanup_corrupted_files(self):
        """Remove any corrupted episode files."""
        corrupted = []
        for episode_file in self.dataset_dir.glob("episode_*.json"):
            try:
                with open(episode_file, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, KeyError):
                corrupted.append(episode_file)
                episode_file.unlink()
        
        if corrupted:
            print(f"Cleaned up {len(corrupted)} corrupted episode files")
        
        return len(corrupted)


# Utility functions for easy usage
def save_episode(dataset_dir: str, episode_data: Dict, metadata: Optional[Dict] = None) -> str:
    """Convenience function to save an episode."""
    manager = DatasetManager(dataset_dir)
    return manager.save_episode(episode_data, metadata)


def load_dataset(dataset_dir: str, transform=None, progress: bool = True, max_episodes: int = None) -> EpisodeDataset:
    """Convenience function to load a PyTorch dataset with optional progress reporting.
    
    Args:
        dataset_dir: Path to dataset directory
        transform: Optional transform to apply to episodes
        progress: Whether to show progress information
        max_episodes: Maximum number of episodes to load (None for all)
    """
    if progress:
        print(f"Loading dataset from: {dataset_dir}")
        if max_episodes:
            print(f"Limiting to first {max_episodes} episodes for faster loading")
        
    manager = DatasetManager(dataset_dir)
    dataset = manager.create_pytorch_dataset(transform=transform, max_episodes=max_episodes)
    
    if progress:
        print(f"Dataset loaded successfully!")
        print(f"Episodes available: {len(dataset)}")
        
        # Estimate data size
        if len(dataset) > 0:
            # Get stats from the dataset directory
            stats = get_dataset_stats(dataset_dir)
            if stats.get('total_steps', 0) > 0:
                total_steps = stats['total_steps']
                if max_episodes:
                    # Estimate steps for limited dataset
                    avg_episode_length = total_steps // stats['total_episodes']
                    estimated_steps = avg_episode_length * len(dataset)
                    print(f"Estimated transitions: {estimated_steps:,}")
                else:
                    print(f"Total transitions: {total_steps:,}")
                print(f"Average episode length: {total_steps // stats['total_episodes']:.1f}")
            
    return dataset


def get_dataset_stats(dataset_dir: str) -> Dict:
    """Convenience function to get dataset statistics."""
    manager = DatasetManager(dataset_dir)
    return manager.get_stats()


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset manager
    dataset_dir = "./test_dataset"
    manager = DatasetManager(dataset_dir)
    
    # Create some dummy episode data
    episode_data = {
        'observations': np.random.rand(100, 64, 64, 3).tolist(),  # Example image observations
        'actions': np.random.rand(100, 4).tolist(),  # Example continuous actions
        'rewards': np.random.randn(100).tolist(),
        'dones': [False] * 99 + [True]
    }
    
    metadata = {
        'reward_function': 'align',
        'env_config': {'object_type': 'single', 'aperture_type': 'elf'}
    }
    
    # Save episode
    episode_id = manager.save_episode(episode_data, metadata)
    print(f"Saved episode: {episode_id}")
    
    # Get stats
    stats = manager.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Create PyTorch dataset
    dataset = manager.create_pytorch_dataset()
    print(f"Dataset length: {len(dataset)}")
    
    # Load first episode
    if len(dataset) > 0:
        episode = dataset[0]
        print(f"Episode observations shape: {episode['observations'].shape}")
        print(f"Episode total reward: {episode['rewards'].sum().item()}")
