"""
Improved dataset manager with chunked episode support for scalability.
Handles episodes with 100k-1M+ transitions efficiently.
"""

import json
import time
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import fcntl  # For file locking on Unix systems


class ChunkedEpisodeDataset(Dataset):
    """PyTorch Dataset for loading chunked RL episodes."""
    
    def __init__(self, dataset_dir: str, transform=None, load_in_memory: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.load_in_memory = load_in_memory
        self.episodes = self._load_episode_index()
        self._memory_cache = {} if load_in_memory else None
        
        if load_in_memory:
            self._preload_episodes()
    
    def _load_episode_index(self) -> List[Dict]:
        """Load episode index with chunk information."""
        episodes = []
        
        # Look for episode index files
        for index_file in sorted(self.dataset_dir.glob("episode_*.idx")):
            try:
                with open(index_file, 'r') as f:
                    episode_index = json.load(f)
                    episodes.append(episode_index)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Skipping corrupted index file {index_file}")
                continue
        
        return episodes
    
    def _preload_episodes(self):
        """Preload all episodes into memory for faster access."""
        for episode_idx, episode_info in enumerate(self.episodes):
            self._memory_cache[episode_idx] = self._load_full_episode(episode_info)
    
    def _load_full_episode(self, episode_info: Dict) -> Dict:
        """Load a complete episode from its chunks."""
        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        # Load each chunk
        for chunk_path in episode_info['chunk_files']:
            chunk_file = self.dataset_dir / chunk_path
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    all_observations.extend(chunk_data['observations'])
                    all_next_observations.extend(chunk_data['next_observations'])
                    all_actions.extend(chunk_data['actions'])
                    all_rewards.extend(chunk_data['rewards'])
                    all_dones.extend(chunk_data['dones'])
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Skipping corrupted chunk {chunk_file}")
                continue
        
        return {
            'observations': torch.tensor(all_observations).float(),
            'next_observations': torch.tensor(all_next_observations).float(),
            'actions': torch.tensor(all_actions).float(),
            'rewards': torch.tensor(all_rewards).float(),
            'dones': torch.tensor(all_dones).bool(),
            'metadata': episode_info['metadata']
        }
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        """Load and return an episode or transition."""
        if self.load_in_memory:
            data = self._memory_cache[idx]
        else:
            episode_info = self.episodes[idx]
            data = self._load_full_episode(episode_info)
        
        if self.transform:
            data = self.transform(data)
            
        return data
    
    def get_transition(self, episode_idx: int, transition_idx: int):
        """Get a specific transition from an episode (efficient for large episodes)."""
        episode_info = self.episodes[episode_idx]
        chunk_size = episode_info.get('chunk_size', 1000)
        chunk_idx = transition_idx // chunk_size
        local_idx = transition_idx % chunk_size
        
        if chunk_idx >= len(episode_info['chunk_files']):
            raise IndexError(f"Transition {transition_idx} not found in episode {episode_idx}")
        
        # Load only the required chunk
        chunk_file = self.dataset_dir / episode_info['chunk_files'][chunk_idx]
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        return {
            'observation': torch.tensor(chunk_data['observations'][local_idx]).float(),
            'next_observation': torch.tensor(chunk_data['next_observations'][local_idx]).float(),
            'action': torch.tensor(chunk_data['actions'][local_idx]).float(),
            'reward': torch.tensor(chunk_data['rewards'][local_idx]).float(),
            'done': torch.tensor(chunk_data['dones'][local_idx]).bool()
        }


class ChunkedDatasetManager:
    """
    Thread-safe dataset manager with chunked episode storage.
    Efficiently handles very long episodes (100k-1M+ transitions).
    """
    
    def __init__(self, dataset_dir: str, chunk_size: int = 10000):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = self.dataset_dir / "dataset_stats.json"
        self.lock_file = self.dataset_dir / ".stats_lock"
        self.chunk_size = chunk_size
        
        # Active episode tracking for streaming writes
        self._active_episodes = {}
    
    def start_episode(self, metadata: Optional[Dict] = None) -> str:
        """
        Start a new episode for streaming writes.
        Returns episode ID for subsequent writes.
        """
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        episode_info = {
            'episode_id': episode_id,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'chunk_size': self.chunk_size,
            'current_chunk': 0,
            'total_transitions': 0,
            'chunk_files': [],
            'current_chunk_data': {
                'observations': [],
                'next_observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
        }
        
        self._active_episodes[episode_id] = episode_info
        return episode_id
    
    def add_transition(self, episode_id: str, observation: Any, next_observation: Any, 
                      action: Any, reward: float, done: bool):
        """
        Add a single transition to an active episode.
        Automatically handles chunking and writes to disk when chunks are full.
        """
        if episode_id not in self._active_episodes:
            raise ValueError(f"Episode {episode_id} not started. Call start_episode() first.")
        
        episode_info = self._active_episodes[episode_id]
        chunk_data = episode_info['current_chunk_data']
        
        # Add transition to current chunk
        chunk_data['observations'].append(observation)
        chunk_data['next_observations'].append(next_observation)
        chunk_data['actions'].append(action)
        chunk_data['rewards'].append(reward)
        chunk_data['dones'].append(done)
        
        episode_info['total_transitions'] += 1
        
        # Check if chunk is full
        if len(chunk_data['observations']) >= self.chunk_size:
            self._write_chunk(episode_id)
    
    def _write_chunk(self, episode_id: str):
        """Write current chunk to disk and reset buffer."""
        episode_info = self._active_episodes[episode_id]
        chunk_data = episode_info['current_chunk_data']
        
        if len(chunk_data['observations']) == 0:
            return  # Nothing to write
        
        # Create chunk filename
        chunk_filename = f"episode_{episode_id}_chunk_{episode_info['current_chunk']:04d}.json"
        chunk_path = self.dataset_dir / chunk_filename
        
        # Write chunk
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        # Update episode info
        episode_info['chunk_files'].append(chunk_filename)
        episode_info['current_chunk'] += 1
        
        # Reset chunk buffer
        episode_info['current_chunk_data'] = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        print(f"Wrote chunk {episode_info['current_chunk']} for episode {episode_id[:8]}...")
    
    def finish_episode(self, episode_id: str) -> str:
        """
        Finish an episode, writing any remaining data and creating index file.
        Returns the episode ID.
        """
        if episode_id not in self._active_episodes:
            raise ValueError(f"Episode {episode_id} not found in active episodes.")
        
        episode_info = self._active_episodes[episode_id]
        
        # Write final chunk if it has data
        if len(episode_info['current_chunk_data']['observations']) > 0:
            self._write_chunk(episode_id)
        
        # Remove current_chunk_data from episode info before saving index
        episode_index = {k: v for k, v in episode_info.items() if k != 'current_chunk_data'}
        
        # Write episode index file
        index_filename = f"episode_{episode_id}.idx"
        index_path = self.dataset_dir / index_filename
        
        with open(index_path, 'w') as f:
            json.dump(episode_index, f, indent=2)
        
        # Update global statistics
        self._update_stats(episode_info)
        
        # Remove from active episodes
        del self._active_episodes[episode_id]
        
        print(f"Finished episode {episode_id[:8]}: {episode_info['total_transitions']} transitions")
        return episode_id
    
    def save_episode_batch(self, episode_data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        """
        Save a complete episode at once (for compatibility with existing code).
        Automatically chunks if the episode is large.
        """
        episode_id = self.start_episode(metadata)
        
        observations = episode_data['observations']
        next_observations = episode_data['next_observations']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        dones = episode_data['dones']
        
        # Add all transitions
        for i in range(len(observations)):
            self.add_transition(
                episode_id,
                observations[i],
                next_observations[i],
                actions[i],
                rewards[i],
                dones[i]
            )
        
        return self.finish_episode(episode_id)
    
    def _update_stats(self, episode_info: Dict):
        """Update global dataset statistics."""
        # Use file locking for thread safety
        lock_acquired = False
        try:
            with open(self.lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                lock_acquired = True
                
                # Read current stats
                if self.stats_file.exists():
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                else:
                    stats = {
                        'total_episodes': 0,
                        'total_steps': 0,
                        'total_reward': 0.0,
                        'created_at': datetime.now().isoformat(),
                        'reward_functions': {},
                        'env_configs': {}
                    }
                
                # Update stats
                stats['total_episodes'] += 1
                stats['total_steps'] += episode_info['total_transitions']
                stats['last_updated'] = datetime.now().isoformat()
                
                # Track reward function usage
                reward_func = episode_info['metadata'].get('reward_function', 'unknown')
                stats['reward_functions'][reward_func] = stats['reward_functions'].get(reward_func, 0) + 1
                
                # Write updated stats
                with open(self.stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
        
        finally:
            if lock_acquired:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.stats_file.exists():
            return {'total_episodes': 0, 'total_steps': 0}
        
        with open(self.stats_file, 'r') as f:
            return json.load(f)
    
    def cleanup_active_episodes(self):
        """Force finish all active episodes (useful for cleanup on exit)."""
        for episode_id in list(self._active_episodes.keys()):
            print(f"Force finishing episode {episode_id[:8]}...")
            self.finish_episode(episode_id)


# Backward compatibility: alias for existing code
DatasetManager = ChunkedDatasetManager
EpisodeDataset = ChunkedEpisodeDataset
