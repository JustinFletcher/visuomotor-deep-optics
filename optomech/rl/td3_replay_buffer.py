"""
TD3 Replay Buffer with LSTM Hidden State Support and Dataset Loading.

This module provides a replay buffer designed for TD3 training with LSTM-based
actor and critic networks. It supports:
- Storing trajectory sequences with LSTM hidden states
- Loading trajectories from existing HDF5 datasets
- Computing initial hidden states using provided models
- TBPTT-style sequence sampling
- Compatibility with existing trajectory dataset structures
- Observation preprocessing (normalization, cropping, log-scaling)
"""

import random
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import defaultdict
import pickle
import sys
import os

# Add models directory to path for AutoencoderConfig import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
try:
    from models.train_autoencoder import AutoencoderConfig
except ImportError:
    # Define a dummy class if import fails
    class AutoencoderConfig:
        pass

# Import preprocessing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.transforms import center_crop_transform


def preprocess_observation(obs: np.ndarray,
                          normalize: bool = True,
                          crop_size: Optional[int] = None,
                          log_scale: bool = True) -> np.ndarray:
    """
    Preprocess observation for autoencoder-based TD3 training.
    
    This function applies the same preprocessing pipeline used during 
    autoencoder training to ensure observations match the encoder's expectations.
    
    Args:
        obs: Observation array, shape [C, H, W] or [H, W] or [B, C, H, W]
        normalize: If True, normalize uint16 data to [0, 1] range
        crop_size: If specified, center crop to crop_size x crop_size
        log_scale: If True, apply log1p scaling
        
    Returns:
        Preprocessed observation array
    """
    # Convert to float and normalize if needed
    if normalize:
        if obs.dtype == np.uint16:
            obs = obs.astype(np.float32) / 65535.0
        elif obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        elif obs.max() > 256:  # Auto-detect unnormalized uint16
            obs = obs.astype(np.float32) / 65535.0
        elif obs.max() > 1.0:  # Auto-detect unnormalized uint8
            obs = obs.astype(np.float32) / 255.0
        else:
            obs = obs.astype(np.float32)
    else:
        obs = obs.astype(np.float32)
    
    # Center crop if specified
    if crop_size is not None:
        # Convert to torch tensor for cropping
        obs_tensor = torch.from_numpy(obs)
        obs_tensor = center_crop_transform(obs_tensor, crop_size)
        obs = obs_tensor.numpy()
    
    # Apply log-scaling
    if log_scale:
        obs = np.log1p(obs)
    
    return obs


class TD3ReplayBufferLSTM:
    """
    Replay buffer for TD3 with LSTM networks.
    
    Stores sequences of transitions with initial hidden states for actor and critics.
    Supports loading trajectories from HDF5 datasets and pre-computing hidden states.
    
    Each sequence entry contains:
    - initial_actor_hidden: (h, c) tuple for actor LSTM
    - initial_qf1_hidden: (h, c) tuple for critic1 LSTM  
    - initial_qf2_hidden: (h, c) tuple for critic2 LSTM
    - observation_sequence: list of observations (preprocessed)
    - action_sequence: list of actions
    - prior_action_sequence: list of previous actions
    - reward_sequence: list of rewards
    - prior_reward_sequence: list of previous rewards
    - next_observation_sequence: list of next observations (preprocessed)
    - done_sequence: list of done flags
    """
    
    def __init__(self, 
                 capacity: int,
                 preprocess_observations: bool = False,
                 normalize_obs: bool = True,
                 obs_crop_size: Optional[int] = None,
                 obs_log_scale: bool = False):
        """
        Args:
            capacity: Maximum number of sequences to store
            preprocess_observations: If True, apply preprocessing to observations
            normalize_obs: If True, normalize observations to [0, 1]
            obs_crop_size: If specified, center crop observations to this size
            obs_log_scale: If True, apply log1p scaling to observations
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.metadata = {}
        
        # Preprocessing settings
        self.preprocess_observations = preprocess_observations
        self.normalize_obs = normalize_obs
        self.obs_crop_size = obs_crop_size
        self.obs_log_scale = obs_log_scale
        
        # Track sequence properties
        self.sequence_length = None
        self.observation_shape = None
        self.action_shape = None
        
    def push(self,
             initial_actor_hidden: Tuple[torch.Tensor, torch.Tensor],
             initial_qf1_hidden: Tuple[torch.Tensor, torch.Tensor],
             initial_qf2_hidden: Tuple[torch.Tensor, torch.Tensor],
             observation_sequence: List[np.ndarray],
             action_sequence: List[np.ndarray],
             prior_action_sequence: List[np.ndarray],
             reward_sequence: List[np.ndarray],
             prior_reward_sequence: List[np.ndarray],
             next_observation_sequence: List[np.ndarray],
             done_sequence: List[np.ndarray],
             metadata: Optional[Dict] = None):
        """
        Add a sequence of transitions to the buffer.
        
        Args:
            initial_actor_hidden: Initial (h, c) hidden state for actor
            initial_qf1_hidden: Initial (h, c) hidden state for critic 1
            initial_qf2_hidden: Initial (h, c) hidden state for critic 2
            observation_sequence: List of observations
            action_sequence: List of actions taken
            prior_action_sequence: List of previous actions
            reward_sequence: List of rewards received
            prior_reward_sequence: List of previous rewards
            next_observation_sequence: List of next observations
            done_sequence: List of episode termination flags
            metadata: Optional metadata dict for this sequence
        """
        # Validate sequence lengths
        seq_len = len(observation_sequence)
        assert all(len(seq) == seq_len for seq in [
            action_sequence, prior_action_sequence, reward_sequence,
            prior_reward_sequence, next_observation_sequence, done_sequence
        ]), "All sequences must have the same length"
        
        # Validate hidden states
        assert isinstance(initial_actor_hidden, tuple) and len(initial_actor_hidden) == 2
        assert isinstance(initial_qf1_hidden, tuple) and len(initial_qf1_hidden) == 2
        assert isinstance(initial_qf2_hidden, tuple) and len(initial_qf2_hidden) == 2
        
        # Apply preprocessing to observations if enabled
        if self.preprocess_observations:
            observation_sequence = [
                preprocess_observation(obs, 
                                     normalize=self.normalize_obs,
                                     crop_size=self.obs_crop_size,
                                     log_scale=self.obs_log_scale)
                for obs in observation_sequence
            ]
            next_observation_sequence = [
                preprocess_observation(obs,
                                     normalize=self.normalize_obs,
                                     crop_size=self.obs_crop_size,
                                     log_scale=self.obs_log_scale)
                for obs in next_observation_sequence
            ]
        
        # Store sequence properties on first push
        if self.sequence_length is None:
            self.sequence_length = seq_len
            self.observation_shape = observation_sequence[0].shape
            self.action_shape = action_sequence[0].shape
        
        # Validate shapes match stored properties
        assert seq_len == self.sequence_length, \
            f"Sequence length must be {self.sequence_length}, got {seq_len}"
        assert all(obs.shape == self.observation_shape for obs in observation_sequence), \
            "All observations must have the same shape"
        assert all(act.shape == self.action_shape for act in action_sequence), \
            "All actions must have the same shape"
        
        # Add to buffer (ring buffer style)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (
            initial_actor_hidden,
            initial_qf1_hidden,
            initial_qf2_hidden,
            observation_sequence,
            action_sequence,
            prior_action_sequence,
            reward_sequence,
            prior_reward_sequence,
            next_observation_sequence,
            done_sequence
        )
        
        if metadata is not None:
            self.metadata[self.position] = metadata
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple:
        """
        Sample a batch of sequences from the buffer.
        
        Args:
            batch_size: Number of sequences to sample
            device: Device to move tensors to (if specified)
            
        Returns:
            Tuple of (actor_hidden, qf1_hidden, qf2_hidden, observations, actions,
                     prior_actions, rewards, prior_rewards, next_observations, dones)
            where hidden states are lists of (h, c) tuples per sequence in batch
        """
        batch = random.sample(self.buffer, batch_size)
        
        (initial_actor_hidden,
         initial_qf1_hidden,
         initial_qf2_hidden,
         observation_sequence,
         action_sequence,
         prior_action_sequence,
         reward_sequence,
         prior_reward_sequence,
         next_observation_sequence,
         done_sequence) = zip(*batch)
        
        # Move hidden states to device if specified
        if device:
            actor_hidden = ([h[0].to(device) for h in initial_actor_hidden],
                          [h[1].to(device) for h in initial_actor_hidden])
            qf1_hidden = ([h[0].to(device) for h in initial_qf1_hidden],
                        [h[1].to(device) for h in initial_qf1_hidden])
            qf2_hidden = ([h[0].to(device) for h in initial_qf2_hidden],
                        [h[1].to(device) for h in initial_qf2_hidden])
        else:
            actor_hidden = ([h[0] for h in initial_actor_hidden],
                          [h[1] for h in initial_actor_hidden])
            qf1_hidden = ([h[0] for h in initial_qf1_hidden],
                        [h[1] for h in initial_qf1_hidden])
            qf2_hidden = ([h[0] for h in initial_qf2_hidden],
                        [h[1] for h in initial_qf2_hidden])
        
        return (actor_hidden, qf1_hidden, qf2_hidden,
                observation_sequence, action_sequence, prior_action_sequence,
                reward_sequence, prior_reward_sequence,
                next_observation_sequence, done_sequence)
    
    def load_from_dataset(self,
                         dataset_path: Union[str, Path],
                         actor_model: torch.nn.Module,
                         qf1_model: torch.nn.Module,
                         qf2_model: torch.nn.Module,
                         num_trajectories: Optional[int] = None,
                         sequence_length: int = 16,
                         device: Optional[torch.device] = None,
                         verbose: bool = True):
        """
        Load trajectories from an HDF5 dataset and pre-compute initial hidden states.
        
        This method:
        1. Loads trajectory data from HDF5 files
        2. Chunks them into sequences of specified length
        3. Computes initial hidden states using the provided models
        4. Adds sequences to the replay buffer
        
        Args:
            dataset_path: Path to dataset directory containing HDF5 files
            actor_model: Actor model for computing actor hidden states
            qf1_model: Critic 1 model for computing critic hidden states
            qf2_model: Critic 2 model for computing critic hidden states
            num_trajectories: Number of trajectories to load (None = all)
            sequence_length: Length of TBPTT sequences
            device: Device to run models on
            verbose: Print loading progress
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        # Find all batch files
        batch_files = sorted(list(dataset_path.glob("batch_*.h5")))
        if not batch_files:
            raise ValueError(f"No batch files found in {dataset_path}")
        
        if verbose:
            print(f"Found {len(batch_files)} batch files in {dataset_path}")
        
        # Group data by episode
        episodes = defaultdict(list)
        
        for batch_file in batch_files:
            with h5py.File(batch_file, 'r') as f:
                # Load data
                observations = f['observations'][:]
                actions = f['actions'][:]
                rewards = f['rewards'][:]
                episode_ids = f['episode_ids'][:]
                episode_steps = f['episode_steps'][:]
                
                # Group by episode
                for i in range(len(observations)):
                    ep_id = episode_ids[i] if isinstance(episode_ids[i], str) else episode_ids[i].decode()
                    episodes[ep_id].append({
                        'observation': observations[i],
                        'action': actions[i],
                        'reward': rewards[i],
                        'episode_step': episode_steps[i]
                    })
        
        if verbose:
            print(f"Loaded {len(episodes)} episodes from dataset")
        
        # Sort episodes by step within each episode
        for ep_id in episodes:
            episodes[ep_id].sort(key=lambda x: x['episode_step'])
        
        # Limit number of trajectories if specified
        episode_ids = list(episodes.keys())
        if num_trajectories is not None:
            episode_ids = episode_ids[:num_trajectories]
            if verbose:
                print(f"Using first {num_trajectories} episodes")
        
        # Process each episode into sequences
        sequences_added = 0
        actor_model.eval()
        qf1_model.eval()
        qf2_model.eval()
        
        with torch.no_grad():
            for ep_id in episode_ids:
                episode = episodes[ep_id]
                
                # Split episode into sequences
                for seq_start in range(0, len(episode) - sequence_length, sequence_length):
                    seq_end = seq_start + sequence_length
                    sequence_data = episode[seq_start:seq_end]
                    
                    # Extract sequences
                    obs_seq = [d['observation'] for d in sequence_data]
                    action_seq = [d['action'] for d in sequence_data]
                    reward_seq = [d['reward'] for d in sequence_data]
                    
                    # Create prior action and reward sequences
                    # For first step, use zeros
                    prior_action_seq = [np.zeros_like(action_seq[0])] + action_seq[:-1]
                    prior_reward_seq = [np.array(0.0, dtype=np.float32)] + reward_seq[:-1]
                    
                    # Create next observations (shifted by 1)
                    next_obs_seq = obs_seq[1:] + [obs_seq[-1]]  # Last next_obs repeats
                    
                    # Create done flags (all False except potentially last)
                    done_seq = [np.array(False)] * (sequence_length - 1) + \
                              [np.array(seq_end >= len(episode))]
                    
                    # Compute initial hidden states using models
                    # Get zero hidden states for start of sequence
                    initial_actor_hidden = actor_model.get_zero_hidden()
                    initial_qf1_hidden = qf1_model.get_zero_hidden()
                    initial_qf2_hidden = qf2_model.get_zero_hidden()
                    
                    # Move to device if specified
                    if device:
                        initial_actor_hidden = (initial_actor_hidden[0].to(device),
                                              initial_actor_hidden[1].to(device))
                        initial_qf1_hidden = (initial_qf1_hidden[0].to(device),
                                            initial_qf1_hidden[1].to(device))
                        initial_qf2_hidden = (initial_qf2_hidden[0].to(device),
                                            initial_qf2_hidden[1].to(device))
                    
                    # Add to buffer
                    self.push(
                        initial_actor_hidden=initial_actor_hidden,
                        initial_qf1_hidden=initial_qf1_hidden,
                        initial_qf2_hidden=initial_qf2_hidden,
                        observation_sequence=obs_seq,
                        action_sequence=action_seq,
                        prior_action_sequence=prior_action_seq,
                        reward_sequence=reward_seq,
                        prior_reward_sequence=prior_reward_seq,
                        next_observation_sequence=next_obs_seq,
                        done_sequence=done_seq,
                        metadata={'episode_id': ep_id, 'sequence_start': seq_start}
                    )
                    sequences_added += 1
        
        if verbose:
            print(f"Added {sequences_added} sequences to replay buffer")
            print(f"Buffer now contains {len(self)} sequences")
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, dir_path: Union[str, Path], chunk_size: int = 100):
        """
        Save the replay buffer to disk in chunks.
        
        Args:
            dir_path: Directory to save buffer chunks
            chunk_size: Number of sequences per chunk file
        """
        import os
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        meta = {
            'capacity': self.capacity,
            'position': self.position,
            'num_chunks': (len(self.buffer) + chunk_size - 1) // chunk_size,
            'chunk_size': chunk_size,
            'sequence_length': self.sequence_length,
            'observation_shape': self.observation_shape,
            'action_shape': self.action_shape,
            'metadata': self.metadata
        }
        
        with open(dir_path / 'meta.pt', 'wb') as f:
            pickle.dump(meta, f)
        
        for i in range(meta['num_chunks']):
            chunk = self.buffer[i * chunk_size:(i + 1) * chunk_size]
            with open(dir_path / f'chunk_{i:05d}.pt', 'wb') as f:
                pickle.dump(chunk, f)
    
    def restore(self, dir_path: Union[str, Path]):
        """
        Restore the replay buffer from disk.
        
        Args:
            dir_path: Directory containing saved buffer chunks
        """
        dir_path = Path(dir_path)
        
        with open(dir_path / 'meta.pt', 'rb') as f:
            meta = pickle.load(f)
        
        self.capacity = meta['capacity']
        self.sequence_length = meta['sequence_length']
        self.observation_shape = meta['observation_shape']
        self.action_shape = meta['action_shape']
        self.metadata = meta.get('metadata', {})
        
        self.buffer = []
        for i in range(meta['num_chunks']):
            chunk_path = dir_path / f'chunk_{i:05d}.pt'
            if not chunk_path.exists():
                break
            with open(chunk_path, 'rb') as f:
                chunk = pickle.dump(f)
                self.buffer.extend(chunk)
        
        self.position = len(self.buffer)


def load_pretrained_encoder(autoencoder_path: Union[str, Path],
                           model_class: type = None,
                           device: Optional[torch.device] = None,
                           freeze: bool = True) -> torch.nn.Module:
    """
    Load a pretrained autoencoder and extract its encoder.
    
    Args:
        autoencoder_path: Path to saved autoencoder checkpoint
        model_class: Autoencoder model class to instantiate (None to auto-detect)
        device: Device to load model on
        freeze: Whether to freeze encoder parameters
        
    Returns:
        Encoder module with weights loaded
    """
    autoencoder_path = Path(autoencoder_path)
    if not autoencoder_path.exists():
        raise ValueError(f"Autoencoder checkpoint not found: {autoencoder_path}")
    
    # Load checkpoint (weights_only=False to support custom classes)
    checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=False)
    
    # If checkpoint contains the full model, extract encoder directly
    if 'model' in checkpoint:
        autoencoder = checkpoint['model']
        if device:
            autoencoder = autoencoder.to(device)
        encoder = autoencoder.encoder
    elif 'model_state_dict' in checkpoint:
        # Check if checkpoint has config to determine architecture
        if 'config' in checkpoint:
            config = checkpoint['config']
            # Import the appropriate model based on config
            try:
                from models.models import AutoEncoderCNN, AutoEncoderResNet
                
                # Get architecture parameters from config or infer from state_dict
                input_channels = getattr(config, 'input_channels', 2)
                latent_dim = getattr(config, 'latent_dim', 256)
                
                # Infer input_channels from first layer if mismatch
                state_dict = checkpoint['model_state_dict']
                if 'encoder.0.weight' in state_dict:
                    actual_channels = state_dict['encoder.0.weight'].shape[1]
                    if actual_channels != input_channels:
                        print(f"Warning: Config says {input_channels} channels but weights have {actual_channels}. Using {actual_channels}.")
                        input_channels = actual_channels
                
                #  Infer latent_dim from bottleneck if present
                if 'bottleneck_encode.0.weight' in state_dict:
                    actual_latent = state_dict['bottleneck_encode.0.weight'].shape[0]
                    if actual_latent != latent_dim:
                        print(f"Warning: Config says {latent_dim} latent_dim but weights have {actual_latent}. Using {actual_latent}.")
                        latent_dim = actual_latent
                
                if hasattr(config, 'arch') and 'resnet' in config.arch.lower():
                    # Use ResNet autoencoder
                    autoencoder = AutoEncoderResNet(input_channels=input_channels, latent_dim=latent_dim)
                else:
                    # Use CNN autoencoder  
                    autoencoder = AutoEncoderCNN(input_channels=input_channels, latent_dim=latent_dim)
                
                autoencoder.load_state_dict(state_dict)
                if device:
                    autoencoder = autoencoder.to(device)
                encoder = autoencoder.encoder
            except Exception as e:
                raise ValueError(f"Could not instantiate autoencoder from config. Error: {e}")
        elif model_class is not None:
            # Use provided model class
            autoencoder = model_class()
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            if device:
                autoencoder = autoencoder.to(device)
            encoder = autoencoder.encoder
        else:
            raise ValueError("Checkpoint has model_state_dict but no config and no model_class provided")
    else:
        raise ValueError(f"Checkpoint format not recognized. Keys: {list(checkpoint.keys())}")
    
    # Freeze if requested
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
    
    if device:
        encoder = encoder.to(device)
    
    return encoder


def load_pretrained_actor(actor_path: Union[str, Path],
                         actor_class: type,
                         device: Optional[torch.device] = None,
                         freeze_encoder: bool = True) -> torch.nn.Module:
    """
    Load a pretrained actor for fine-tuning.
    
    Args:
        actor_path: Path to saved actor checkpoint
        actor_class: Actor model class to instantiate
        device: Device to load model on
        freeze_encoder: Whether to freeze the encoder portion
        
    Returns:
        Actor model with weights loaded
    """
    actor_path = Path(actor_path)
    if not actor_path.exists():
        raise ValueError(f"Actor checkpoint not found: {actor_path}")
    
    # Load checkpoint
    checkpoint = torch.load(actor_path, map_location=device)
    
    # Instantiate model
    actor = actor_class()
    actor.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally freeze encoder
    if freeze_encoder and hasattr(actor, 'visual_encoder'):
        for param in actor.visual_encoder.parameters():
            param.requires_grad = False
    
    if device:
        actor = actor.to(device)
    
    return actor
