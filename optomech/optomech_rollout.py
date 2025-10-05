#!/usr/bin/env python3
"""
Universal Optomech Rollout Script

This script provides a universal interface for evaluating ANY trained model 
(SML, RL, etc.) in the Optomech environment. It's designed to be completely 
model-agnostic and support various model architectures including:

- Feedforward networks (CNN, ResNet, etc.)
- Recurrent networks (LSTM, GRU, etc.) 
- Custom architectures with arbitrary interfaces

The script automatically detects model properties and adapts its behavior
accordingly, ensuring fair apples-to-apples comparisons across different
training methods.

Key Features:
- Universal model loading (PyTorch, custom formats)
- Automatic hidden state management for recurrent models
- Configurable action noise injection
- Comprehensive episode recording and statistics
- Support for multiple evaluation episodes
- Detailed logging and progress tracking
- Environment configuration support (including incremental control mode)
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import tyro

# Import TrainingConfig for model loading compatibility
try:
    from supervised_ml.train_sml_model import TrainingConfig
except ImportError:
    # Fallback if import fails
    @dataclass
    class TrainingConfig:
        pass


# =============================================================================
# MODEL INTERFACE CLASSES
# =============================================================================

class ModelInterface:
    """
    Abstract base class defining the universal model interface.
    
    All model wrappers must implement these methods to work with the rollout script.
    This ensures consistent behavior regardless of the underlying model architecture.
    """
    
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the model interface.
        
        Args:
            model_path: Path to the saved model file
            device: PyTorch device to load the model on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.has_hidden_state = False
        self.hidden_state = None
        
    def load_model(self) -> None:
        """Load the model from the specified path."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make a prediction given an observation.
        
        Args:
            observation: Environment observation
            **kwargs: Additional arguments (prior_action, prior_reward, etc.)
            
        Returns:
            Predicted action as numpy array
        """
        raise NotImplementedError("Subclasses must implement predict")
    
    def reset_hidden_state(self) -> None:
        """Reset hidden state for recurrent models (no-op for feedforward models)."""
        if self.has_hidden_state:
            self.hidden_state = None
    
    def get_action_scale(self) -> float:
        """Get the action scale for exploration noise (default: 1.0)."""
        return getattr(self.model, 'action_scale', 1.0)


class SMLModelInterface(ModelInterface):
    """
    Interface for Supervised Machine Learning (SML) models.
    
    Handles CNN/ResNet models trained to predict perfect actions from observations.
    These are typically feedforward models without hidden state.
    """
    
    def load_model(self) -> None:
        """Load an SML model (checkpoint format from training script)."""
        print(f"🧠 Loading SML model from: {self.model_path}")
        
        try:
            # Load the checkpoint saved by the training script
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print(f"✅ Checkpoint loaded successfully, type: {type(checkpoint)}")
            
            # Check if it's a checkpoint dictionary (from training script)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("✅ Loading SML model from checkpoint")
                
                # Check checkpoint for model architecture and parameters
                model_arch = "sml_cnn"  # default
                input_channels = 2  # default
                action_dim = 15     # default
                
                # Try to get values from top-level checkpoint data first (new format)
                if 'model_arch' in checkpoint:
                    model_arch = checkpoint['model_arch']
                    print(f"✅ Found model architecture: {model_arch}")
                if 'input_channels' in checkpoint:
                    input_channels = checkpoint['input_channels']
                if 'action_dim' in checkpoint:
                    action_dim = checkpoint['action_dim']
                
                # Fallback: Try to get from config if available
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    if hasattr(config, 'action_dim'):
                        action_dim = config.action_dim
                        print(f"🔍 Config override action_dim: {action_dim}")
                    if hasattr(config, 'model_arch'):
                        model_arch = config.model_arch
                        print(f"🔍 Config override model_arch: {model_arch}")
                
                # Fallback: Extract parameters from state_dict
                state_dict = checkpoint['model_state_dict']
                
                # Extract input channels from the first conv layer
                if 'features.0.weight' in state_dict:
                    input_channels = state_dict['features.0.weight'].shape[1]
                elif 'conv1.weight' in state_dict:
                    input_channels = state_dict['conv1.weight'].shape[1]
                print(f"🔍 Detected input_channels: {input_channels}")
                
                # Extract action_dim from the final linear layer
                if 'classifier.6.weight' in state_dict:
                    action_dim = state_dict['classifier.6.weight'].shape[0]
                elif 'fc.weight' in state_dict:
                    action_dim = state_dict['fc.weight'].shape[0]
                print(f"🔍 Detected action_dim: {action_dim}")
                
                # Try to detect architecture from state_dict keys if config doesn't have it
                if model_arch == "sml_cnn" and 'conv1.weight' in state_dict:
                    if 'gn1.weight' in state_dict:
                        model_arch = "sml_resnet_gn"
                    else:
                        model_arch = "sml_resnet"
                    print(f"✅ Auto-detected model architecture from state_dict: {model_arch}")
                
                # Import model definitions from training script
                try:
                    import sys
                    from pathlib import Path
                    current_dir = Path(__file__).parent
                    project_root = current_dir.parent
                    sys.path.insert(0, str(project_root))
                    
                    # Import all model classes from the training script
                    from optomech.supervised_ml.train_sml_model import (
                        SMLModel, SMLResNet, SMLResNetGN, SMLSimple, 
                        SMLHRNet, SMLVanillaConv, create_model
                    )
                    
                    # Create model using the factory function
                    self.model = create_model(
                        arch=model_arch,
                        input_channels=input_channels,
                        action_dim=action_dim
                    )
                    print(f"✅ Created {model_arch} model: input_channels={input_channels}, action_dim={action_dim}")
                    
                except ImportError as e:
                    print(f"⚠️  Could not import from training script: {e}")
                    print("🔄 Falling back to basic SMLModel definition")
                    
                    # Fallback: Define basic SMLModel class here
                    import torch.nn as nn
                    
                    class SMLModel(nn.Module):
                        """CNN model for predicting perfect actions from observations"""
                        
                        def __init__(self, input_channels=2, action_dim=15):
                            super(SMLModel, self).__init__()
                            
                            # CNN feature extractor
                            self.features = nn.Sequential(
                                nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool2d((4, 4))
                            )
                            
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
                            x = x.view(x.size(0), -1)
                            x = self.classifier(x)
                            return x
                    
                    # Create fallback model instance
                    self.model = SMLModel(input_channels=input_channels, action_dim=action_dim)
                    print(f"✅ Created fallback SML model: input_channels={input_channels}, action_dim={action_dim}")
                
                # Load the saved state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded SML model (epoch {checkpoint.get('epoch', 'unknown')})")
                
            else:
                # Try loading as a complete model (legacy support)
                self.model = checkpoint
                print("✅ Loaded complete model object")
                
        except Exception as e:
            print(f"❌ Error loading SML model: {e}")
            raise ValueError(f"Could not load SML model: {e}")
        
        if self.model is None:
            raise ValueError("Model loading failed - self.model is None")
            
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.has_hidden_state = False  # SML models are typically feedforward
        print(f"✅ Model loaded and ready: {type(self.model)}")
        
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict action from observation using the SML model.
        
        Args:
            observation: Environment observation (typically image)
            **kwargs: Ignored for SML models
            
        Returns:
            Predicted action
        """
        # Debug: Check if model is None
        if self.model is None:
            raise ValueError("Model is None - load_model was not called or failed")
        
        # Convert observation to tensor and add batch dimension if needed
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        if len(obs_tensor.shape) == 3:  # Add batch dimension for single observation
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Debug: Check tensor shapes
        # print(f"    DEBUG: obs_tensor shape: {obs_tensor.shape}")
        # print(f"    DEBUG: model first layer weight shape: {list(self.model.parameters())[0].shape}")
        
        # Make prediction
        with torch.no_grad():
            try:
                action_tensor = self.model(obs_tensor)
            except Exception as e:
                print(f"    DEBUG: Error in model forward pass: {e}")
                raise
            
        # Convert back to numpy and remove batch dimension
        action = action_tensor.cpu().numpy()
        if action.shape[0] == 1:  # Remove batch dimension if present
            action = action.squeeze(0)
            
        return action
    
    def get_action_scale(self) -> float:
        """Get the action scale for SML models (default: 1.0 for normalized actions)."""
        return 1.0


class RLModelInterface(ModelInterface):
    """
    Interface for Reinforcement Learning models.
    
    Handles actor-critic models, policy networks, and recurrent RL architectures.
    Can support both feedforward and recurrent models with hidden states.
    """
    
    def __init__(self, model_path: str, device: torch.device, model_type: str = "auto"):
        """
        Initialize RL model interface.
        
        Args:
            model_path: Path to saved model
            device: PyTorch device
            model_type: Type hint for model architecture ("actor", "ddpg", "impala", etc.)
        """
        super().__init__(model_path, device)
        self.model_type = model_type
        self.actor_model = None
        
    def load_model(self) -> None:
        """Load an RL model (can be actor-only or actor-critic pair)."""
        print(f"🎯 Loading RL model from: {self.model_path}")
        
        try:
            # Load the model/checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Handle dictionary checkpoint format
                if 'model_state_dict' in checkpoint:
                    print("🧠 Loading SML model from checkpoint")
                    # This is an SML model - need to reconstruct it
                    from supervised_ml.train_sml_model import SMLModel
                    
                    # Check the actual input channels from the saved weights
                    first_layer_weight = checkpoint['model_state_dict']['features.0.weight']
                    input_channels = first_layer_weight.shape[1]
                    # DEBUG: print(f"Detected {input_channels} input channels from saved weights")
                    
                    self.model = SMLModel(input_channels=input_channels, action_dim=15)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ SML model loaded successfully")
                elif 'model' in checkpoint:
                    # DEBUG: print("Found 'model' key in checkpoint")
                    self.model = checkpoint['model']
                elif 'actor' in checkpoint:
                    # DEBUG: print("Found 'actor' key in checkpoint")
                    self.model = checkpoint['actor']
                elif 'state_dict' in checkpoint:
                    # DEBUG: print("Found 'state_dict' key in checkpoint")
                    # This would need model architecture to load into
                    raise NotImplementedError("state_dict loading requires model architecture specification")
                else:
                    # DEBUG: print(f"Available keys: {list(checkpoint.keys())}")
                    # Try to use first model-like key
                    for key in checkpoint.keys():
                        if 'model' in key.lower() or 'actor' in key.lower():
                            # DEBUG: print(f"Using key: {key}")
                            self.model = checkpoint[key]
                            break
                    if self.model is None:
                        raise ValueError(f"Could not find model in checkpoint keys: {list(checkpoint.keys())}")
            elif isinstance(checkpoint, tuple) and len(checkpoint) == 2:
                # DDPG-style: (actor_params, critic_params)
                # DEBUG: print("Detected DDPG-style actor-critic checkpoint")
                actor_params, _ = checkpoint  # We only need the actor for rollouts
                # This requires knowing the actor architecture
                raise NotImplementedError("DDPG checkpoint loading requires actor architecture specification")
                
            elif hasattr(checkpoint, 'predict') or callable(checkpoint):
                # Complete model object
                print("✅ Loaded RL model successfully")
                self.model = checkpoint
                
            else:
                # Assume it's the actor model directly
                print("🎭 Loaded RL actor model")
                self.model = checkpoint
                
        except Exception as e:
            raise ValueError(f"Could not load RL model: {e}")
        
        if self.model is None:
            raise ValueError("Model loading failed - no valid model found")
            
        self.model.eval()
        
        # Check if model has hidden state (common in LSTM-based RL models)
        self._detect_hidden_state_support()
        
    def _detect_hidden_state_support(self) -> None:
        """Automatically detect if the model uses hidden states."""
        # Look for LSTM/GRU layers or hidden state parameters
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                # DEBUG: print(f"Detected recurrent layer: {name}")
                self.has_hidden_state = True
                break
                
        # Check for hidden state methods
        if hasattr(self.model, 'init_hidden') or hasattr(self.model, 'reset_hidden'):
            # DEBUG: print("Detected hidden state methods in model")
            self.has_hidden_state = True
            
    def reset_hidden_state(self) -> None:
        """Reset hidden state for recurrent RL models."""
        if self.has_hidden_state:
            if hasattr(self.model, 'reset_hidden'):
                self.hidden_state = self.model.reset_hidden()
            elif hasattr(self.model, 'init_hidden'):
                self.hidden_state = self.model.init_hidden()
            else:
                # Generic LSTM hidden state initialization
                # This would need to be customized based on the specific model
                self.hidden_state = self._create_default_hidden_state()
                
    def _create_default_hidden_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create default hidden state for LSTM models."""
        # Default values - should be customized based on actual model architecture
        num_layers = 1
        hidden_dim = 256
        batch_size = 1
        
        h = torch.zeros(num_layers, batch_size, hidden_dim, device=self.device)
        c = torch.zeros(num_layers, batch_size, hidden_dim, device=self.device)
        return (h, c)
        
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict action using the RL model.
        
        Args:
            observation: Environment observation
            **kwargs: Additional inputs (prior_action, prior_reward, hidden_state)
            
        Returns:
            Predicted action
        """
        # Convert inputs to tensors
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        if len(obs_tensor.shape) == 3:  # Add batch dimension
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            if self.has_hidden_state:
                # Handle recurrent models
                prior_action = kwargs.get('prior_action', None)
                prior_reward = kwargs.get('prior_reward', None)
                
                if prior_action is not None:
                    prior_action = torch.FloatTensor(prior_action).to(self.device)
                    if len(prior_action.shape) == 1:
                        prior_action = prior_action.unsqueeze(0)
                        
                if prior_reward is not None:
                    prior_reward = torch.FloatTensor([prior_reward]).to(self.device)
                    
                # Call model with hidden state
                if prior_action is not None and prior_reward is not None:
                    # Full recurrent call with prior action and reward
                    action_tensor, self.hidden_state = self.model(
                        obs_tensor, prior_action, prior_reward, self.hidden_state
                    )
                else:
                    # Simple recurrent call
                    output = self.model(obs_tensor, self.hidden_state)
                    if isinstance(output, tuple):
                        action_tensor, self.hidden_state = output
                    else:
                        action_tensor = output
            else:
                # Feedforward model
                action_tensor = self.model(obs_tensor)
                
        # Convert to numpy
        action = action_tensor.cpu().numpy()
        if action.shape[0] == 1:
            action = action.squeeze(0)
            
        return action


class CustomModelInterface(ModelInterface):
    """
    Interface for custom model formats.
    
    Provides a flexible wrapper for models that don't fit standard patterns.
    Users can subclass this to handle specific custom architectures.
    """
    
    def __init__(self, model_path: str, device: torch.device, 
                 load_fn: Optional[Callable] = None,
                 predict_fn: Optional[Callable] = None):
        """
        Initialize custom model interface.
        
        Args:
            model_path: Path to model file
            device: PyTorch device
            load_fn: Custom loading function
            predict_fn: Custom prediction function
        """
        super().__init__(model_path, device)
        self.load_fn = load_fn
        self.predict_fn = predict_fn
        
    def load_model(self) -> None:
        """Load model using custom loading function."""
        if self.load_fn is not None:
            print(f"🔧 Loading custom model from: {self.model_path}")
            self.model = self.load_fn(self.model_path, self.device)
        else:
            # Default: try to load as pickle or torch
            try:
                self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Make prediction using custom prediction function."""
        if self.predict_fn is not None:
            return self.predict_fn(self.model, observation, **kwargs)
        else:
            # Default behavior
            return self.model(observation)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model_interface(model_path: str, model_type: str, device: torch.device, 
                          **kwargs) -> ModelInterface:
    """
    Factory function to create the appropriate model interface.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ("sml", "rl", "ddpg", "impala", "custom")
        device: PyTorch device
        **kwargs: Additional arguments for specific model types
        
    Returns:
        Initialized model interface
    """
    model_type = model_type.lower()
    
    if model_type in ["sml", "supervised", "cnn", "resnet"]:
        interface = SMLModelInterface(model_path, device)
        interface.load_model()
        return interface
    elif model_type in ["rl", "ddpg", "td3", "sac", "ppo", "a2c"]:
        interface = RLModelInterface(model_path, device, model_type)
        interface.load_model()
        return interface
    elif model_type in ["impala", "lstm", "recurrent"]:
        interface = RLModelInterface(model_path, device, model_type)
        interface.load_model()
        return interface
    elif model_type == "custom":
        interface = CustomModelInterface(model_path, device, **kwargs)
        interface.load_model()
        return interface
    else:
        # Auto-detect based on file content
        print(f"🔍 Auto-detecting model type for: {model_path}")
        try:
            # Try RL model first (most common)
            interface = RLModelInterface(model_path, device, "auto")
            interface.load_model()
            return interface
        except:
            try:
                # Try SML model
                interface = SMLModelInterface(model_path, device)
                interface.load_model()
                return interface
            except:
                # Fallback to custom
                interface = CustomModelInterface(model_path, device)
                interface.load_model()
                return interface


# =============================================================================
# ENVIRONMENT UTILITIES
# =============================================================================

def make_env(env_id: str, env_args: Namespace) -> Callable:
    """
    Create environment factory function.
    
    Args:
        env_id: Environment identifier
        env_args: Environment configuration arguments
        
    Returns:
        Function that creates environment instances
    """
    def thunk():
        if env_id == "optomech-v1":
            # Safely extract environment arguments, filtering out None values
            env_kwargs = {}
            if env_args is not None:
                for key, value in vars(env_args).items():
                    if value is not None:
                        env_kwargs[key] = value
            env = gym.make(env_id, **env_kwargs)
        else:
            env = gym.make(env_id)
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    return thunk


def setup_environment(env_id: str, env_args: Namespace, num_envs: int = 1) -> gym.vector.VectorEnv:
    """
    Setup and configure the environment.
    
    Args:
        env_id: Environment identifier
        env_args: Environment configuration
        num_envs: Number of parallel environments
        
    Returns:
        Configured vector environment
    """
    # Register custom environments
    if env_id == "optomech-v1":
        gym.envs.registration.register(
            id='optomech-v1',
            entry_point='optomech.optomech:OptomechEnv',
            max_episode_steps=env_args.max_episode_steps,
        )
    elif env_id == "VisualPendulum-v1":
        gym.envs.registration.register(
            id='VisualPendulum-v1',
            entry_point='visual_pendulum:VisualPendulumEnv',
            max_episode_steps=env_args.max_episode_steps,
        )
    
    # Create vector environment
    if getattr(env_args, 'subproc_env', False):
        print("🚀 Using SubprocVectorEnv for parallel environments")
        envs = gym.vector.SubprocVectorEnv(
            [make_env(env_id, env_args) for _ in range(num_envs)]
        )
    elif getattr(env_args, 'async_env', False):
        print("⚡ Using AsyncVectorEnv for parallel environments")
        envs = gym.vector.AsyncVectorEnv(
            [make_env(env_id, env_args) for _ in range(num_envs)]
        )
    else:
        print("🔄 Using SyncVectorEnv for environments")
        envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, env_args) for _ in range(num_envs)]
        )
    
    return envs


# =============================================================================
# ROLLOUT ENGINE
# =============================================================================

class UniversalRolloutEngine:
    """
    Universal rollout engine that can evaluate any model in the Optomech environment.
    
    This class handles the core rollout logic while remaining completely agnostic
    to the specific model architecture or training method used.
    """
    
    def __init__(self, 
                 model_interface: Optional[ModelInterface] = None,
                 env_args: Optional[Namespace] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the rollout engine.
        
        Args:
            model_interface: Model interface for action prediction
            env_args: Environment configuration
            device: PyTorch device for computations
        """
        self.model_interface = model_interface
        self.env_args = env_args
        self.device = device or self._get_device()
        
        # Rollout state
        self.episodic_returns = []
        self.episode_data = []
        self.step_wise_rewards = []  # List of episodes, each episode is a list of step rewards
        self.global_step = 0
        self.env_metadata = {}  # Store environment metadata for rendering compatibility
        
    def _get_device(self) -> torch.device:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            print("🚀 Using CUDA acceleration")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("🍎 Using MPS acceleration")
            return torch.device("mps")
        else:
            print("💻 Using CPU")
            return torch.device("cpu")
    
    def run_rollout(self, 
                   num_episodes: int = 1,
                   exploration_noise: float = 0.0,
                   save_path: Optional[str] = None,
                   save_episodes: bool = False,
                   random_policy: bool = False,
                   zero_policy: bool = False) -> Tuple[List[float], List[List[float]]]:
        """
        Run rollout evaluation for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            exploration_noise: Standard deviation of Gaussian noise to add to actions
            save_path: Directory to save episode data (optional)
            save_episodes: Whether to save detailed episode information
            random_policy: Whether to use random actions instead of model
            zero_policy: Whether to use all-zeros actions instead of model
            
        Returns:
            Tuple of (episodic_returns, step_wise_rewards)
            - episodic_returns: List of cumulative rewards for each episode
            - step_wise_rewards: List of episodes, each containing step-by-step rewards
        """
        print(f"🎬 Starting rollout evaluation for {num_episodes} episodes")
        if zero_policy:
            print(f"🎯 Model: Zero Policy (all-zeros actions)")
        elif random_policy:
            print(f"🎯 Model: Random Policy")
        else:
            print(f"🎯 Model: {self.model_interface.__class__.__name__}")
        print(f"🔊 Exploration noise: {exploration_noise}")
        
        # Setup environment
        env_id = getattr(self.env_args, 'env_id', 'optomech-v1')
        num_envs = getattr(self.env_args, 'num_envs', 1)
        envs = setup_environment(env_id, self.env_args, num_envs)
        
        # Get environment metadata for render_history compatibility
        try:
            env_metadata = envs.get_attr('metadata')[0]  # Remove indices parameter
            self.env_metadata = {
                "frames_per_decision": env_metadata.get("frames_per_decision", 1),
                "commands_per_decision": env_metadata.get("commands_per_decision", 1), 
                "commands_per_frame": env_metadata.get("commands_per_frame", 1),
                "ao_steps_per_frame": env_metadata.get("ao_steps_per_frame", 1),
            }
            print(f"📊 Retrieved environment metadata: frames_per_decision={self.env_metadata['frames_per_decision']}")
        except Exception as e:
            print(f"⚠️  Could not retrieve environment metadata: {e}")
            self.env_metadata = {
                "frames_per_decision": 1,
                "commands_per_decision": 1,
                "commands_per_frame": 1,
                "ao_steps_per_frame": 1,
            }
        
        # Initialize rollout state
        self.episodic_returns = []
        self.episode_data = []
        self.step_wise_rewards = []
        self.current_episode_rewards = [[] for _ in range(num_envs)]  # Track step rewards for current episodes
        self.global_step = 0
        
        # Reset environment and prepare tracking variables
        obs, _ = envs.reset()
        obs = self._preprocess_observation(obs)
        
        # Initialize prior state for recurrent models
        prior_actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        _, prior_rewards, _, _, _ = envs.step(prior_actions)
        
        # Reset model hidden state
        if self.model_interface and self.model_interface.has_hidden_state:
            self.model_interface.reset_hidden_state()
        
        # Generate unique IDs for each environment
        env_uuids = [str(uuid.uuid4()) for _ in range(num_envs)]
        
        # Main rollout loop
        start_time = time.time()
        while len(self.episodic_returns) < num_episodes:
            # Progress reporting
            if self.global_step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"📊 Episode {len(self.episodic_returns)}/{num_episodes} | "
                      f"Step {self.global_step} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            # Create save directories if needed
            if save_episodes and save_path:
                self._create_episode_directories(env_uuids, save_path)
            
            # Get actions from model, random policy, or zero policy
            if zero_policy:
                actions = self._sample_zero_actions(envs, num_envs)
            elif random_policy or self.model_interface is None:
                actions = self._sample_random_actions(envs, num_envs)
            else:
                try:
                    actions = self._get_model_actions(obs, prior_actions, prior_rewards, 
                                                    exploration_noise, envs)
                except Exception as e:
                    print(f"    Error in _get_model_actions: {e}")
                    raise
            
            # Step environment
            try:
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                next_obs = self._preprocess_observation(next_obs)
            except Exception as e:
                print(f"    Error in envs.step: {e}")
                raise
            
            # Track step-wise rewards for each environment
            for env_idx, reward in enumerate(rewards):
                self.current_episode_rewards[env_idx].append(float(reward))
            
            # Update tracking variables
            prior_actions = actions
            prior_rewards = rewards
            
            # Save episode data if requested
            if save_episodes:
                self._save_step_data(actions, rewards, env_uuids, save_path, 
                                   next_obs, terminations, truncations, infos)
            
            # Check for episode completion
            if "final_info" in infos:
                self._handle_episode_completion(infos, env_uuids, save_path)
                
                # Reset hidden state for new episodes
                if self.model_interface and self.model_interface.has_hidden_state:
                    self.model_interface.reset_hidden_state()
                
                # Generate new UUIDs for next episodes
                env_uuids = [str(uuid.uuid4()) for _ in range(num_envs)]
            
            # Update for next step
            obs = next_obs
            self.global_step += 1
        
        # Cleanup
        envs.close()
        
        # Post-rollout rendering if requested
        if getattr(self.env_args, 'render', False) and save_episodes and save_path:
            print(f"🎨 Starting post-rollout rendering...")
            self._render_rollout_results(save_path)
        
        # Report final statistics
        total_time = time.time() - start_time
        print(f"✅ Rollout complete!")
        print(f"📈 Episodes: {len(self.episodic_returns)}")
        print(f"📊 Mean return: {np.mean(self.episodic_returns):.3f} ± {np.std(self.episodic_returns):.3f}")
        print(f"📏 Min/Max return: {np.min(self.episodic_returns):.3f} / {np.max(self.episodic_returns):.3f}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🚀 Steps per second: {self.global_step / total_time:.1f}")
        
        return self.episodic_returns, self.step_wise_rewards
    
    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Preprocess observations (e.g., normalize images).
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Preprocessed observation
        """
        # Convert uint8 images to float32 and normalize
        if isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
            return (obs / 255.0).astype(np.float32)
        return obs
    
    def _sample_random_actions(self, envs: gym.vector.VectorEnv, num_envs: int) -> np.ndarray:
        """Sample random actions from the action space."""
        return np.array([envs.single_action_space.sample() for _ in range(num_envs)])
    
    def _sample_zero_actions(self, envs: gym.vector.VectorEnv, num_envs: int) -> np.ndarray:
        """Sample all-zeros actions within valid action space bounds."""
        action_space = envs.single_action_space
        action_shape = action_space.shape
        
        # If action space has bounds, check if zeros are valid
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            low = action_space.low
            high = action_space.high
            
            # Check if zero is within bounds for each dimension
            zero_valid = np.all((0.0 >= low) & (0.0 <= high))
            
            if zero_valid:
                # Use actual zeros if they're valid, with correct dtype
                zero_actions = np.zeros((num_envs,) + action_shape, dtype=action_space.dtype)
                
                # Double check with action_space.contains()
                test_action = zero_actions[0]
                contains_check = action_space.contains(test_action)
                if not contains_check:
                    # Fallback to middle of action space
                    middle_action = (low + high) / 2.0
                    zero_actions = np.tile(middle_action, (num_envs, 1))
            else:
                # Use middle of action space instead
                middle_action = (low + high) / 2.0
                zero_actions = np.tile(middle_action, (num_envs, 1))
        else:
            # No bounds information, use zeros
            zero_actions = np.zeros((num_envs,) + action_shape, dtype=action_space.dtype)
        
        return zero_actions
    
    def _get_model_actions(self, obs: np.ndarray, prior_actions: np.ndarray, 
                          prior_rewards: np.ndarray, exploration_noise: float,
                          envs: gym.vector.VectorEnv) -> np.ndarray:
        """
        Get actions from the model with optional exploration noise.
        
        Args:
            obs: Current observations
            prior_actions: Previous actions (for recurrent models)
            prior_rewards: Previous rewards (for recurrent models)
            exploration_noise: Noise standard deviation
            envs: Environment for action space bounds
            
        Returns:
            Actions to take
        """
        # Get base actions from model
        try:
            if self.model_interface.has_hidden_state:
                # Recurrent model - pass additional context
                actions = self.model_interface.predict(
                    obs, 
                    prior_action=prior_actions[0],  # Use first env's prior action
                    prior_reward=prior_rewards[0]   # Use first env's prior reward
                )
            else:
                # Feedforward model - pass observation for first environment
                actions = self.model_interface.predict(obs[0])  # Extract first env's observation
        except Exception as e:
            print(f"    Error in model_interface.predict: {e}")
            print(f"    model_interface: {self.model_interface}")
            print(f"    obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            raise
        
        # Expand actions to match number of environments (for vectorized envs)
        num_envs = obs.shape[0]
        if len(actions.shape) == 1:  # Single action vector
            actions = np.tile(actions, (num_envs, 1))  # Shape: (num_envs, action_dim)
        
        # Add exploration noise if requested
        if exploration_noise > 0.0:
            action_scale = self.model_interface.get_action_scale()
            noise = np.random.normal(0, action_scale * exploration_noise, actions.shape)
            actions = actions + noise
        
        # Clip to action space bounds
        actions = np.clip(actions, envs.single_action_space.low, envs.single_action_space.high)
        
        return actions
    
    def _create_episode_directories(self, env_uuids: List[str], save_path: str) -> None:
        """Create directories for saving episode data."""
        for env_uuid in env_uuids:
            episode_save_path = Path(save_path) / env_uuid
            episode_save_path.mkdir(parents=True, exist_ok=True)
            
            # Save episode metadata
            metadata_path = episode_save_path / "episode_metadata.json"
            with open(metadata_path, 'w') as f:
                metadata = vars(self.env_args) if self.env_args else {}
                json.dump(metadata, f, indent=2)
    
    def _save_step_data(self, actions: np.ndarray, rewards: np.ndarray, 
                       env_uuids: List[str], save_path: str,
                       observations: np.ndarray = None,
                       terminations: np.ndarray = None, 
                       truncations: np.ndarray = None,
                       infos: Dict = None) -> None:
        """Save step-level data for detailed analysis."""
        if not save_path:
            return
            
        # Save basic action/reward data
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            self.episode_data.append([action.tolist(), float(reward)])
        
        # Handle special saving for optomech environments with state info
        env_id = getattr(self.env_args, 'env_id', '')
        write_env_state = getattr(self.env_args, 'write_env_state_info', False)
        record_env_state = getattr(self.env_args, 'record_env_state_info', False)
        
        if (env_id == "optomech-v1") and write_env_state and record_env_state:
            # Debug: Check structure of infos
            # print(f"DEBUG: infos type: {type(infos)}")
            # print(f"DEBUG: len(infos): {len(infos) if hasattr(infos, '__len__') else 'No len'}")
            
            # Handle vector environment infos - should be a list of dicts, one per env
            if isinstance(infos, dict):
                # Single environment case - convert to list
                infos_list = [infos]
            else:
                # Multiple environments case 
                infos_list = infos
                
            # DEBUG: print(f"infos_list length: {len(infos_list)}")
            # DEBUG: if len(infos_list) > 0:
            #     print(f"infos_list[0] type: {type(infos_list[0])}")
            #     if hasattr(infos_list[0], 'keys'):
            #         print(f"infos_list[0] keys: {list(infos_list[0].keys())}")
            
            # Match the exact approach from rollout.py - use infos directly 
            # zip over each environment and save the state information.
            for i, (action,
                    next_ob,
                    reward,
                    termination,
                    truncation,
                    info) in enumerate(zip(actions,
                                            observations if observations is not None else [None] * len(actions),
                                            rewards,
                                            terminations if terminations is not None else [None] * len(actions),
                                            truncations if truncations is not None else [None] * len(actions),
                                            infos_list)):

                # Exactly match rollout.py's info structure
                info["step_index"] = self.global_step
                info["reward"] = reward
                info["terminated"] = termination if termination is not None else False
                info["truncated"] = truncation if truncation is not None else False
                info["action"] = action
                info["observation"] = next_ob

                episode_save_path = os.path.join(
                            save_path,
                            env_uuids[i],
                        )

                # Save the info dictionary exactly like rollout.py
                path = os.path.join(
                    episode_save_path,
                    'step_' + str(self.global_step) + '.pkl'
                )
                with open(path, 'wb') as f:
                    pickle.dump(info, f)

    def _handle_episode_completion(self, infos: Dict, env_uuids: List[str], 
                                  save_path: Optional[str]) -> None:
        """Handle episode completion and save results."""
        for info in infos["final_info"]:
            if "episode" in info:
                episode_return = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                # Convert numpy types to Python types for formatting
                episode_return_val = float(episode_return) if hasattr(episode_return, 'item') else episode_return
                episode_length_val = int(episode_length) if hasattr(episode_length, 'item') else episode_length
                
                print(f"🎯 Episode {len(self.episodic_returns) + 1} complete: "
                      f"Return = {episode_return_val:.3f}, Length = {episode_length_val}")
                
                self.episodic_returns.append(episode_return_val)
                
                # Save step-wise rewards for this episode (use first environment for now)
                if len(self.current_episode_rewards) > 0 and len(self.current_episode_rewards[0]) > 0:
                    self.step_wise_rewards.append(self.current_episode_rewards[0].copy())
                    # Reset current episode tracking for next episode
                    self.current_episode_rewards = [[] for _ in range(len(self.current_episode_rewards))]
                
                # Save episode data if path provided  
                if save_path and self.episode_data:
                    # Use same directory structure as step data (always use save_path for consistency)
                    episode_save_path = Path(save_path) / env_uuids[0]
                    episode_save_path.mkdir(parents=True, exist_ok=True)
                    
                    data_file = episode_save_path / f"episode_{len(self.episodic_returns)}.json"
                    with open(data_file, 'w') as f:
                        json.dump(self.episode_data, f)
                    
                    # Save episode metadata for render_history compatibility
                    # Convert env_args to a dictionary, handling nested namespaces
                    env_config = {}
                    for key, value in vars(self.env_args).items():
                        if isinstance(value, Namespace):
                            # Handle nested namespace
                            env_config.update(vars(value))
                        else:
                            env_config[key] = value
                    
                    metadata = {
                        **env_config,  # Include full environment configuration
                        **self.env_metadata,  # Add computed environment metadata (frames_per_decision, etc.)
                        # Override with episode-specific values
                        "episode_length": episode_length_val,
                        "episode_return": episode_return_val,
                        "save_path": str(episode_save_path),
                    }
                    metadata_file = episode_save_path / "episode_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f)
                
                # Reset episode data for next episode
                self.episode_data = []

    def _render_rollout_results(self, save_path: str) -> None:
        """
        Render rollout results using the render_history module.
        
        Args:
            save_path: Base path where episode data was saved
        """
        try:
            # Import the render_history module
            import sys
            from pathlib import Path
            
            # Add the optomech directory to path for render_history import
            optomech_dir = Path(__file__).parent
            if str(optomech_dir) not in sys.path:
                sys.path.insert(0, str(optomech_dir))
            
            from render_history import cli_main
            from argparse import Namespace
            
            # Get state info directory if available, otherwise use regular save path
            # NOTE: Step files (.pkl) are always saved to save_path, not state_save_dir
            # So for rendering, we need to look where the step files actually are
            state_save_dir = getattr(self.env_args, 'state_info_save_dir', None)
            episode_dir = save_path  # Always use save_path where step files are located
            
            # Check if episode data exists
            episode_path = Path(episode_dir)
            if not episode_path.exists():
                print(f"❌ Episode directory not found: {episode_path}")
                return
                
            # Look for pickle files to verify we have data to render
            pickle_files = list(episode_path.glob("**/*.pkl"))
            if not pickle_files:
                print(f"❌ No pickle files found in {episode_path}")
                return
            
            print(f"🎨 Found {len(pickle_files)} step files to render")
            print(f"📁 Rendering from: {episode_path}")
            
            # Create render flags based on environment configuration
            render_flags = Namespace(
                episode_info_dir=str(episode_path),
                render_dpi=getattr(self.env_args, 'render_dpi', 400),
                render_mode=getattr(self.env_args, 'render_mode', 'simple'),
                render_interval=getattr(self.env_args, 'render_interval', 1),
                log_scale_images=getattr(self.env_args, 'log_scale_images', False)
            )
            
            # Call the render_history CLI function
            print("🎨 Starting rendering process...")
            cli_main(render_flags)
            
            # Check if renders were created
            renders_dir = episode_path / "renders"
            if renders_dir.exists():
                render_files = list(renders_dir.glob("*.png"))
                gif_files = list(renders_dir.glob("*.gif"))
                
                print(f"✅ Rendering complete!")
                print(f"🖼️  Created {len(render_files)} image files")
                print(f"🎬 Created {len(gif_files)} gif files")
                print(f"📁 Renders saved to: {renders_dir}")
            else:
                print("⚠️  Warning: Renders directory not found after rendering")
                
        except ImportError as e:
            print(f"❌ Could not import render_history module: {e}")
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# CONFIGURATION AND CLI
# =============================================================================

@dataclass
class RolloutArgs:
    """Configuration arguments for universal rollout evaluation."""
    
    # Model configuration
    model_path: Optional[str] = None
    """Path to the trained model file (None for random policy)"""
    
    model_type: str = "auto"
    """Type of model: 'sml', 'rl', 'ddpg', 'impala', 'custom', or 'auto' for detection"""
    
    # Evaluation configuration
    num_episodes: int = 10
    """Number of episodes to evaluate"""
    
    exploration_noise: float = 0.0
    """Standard deviation of Gaussian noise added to actions (0.0 = no noise)"""
    
    random_policy: bool = False
    """Use random actions instead of model predictions"""
    
    zero_policy: bool = False
    """Use all-zeros actions instead of model predictions"""
    
    # Environment configuration
    env_id: str = "optomech-v1"
    """Environment identifier"""
    
    env_vars_path: Optional[str] = None
    """Path to JSON file containing environment configuration"""
    
    num_envs: int = 1
    """Number of parallel environments"""
    
    subproc_env: bool = False
    """Use SubprocVectorEnv for true parallelism"""
    
    async_env: bool = False
    """Use AsyncVectorEnv for asynchronous execution"""
    
    # Saving configuration
    save_path: Optional[str] = None
    """Directory to save evaluation results and episode data"""
    
    save_episodes: bool = False
    """Save detailed episode data for analysis"""
    
    render: bool = False
    """Enable post-rollout rendering of saved episodes"""
    
    # Environment-specific arguments (will be passed through)
    max_episode_steps: int = 250
    """Maximum steps per episode"""
    
    incremental_control: bool = False
    """Enable incremental control mode in the environment"""
    
    seed: Optional[int] = None
    """Random seed for reproducibility"""
    
    # Device configuration
    device: str = "auto"
    """Device to use: 'cuda', 'mps', 'cpu', or 'auto' for automatic selection"""


def load_environment_config(env_vars_path: str) -> Namespace:
    """
    Load environment configuration from JSON file.
    
    Args:
        env_vars_path: Path to JSON configuration file
        
    Returns:
        Namespace object with environment configuration
    """
    print(f"📁 Loading environment config from: {env_vars_path}")
    with open(env_vars_path, 'r') as f:
        config = json.load(f)
    
    # Check if this is a job config with environment_flags
    if 'environment_flags' in config:
        environment_flags = config['environment_flags']
        print(f"✅ Found {len(environment_flags)} environment flags")
        
        # Import the configuration merger
        import sys
        from pathlib import Path
        optomech_path = Path(__file__).parent / 'optomech'
        sys.path.insert(0, str(optomech_path.parent))
        from optomech.env_config import merge_config_with_flags
        
        # Use merge_config_with_flags to properly parse CLI-style flags
        env_args = merge_config_with_flags(
            config=None,  # Use default config
            flags_list=environment_flags,
            # Add any additional config from the JSON file
            **{k: v for k, v in config.items() if k != 'environment_flags'},
            # Ensure state recording is enabled for step file saving
            record_env_state_info=True,
            write_env_state_info=True,
        )
        return env_args
    else:
        # Simple dictionary conversion for direct parameter configs
        return Namespace(**config)


def main(args: RolloutArgs) -> None:
    """
    Main entry point for universal rollout evaluation.
    
    Args:
        args: Configuration arguments
    """
    print("🚀 Universal Optomech Rollout Evaluation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        print(f"🎲 Setting random seed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load environment configuration
    if args.env_vars_path:
        env_args = load_environment_config(args.env_vars_path)
        # Override with CLI arguments
        for key, value in vars(args).items():
            if value is not None and key != 'env_vars_path':
                setattr(env_args, key, value)
    else:
        env_args = Namespace(**vars(args))
    
    # Setup device
    if args.device == "auto":
        device = None  # Let the rollout engine auto-select
    else:
        device = torch.device(args.device)
        print(f"🔧 Using specified device: {device}")
    
    # Validate policy flags
    if args.zero_policy and args.random_policy:
        raise ValueError("Cannot enable both --zero_policy and --random_policy")
    
    # Create model interface
    model_interface = None
    if args.model_path and not args.random_policy and not args.zero_policy:
        print(f"🧠 Loading model: {args.model_path}")
        model_interface = create_model_interface(
            args.model_path, 
            args.model_type, 
            device or torch.device('cpu')
        )
        model_interface.load_model()
    elif args.zero_policy:
        print("🔷 Using zero policy (all-zeros actions)")
    elif args.random_policy:
        print("🎲 Using random policy (no model loaded)")
    else:
        print("⚠️  No model specified - using random policy")
        args.random_policy = True
    
    # Create rollout engine
    engine = UniversalRolloutEngine(
        model_interface=model_interface,
        env_args=env_args,
        device=device
    )
    
    # Run evaluation
    episodic_returns, step_wise_rewards = engine.run_rollout(
        num_episodes=args.num_episodes,
        exploration_noise=args.exploration_noise,
        save_path=args.save_path,
        save_episodes=args.save_episodes,
        random_policy=args.random_policy,
        zero_policy=args.zero_policy
    )
    
    # Save summary results
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = save_path / "rollout_config.json"
        with open(config_file, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Save results
        results_file = save_path / "rollout_results.json"
        results = {
            "episodic_returns": episodic_returns.tolist() if hasattr(episodic_returns, 'tolist') else list(episodic_returns),
            "mean_return": float(np.mean(episodic_returns)),
            "std_return": float(np.std(episodic_returns)),
            "min_return": float(np.min(episodic_returns)),
            "max_return": float(np.max(episodic_returns)),
            "num_episodes": len(episodic_returns)
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {save_path}")


if __name__ == "__main__":
    args = tyro.cli(RolloutArgs)
    main(args)
