"""
World Model Inference Agent

This module provides an inference wrapper for trained world models, enabling them to
be used as agents for rollouts in the optomech environment.

The agent splits the model into:
1. State Estimator: Encoder + LSTM (produces state estimate from observation)
2. Decoder: Action embedding + State fusion + Decoder (produces next observation prediction)

Action Selection Strategies:
- 'best_guess': Evaluates random actions and selects the one that minimizes MSE with target
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import optomech environment
try:
    import gymnasium as gym
    from optomech import optomech_gym
except ImportError:
    print("⚠️  Warning: Could not import optomech. Install with: pip install -e optomech/")


@dataclass
class WorldModelAgentConfig:
    """Configuration for WorldModelAgent rollouts"""
    # Model settings
    model_path: str = None  # Path to trained world model checkpoint
    action_selection: str = "best_guess"  # Action selection strategy: "best_guess" or "differentiation"
    num_action_samples: int = 100  # Number of random actions to evaluate per step (best_guess mode)
    num_action_seeds: int = 10  # Number of random action initializations to try (differentiation mode)
    action_opt_steps: int = 50  # Number of gradient optimization steps (differentiation mode)
    action_learning_rate: float = 0.01  # Learning rate for action optimization (differentiation mode)
    action_init_zero: bool = False  # Initialize actions at zero for optimization (default: random init)
    warmup_steps: int = 0  # Number of random action steps before using action selection strategy
    device: str = "auto"  # Device: auto, cuda, mps, cpu
    show_model_internals: bool = False  # Show visualization of model predictions
    debug_predictions: bool = False  # Print debug information about prediction tensor values
    
    # Environment settings
    env_id: str = "optomech-v1"
    object_type: str = "single"
    ao_interval_ms: float = 5.0
    control_interval_ms: float = 5.0
    frame_interval_ms: float = 5.0
    decision_interval_ms: float = 5.0
    num_atmosphere_layers: int = 0
    aperture_type: str = "elf"
    focal_plane_image_size_pixels: int = 512
    observation_mode: str = "image_only"
    command_secondaries: bool = True
    incremental_control: bool = False
    init_differential_motion: bool = True
    model_wind_diff_motion: bool = True
    num_envs: int = 1
    reward_function: str = "align"
    dataset: bool = True
    max_episode_steps: int = 100
    
    # Rollout settings
    num_episodes: int = 1
    render: bool = False
    save_rollout: bool = False
    rollout_save_path: str = "rollouts"
    viz_interval: int = 1  # Generate visualization every N steps (when window is full)
    seed: int = 42


class WorldModelAgent:
    """
    Inference wrapper for world models that enables their use as agents.
    
    The agent decomposes the world model into:
    - State Estimator: observation -> state estimate (via encoder + LSTM)
    - Decoder: (state estimate, action) -> predicted next observation
    
    Actions are selected by evaluating candidate actions and choosing the one
    that produces a predicted next observation closest to the target.
    """
    
    def __init__(
        self,
        model_path: str,
        target_observation: np.ndarray,
        action_selection: str = "best_guess",
        num_action_samples: int = 100,
        num_action_seeds: int = 10,
        action_opt_steps: int = 50,
        action_learning_rate: float = 0.01,
        action_init_zero: bool = False,
        warmup_steps: int = 0,
        device: str = "auto",
        action_dim: int = None,
        input_crop_size: int = None,
        log_scale: bool = False,
        show_model_internals: bool = False,
        debug_predictions: bool = False
    ):
        """
        Initialize the WorldModelAgent.
        
        Args:
            model_path: Path to trained world model checkpoint
            target_observation: Target observation to match (from env.optical_system.target_image)
            action_selection: Strategy for selecting actions ('best_guess' or 'differentiation')
            num_action_samples: Number of random actions to evaluate per step (best_guess mode)
            num_action_seeds: Number of random action initializations to try (differentiation mode)
            action_opt_steps: Number of gradient optimization steps (differentiation mode)
            action_learning_rate: Learning rate for action optimization (differentiation mode)
            action_init_zero: Initialize actions at zero for optimization (default: random init)
            warmup_steps: Number of random action steps before using action selection strategy
            device: Device to run inference on
            action_dim: Action dimension (inferred from model if None)
            input_crop_size: Size to center crop observations (None = no crop)
            log_scale: Whether to apply log-scaling to observations
            show_model_internals: Whether to visualize model predictions during action selection
        """
        self.action_selection = action_selection
        self.num_action_samples = num_action_samples
        self.num_action_seeds = num_action_seeds
        self.action_opt_steps = action_opt_steps
        self.action_learning_rate = action_learning_rate
        self.action_init_zero = action_init_zero
        self.warmup_steps = warmup_steps
        self.device = self._get_device(device)
        self.input_crop_size = input_crop_size
        self.log_scale = log_scale
        self.show_model_internals = show_model_internals
        self.debug_predictions = debug_predictions
        
        # Load model checkpoint
        print(f"🔧 Loading world model from: {model_path}")
        
        # Import WorldModelConfig and make it available for unpickling
        # Checkpoints saved from train_world_model.py have __main__.WorldModelConfig
        try:
            from optomech.world_models.train_world_model import WorldModelConfig as TrainingConfig
            # Make it available as if it were defined in this module's __main__
            import sys
            sys.modules['__main__'].WorldModelConfig = TrainingConfig
        except ImportError:
            pass  # Config import not critical
        
        # Load checkpoint (weights_only=False allows loading pickled config objects)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.config = checkpoint.get('config', None)
        else:
            state_dict = checkpoint
            self.config = None
        
        # Load the full model to extract components
        from models.world_model import WorldModel
        
        # Infer architecture parameters from state dict
        self.action_dim = action_dim or self._infer_action_dim(state_dict)
        self.latent_dim = self._infer_latent_dim(state_dict)
        self.hidden_dim = self._infer_hidden_dim(state_dict)
        self.num_lstm_layers = self._infer_num_lstm_layers(state_dict)
        
        print(f"📊 Model architecture:")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   LSTM layers: {self.num_lstm_layers}")
        
        # Debug: print some state dict keys to understand structure
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')][:3]
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')][:3]
        print(f"   Sample encoder keys: {encoder_keys}")
        print(f"   Sample decoder keys: {decoder_keys}")
        
        # Create a full WorldModel and load the state dict, then extract components
        # This ensures we get the properly wrapped encoder/decoder with bottleneck layers
        from models.world_model import WorldModel
        from models.models import create_model
        
        # Infer architecture from encoder structure
        has_resnet_blocks = any('gn1' in k or 'gn2' in k for k in state_dict.keys())
        arch = 'autoencoder_resnet' if has_resnet_blocks else 'autoencoder_cnn'
        
        print(f"   Detected architecture: {arch}")
        
        # Create a dummy autoencoder to get the structure
        dummy_autoencoder = create_model(
            arch=arch,
            input_channels=1,
            latent_dim=self.latent_dim,
            input_crop_size=None
        )
        
        # Create full WorldModel using the autoencoder
        from models.world_model import create_world_model_from_autoencoder
        full_model = create_world_model_from_autoencoder(
            autoencoder=dummy_autoencoder,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            action_hidden_dim=128,
            freeze_encoder=False,
            freeze_decoder=False
        )
        
        # Load the full state dict
        full_model.load_state_dict(state_dict, strict=True)
        
        # Extract the components (encoder/decoder are already wrapped with bottleneck)
        self.encoder = full_model.encoder
        self.decoder = full_model.decoder
        self.lstm = full_model.lstm
        self.fusion_mlp = full_model.fusion_mlp
        self.action_encoder = full_model.action_encoder
        
        # Move components to device and set to eval mode
        self.encoder = self.encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()
        self.lstm = self.lstm.to(self.device).eval()
        self.fusion_mlp = self.fusion_mlp.to(self.device).eval()
        self.action_encoder = self.action_encoder.to(self.device).eval()
        
        # Freeze all model parameters (they stay frozen even for differentiation mode)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.fusion_mlp.parameters():
            param.requires_grad = False
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        
        # Set target observation
        self.set_target_observation(target_observation)
        
        # Initialize LSTM hidden state
        self.hidden = None
        self.reset_hidden_state()
        
        print(f"✅ WorldModelAgent initialized with '{action_selection}' action selection")
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get the appropriate device"""
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)
        print(f"🔧 Using device: {device}")
        return device
    
    def _infer_action_dim(self, state_dict: Dict) -> int:
        """Infer action dimension from state dict"""
        # Action encoder first layer
        if 'action_encoder.0.weight' in state_dict:
            return state_dict['action_encoder.0.weight'].shape[1]
        raise ValueError("Could not infer action_dim from state dict")
    
    def _infer_latent_dim(self, state_dict: Dict) -> int:
        """Infer latent dimension from state dict"""
        # LSTM input size
        if 'lstm.weight_ih_l0' in state_dict:
            return state_dict['lstm.weight_ih_l0'].shape[1]
        raise ValueError("Could not infer latent_dim from state dict")
    
    def _infer_hidden_dim(self, state_dict: Dict) -> int:
        """Infer LSTM hidden dimension from state dict"""
        if 'lstm.weight_hh_l0' in state_dict:
            return state_dict['lstm.weight_hh_l0'].shape[1]
        raise ValueError("Could not infer hidden_dim from state dict")
    
    def _infer_num_lstm_layers(self, state_dict: Dict) -> int:
        """Infer number of LSTM layers from state dict"""
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith('lstm.weight_ih_l'):
                layer_num = int(key.split('_l')[1].split('.')[0])
                num_layers = max(num_layers, layer_num + 1)
        return max(1, num_layers)
    
    def set_target_observation(self, target_observation: np.ndarray):
        """
        Set the target observation to match.
        Applies the same preprocessing as training: normalize, crop, log-scale.
        
        Args:
            target_observation: Target observation array (H, W) or (C, H, W)
        """
        # Store raw target before preprocessing for visualization
        self.raw_target_observation = target_observation.copy() if isinstance(target_observation, np.ndarray) else target_observation.clone()
        
        # Convert to tensor
        if isinstance(target_observation, np.ndarray):
            target_observation = torch.from_numpy(target_observation).float()
        
        # Normalize from [0, 65535] to [0, 1] (auto-detect range)
        max_val = target_observation.max()
        if max_val > 256:
            target_observation = target_observation / 65535.0
        elif max_val > 1.0:
            target_observation = target_observation / 255.0
        # else: already in [0, 1]
        
        # Ensure shape is [C, H, W]
        if target_observation.ndim == 2:
            target_observation = target_observation.unsqueeze(0)  # [1, H, W]
        
        # Crop target to match encoder output size
        # (Input observations are NOT cropped here - encoder crops them internally)
        if self.input_crop_size:
            from utils.transforms import center_crop_transform
            target_observation = center_crop_transform(target_observation, self.input_crop_size)
        
        # Apply log-scaling if enabled (log(1+x))
        if self.log_scale:
            target_observation = torch.log1p(target_observation)
        
        # Add batch dimension: [1, C, H, W]
        self.target_observation = target_observation.unsqueeze(0).to(self.device)
        print(f"🎯 Target observation set: {self.target_observation.shape}")
        if self.input_crop_size:
            print(f"   Target cropped to: {self.input_crop_size}x{self.input_crop_size}")
        if self.log_scale:
            print(f"   Log-scaled: True")
    
    def reset_hidden_state(self, batch_size: int = 1):
        """Reset LSTM hidden state"""
        self.hidden = (
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=self.device),
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=self.device)
        )
        self.step_counter = 0  # Reset step counter for warmup tracking
    
    def compute_state_estimate(self, observation: np.ndarray, return_lstm_out: bool = False) -> torch.Tensor:
        """
        Compute latent encoding and LSTM state from observation.
        
        In the new architecture, we need both encoder output and LSTM output for prediction.
        This method stores both internally and optionally returns them.
        
        Args:
            observation: Current observation (H, W) or (C, H, W)
            return_lstm_out: If True, return (latent, lstm_out) tuple
            
        Returns:
            latent: Encoder latent tensor [1, latent_dim]
            lstm_out: (optional) LSTM output tensor [1, hidden_dim] if return_lstm_out=True
        """
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        # Normalize from [0, 65535] to [0, 1] (auto-detect range)
        max_val = observation.max()
        if max_val > 256:
            observation = observation / 65535.0
        elif max_val > 1.0:
            observation = observation / 255.0
        # else: already in [0, 1]
        
        # Ensure shape is [C, H, W]
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        
        # NOTE: Cropping is handled by the encoder, not here!
        # The encoder has input_crop_size built in and will crop during forward pass
        
        # Apply log-scaling if enabled (log(1+x))
        if self.log_scale:
            observation = torch.log1p(observation)
        
        # Add batch dimension: [1, C, H, W]
        observation = observation.unsqueeze(0).to(self.device)
        
        # Store preprocessed observation for visualization
        # Crop to match what encoder will see (encoder crops internally)
        if self.input_crop_size:
            from utils.transforms import center_crop_transform
            self.model_input_observation = center_crop_transform(observation.clone(), self.input_crop_size)
        else:
            self.model_input_observation = observation.clone()
        
        with torch.no_grad():
            # Encode observation (encoder includes bottleneck for ResNet models, outputs flat latent)
            # Encoder will crop internally if input_crop_size is set
            latent = self.encoder(observation)  # [1, latent_dim]
            
            # Store latent for prediction
            self.current_latent = latent
            
            # Add sequence dimension for LSTM
            latent_seq = latent.unsqueeze(1)  # [1, 1, latent_dim]
            
            # LSTM forward pass
            lstm_out, self.hidden = self.lstm(latent_seq, self.hidden)  # [1, 1, hidden_dim]
            
            # Flatten LSTM output
            lstm_out_flat = lstm_out.squeeze(1)  # [1, hidden_dim]
            
            # Store LSTM output for prediction
            self.current_lstm_out = lstm_out_flat
        
        if return_lstm_out:
            return latent, lstm_out_flat
    def predict_next_observation(self, action: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """
        Predict next observation given action.
        Uses the stored latent and LSTM outputs from compute_state_estimate().
        
        Args:
            action: Action tensor [1, action_dim]
            enable_grad: If True, allows gradients to flow (for differentiation mode)
            
        Returns:
            next_obs_pred: Predicted next observation [1, C, H, W]
        """
        debug = getattr(self, 'debug_predictions', False)
        
        # Get stored latent and LSTM outputs
        latent = self.current_latent  # [1, latent_dim]
        lstm_out = self.current_lstm_out  # [1, hidden_dim]
        
        if debug:
            print(f"  🔍 DEBUG predict_next_observation:")
            print(f"    latent shape: {latent.shape}, range: [{latent.min():.4f}, {latent.max():.4f}], mean: {latent.mean():.4f}")
            print(f"    lstm_out shape: {lstm_out.shape}, range: [{lstm_out.min():.4f}, {lstm_out.max():.4f}], mean: {lstm_out.mean():.4f}")
            print(f"    action shape: {action.shape}, range: [{action.min():.4f}, {action.max():.4f}], mean: {action.mean():.4f}")
        
        if enable_grad:
            # Enable gradient flow for action optimization
            # Encode action
            action_encoded = self.action_encoder(action)  # [1, latent_dim]
            
            if debug:
                print(f"    action_encoded shape: {action_encoded.shape}, range: [{action_encoded.min():.4f}, {action_encoded.max():.4f}], mean: {action_encoded.mean():.4f}")
            
            # Concatenate: [latent, lstm_out, action_encoded]
            concat_features = torch.cat([latent, lstm_out, action_encoded], dim=-1)  # [1, latent_dim + hidden_dim + latent_dim]
            
            if debug:
                print(f"    concat_features shape: {concat_features.shape}, range: [{concat_features.min():.4f}, {concat_features.max():.4f}], mean: {concat_features.mean():.4f}")
            
            # Fusion MLP
            fused = self.fusion_mlp(concat_features)  # [1, latent_dim]
            
            if debug:
                print(f"    fused shape: {fused.shape}, range: [{fused.min():.4f}, {fused.max():.4f}], mean: {fused.mean():.4f}")
            
            # Decode (decoder includes bottleneck for ResNet models)
            next_obs_pred = self.decoder(fused)  # [1, C, H, W]
            
            if debug:
                print(f"    next_obs_pred shape: {next_obs_pred.shape}, range: [{next_obs_pred.min():.4f}, {next_obs_pred.max():.4f}], mean: {next_obs_pred.mean():.4f}")
        else:
            # Default behavior: no gradients
            with torch.no_grad():
                # Encode action
                action_encoded = self.action_encoder(action)  # [1, latent_dim]
                
                if debug:
                    print(f"    action_encoded shape: {action_encoded.shape}, range: [{action_encoded.min():.4f}, {action_encoded.max():.4f}], mean: {action_encoded.mean():.4f}")
                
                # Concatenate: [latent, lstm_out, action_encoded]
                concat_features = torch.cat([latent, lstm_out, action_encoded], dim=-1)  # [1, latent_dim + hidden_dim + latent_dim]
                
                if debug:
                    print(f"    concat_features shape: {concat_features.shape}, range: [{concat_features.min():.4f}, {concat_features.max():.4f}], mean: {concat_features.mean():.4f}")
                
                # Fusion MLP
                fused = self.fusion_mlp(concat_features)  # [1, latent_dim]
                
                if debug:
                    print(f"    fused shape: {fused.shape}, range: [{fused.min():.4f}, {fused.max():.4f}], mean: {fused.mean():.4f}")
                
                # Decode (decoder includes bottleneck for ResNet models)
                next_obs_pred = self.decoder(fused)  # [1, C, H, W]
                
                if debug:
                    print(f"    next_obs_pred shape: {next_obs_pred.shape}, range: [{next_obs_pred.min():.4f}, {next_obs_pred.max():.4f}], mean: {next_obs_pred.mean():.4f}")
        
        return next_obs_pred
    
    def select_action_best_guess(self) -> np.ndarray:
        """
        Select action using 'best_guess' strategy.
        
        Evaluates random actions and selects the one that minimizes MSE
        between predicted next observation and target observation.
        Only selects an action if it improves upon the current input MSE.
        
        Uses stored latent and LSTM outputs from compute_state_estimate().
        
        Returns:
            best_action: Selected action as numpy array (or zeros if no improvement)
        """
        target = self.target_observation
        
        # Compute baseline MSE between current input observation and target
        input_mse = torch.nn.functional.mse_loss(
            self.model_input_observation, target
        ).item()
        print(f"  📊 Input MSE (baseline): {input_mse:.6f}")
        
        best_action = None
        best_mse = float('inf')
        best_next_obs_pred = None
        
        for i in range(self.num_action_samples):
            # Sample random action (uniform distribution in [-1, 1])
            action = torch.rand(1, self.action_dim, device=self.device) * 2 - 1
            
            # Predict next observation
            next_obs_pred = self.predict_next_observation(action)
            
            # Compute MSE (target and prediction should already be same shape)
            mse = torch.nn.functional.mse_loss(next_obs_pred, target).item()
            
            # Update best action
            if mse < best_mse:
                best_mse = mse
                best_action = action.cpu().numpy()[0]
                best_next_obs_pred = next_obs_pred
        
        print(f"  🎯 Best action MSE: {best_mse:.6f}")
        
        # Only use the best action if it improves upon input MSE
        if best_mse < input_mse:
            print(f"  ✅ Using best action (improvement: {input_mse - best_mse:.6f})")
            selected_action = best_action
        else:
            print(f"  ⏸️  No improvement found, using zero action")
            selected_action = np.zeros(self.action_dim, dtype=np.float32)
            best_next_obs_pred = self.model_input_observation  # Show current obs in viz
        
        # Visualize model internals if requested
        if self.show_model_internals:
            self._visualize_model_internals(
                agent_input_obs=self.current_observation,
                model_input_obs=self.model_input_observation,
                selected_action=selected_action,
                predicted_next_obs=best_next_obs_pred,
                target_obs=target,
                mse=best_mse if best_mse < input_mse else input_mse
            )
        
        # Convert to float32 for environment compatibility
        return selected_action.astype(np.float32)
    
    def select_action_differentiation(self) -> np.ndarray:
        """
        Select action using gradient-based optimization.
        
        Optimizes the action to minimize MSE between predicted next observation
        and target observation by backpropagating through the frozen model.
        Tries multiple random action initializations and selects the best final action.
        Only selects the optimized action if it improves upon the current input MSE.
        
        Uses stored latent and LSTM outputs from compute_state_estimate().
        
        Returns:
            optimized_action: Selected action as numpy array (or zeros if no improvement)
        """
        target = self.target_observation
        
        # Compute baseline MSE between current input observation and target
        input_mse = torch.nn.functional.mse_loss(
            self.model_input_observation, target
        ).item()
        print(f"  📊 Input MSE (baseline): {input_mse:.6f}")
        
        # Track the best action across all seeds
        global_best_mse = float('inf')
        global_best_action = None
        global_best_next_obs_pred = None
        
        # Outer loop: try multiple action initializations
        init_type = "zero" if self.action_init_zero else "random"
        print(f"  🌱 Trying {self.num_action_seeds} {init_type} action seeds...")
        for seed_idx in range(self.num_action_seeds):
            # Initialize action as a learnable parameter
            if self.action_init_zero:
                action = torch.zeros(1, self.action_dim, device=self.device)
            else:
                action = torch.randn(1, self.action_dim, device=self.device) * 0.1  # Scale initialization
            action.requires_grad = True
            
            # Optimizer for the action
            optimizer = torch.optim.Adam([action], lr=self.action_learning_rate)
            
            seed_best_mse = float('inf')
            seed_best_action = None
            seed_best_next_obs_pred = None
            
            # Gradient descent loop for this seed
            for step in range(self.action_opt_steps):
                optimizer.zero_grad()
                
                # Predict next observation with current action (enable gradients)
                next_obs_pred = self.predict_next_observation(action, enable_grad=True)
                
                # Compute MSE loss
                mse_loss = torch.nn.functional.mse_loss(next_obs_pred, target)
                
                # Backpropagate to compute gradients w.r.t. action
                mse_loss.backward()
                
                # Update action
                optimizer.step()
                
                # Clamp action to [-1, 1] range after optimizer step
                with torch.no_grad():
                    action.data = torch.clamp(action.data, -1.0, 1.0)
                
                # Track best action for this seed (with clamping applied)
                current_mse = mse_loss.item()
                if current_mse < seed_best_mse:
                    seed_best_mse = current_mse
                    seed_best_action = action.detach().cpu().numpy()[0]
                    seed_best_next_obs_pred = next_obs_pred.detach()
            
            # Print seed results
            print(f"    Seed {seed_idx+1}/{self.num_action_seeds}: Final MSE = {seed_best_mse:.6f}")
            
            # Update global best if this seed is better
            if seed_best_mse < global_best_mse:
                global_best_mse = seed_best_mse
                global_best_action = seed_best_action
                global_best_next_obs_pred = seed_best_next_obs_pred
        
        print(f"  🎯 Best optimized MSE (across all seeds): {global_best_mse:.6f}")
        
        # Only use the optimized action if it improves upon input MSE
        if global_best_mse < input_mse:
            print(f"  ✅ Using optimized action (improvement: {input_mse - global_best_mse:.6f})")
            selected_action = global_best_action
            # Ensure action is strictly within bounds (safety check)
            selected_action = np.clip(selected_action, -1.0, 1.0)
            best_next_obs_pred = global_best_next_obs_pred
            best_mse = global_best_mse
        else:
            print(f"  ⏸️  No improvement found, using zero action")
            selected_action = np.zeros(self.action_dim, dtype=np.float32)
            best_next_obs_pred = self.model_input_observation  # Show current obs in viz
            best_mse = input_mse
        
        # Visualize model internals if requested
        if self.show_model_internals:
            self._visualize_model_internals(
                agent_input_obs=self.current_observation,
                model_input_obs=self.model_input_observation,
                selected_action=selected_action,
                predicted_next_obs=best_next_obs_pred,
                target_obs=target,
                mse=best_mse
            )
        
        # Convert to float32 for environment compatibility
        return selected_action.astype(np.float32)
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Select action given current observation.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Selected action as numpy array
        """
        # Store current observation for visualization
        self.current_observation = observation
        
        # Compute latent and LSTM state (updates LSTM hidden state and stores latent/lstm_out)
        self.compute_state_estimate(observation)
        
        # Increment step counter
        self.step_counter += 1
        
        # During warmup: return random action without running action selection
        if self.step_counter <= self.warmup_steps:
            random_action = np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)
            print(f"  🔥 Warmup step {self.step_counter}/{self.warmup_steps}: using random action")
            return random_action
        
        # After warmup: use configured action selection strategy
        if self.action_selection == "best_guess":
            return self.select_action_best_guess()
        elif self.action_selection == "differentiation":
            return self.select_action_differentiation()
        else:
            raise NotImplementedError(
                f"Action selection strategy '{self.action_selection}' not implemented. "
                f"Available strategies: 'best_guess', 'differentiation'"
            )
    
    def _visualize_model_internals(
        self,
        agent_input_obs: np.ndarray,
        model_input_obs: torch.Tensor,
        selected_action: np.ndarray,
        predicted_next_obs: torch.Tensor,
        target_obs: torch.Tensor,
        mse: float
    ):
        """
        Visualize model internals during action selection.
        
        Shows: agent input observation, model input observation (preprocessed),
        selected action, predicted next observation, target observation (preprocessed),
        target observation (raw), and residual between prediction and target.
        
        Args:
            agent_input_obs: Raw observation from agent [C, H, W] numpy array
            model_input_obs: Preprocessed observation that goes to model [C, H, W] tensor
            selected_action: Selected action [action_dim] numpy array
            predicted_next_obs: Predicted next observation [1, C, H, W] tensor
            target_obs: Target observation [1, C, H, W] tensor (preprocessed)
            mse: MSE between prediction and target
        """
        # Convert tensors to numpy
        model_input_np = model_input_obs.cpu().detach().numpy()  # [C, H, W]
        pred_np = predicted_next_obs.cpu().detach().squeeze(0).numpy()  # [C, H, W]
        target_np = target_obs.cpu().detach().squeeze(0).numpy()  # [C, H, W]
        
        # If multi-channel, take first channel for visualization
        if pred_np.ndim == 3 and pred_np.shape[0] > 1:
            pred_np = pred_np[0]  # [H, W]
            target_np = target_np[0]  # [H, W]
            model_input_vis = model_input_np[0]  # [H, W]
            agent_input_vis = agent_input_obs[0] if agent_input_obs.ndim == 3 else agent_input_obs
        else:
            # Single channel - squeeze channel dimension
            pred_np = pred_np.squeeze()  # [H, W]
            target_np = target_np.squeeze()  # [H, W]
            model_input_vis = model_input_np.squeeze()  # [H, W]
            agent_input_vis = agent_input_obs.squeeze()  # [H, W]
        
        # Compute residual
        residual = pred_np - target_np
        
        # Get raw target image for comparison
        raw_target = self.raw_target_observation
        if raw_target.ndim == 3 and raw_target.shape[0] > 1:
            raw_target_vis = raw_target[0]
        else:
            raw_target_vis = raw_target.squeeze()
        
        # Create figure with 7 subplots
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        
        # 1. Agent input observation (raw)
        im0 = axes[0].imshow(agent_input_vis, cmap='gray')
        axes[0].set_title('Agent Input Ob')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # 2. Model input observation (preprocessed)
        im1 = axes[1].imshow(model_input_vis, cmap='gray')
        axes[1].set_title('Model Input Ob')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # 3. Selected action (bar plot)
        action_indices = np.arange(len(selected_action))
        axes[2].bar(action_indices, selected_action)
        axes[2].set_title('Selected Action')
        axes[2].set_xlabel('Action Dimension')
        axes[2].set_ylabel('Value')
        axes[2].set_ylim([-1.1, 1.1])
        axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Predicted next observation
        im3 = axes[3].imshow(pred_np, cmap='gray')
        axes[3].set_title('Predicted Next Obs')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
        
        # 5. Target observation (preprocessed: normalized, cropped, log-scaled)
        im4 = axes[4].imshow(target_np, cmap='gray')
        axes[4].set_title('Target Obs (preprocessed)')
        axes[4].axis('off')
        plt.colorbar(im4, ax=axes[4], fraction=0.046)
        
        # 6. Target observation (raw, before preprocessing)
        im5 = axes[5].imshow(raw_target_vis, cmap='gray')
        axes[5].set_title('Target Obs (raw)')
        axes[5].axis('off')
        plt.colorbar(im5, ax=axes[5], fraction=0.046)
        
        # 7. Residual (prediction - target)
        im6 = axes[6].imshow(residual, cmap='RdBu_r', vmin=-residual.std()*3, vmax=residual.std()*3)
        axes[6].set_title(f'Residual (MSE: {mse:.6f})')
        axes[6].axis('off')
        plt.colorbar(im6, ax=axes[6], fraction=0.046)
        
        plt.tight_layout()
        plt.show(block=True)  # Blocking display
    
    def reset(self):
        """Reset agent state (LSTM hidden state)"""
        self.reset_hidden_state()


def create_rollout_visualization(
    obs_window: list,
    actions_window: list,
    predictions_window: list,
    targets_window: list,
    lstm_outs_window: list,
    episode_num: int,
    step_num: int,
    save_path: str = None
):
    """
    Create a visualization of the rollout similar to training visualization.
    
    Visualizes a 4-step window showing:
    - Row 0: Target Residual (|prediction - target|)
    - Row 1: Prediction Residual (|prior_prediction - current_obs|)
    - Row 2: Input observation
    - Row 3: Action vector
    - Row 4: Hidden state (LSTM output)
    - Row 5: Target observation
    - Row 6: Predicted next observation
    
    Args:
        obs_window: List of 4 observation tensors [C, H, W]
        actions_window: List of 4 action arrays
        predictions_window: List of 4 prediction tensors [1, C, H, W]
        targets_window: List of 4 target tensors [1, C, H, W]
        lstm_outs_window: List of 4 LSTM output tensors [1, hidden_dim]
        episode_num: Episode number
        step_num: Current step number
        save_path: Path to save the visualization (optional)
    """
    num_timesteps = len(obs_window)
    
    # Convert to numpy and compute residuals
    obs_list = []
    target_list = []
    pred_list = []
    target_residual_list = []
    pred_residual_list = []
    
    for i in range(num_timesteps):
        # Observations
        obs_np = obs_window[i].cpu().detach().numpy() if isinstance(obs_window[i], torch.Tensor) else obs_window[i]
        if obs_np.ndim == 3:
            obs_np = obs_np[0]  # Take first channel if multi-channel
        obs_list.append(obs_np)
        
        # Targets
        target_np = targets_window[i].cpu().detach().squeeze(0).numpy()
        if target_np.ndim == 3:
            target_np = target_np[0]
        target_list.append(target_np)
        
        # Predictions
        pred_np = predictions_window[i].cpu().detach().squeeze(0).numpy()
        if pred_np.ndim == 3:
            pred_np = pred_np[0]
        pred_list.append(pred_np)
        
        # Target residuals (prediction - target)
        target_residual_list.append(np.abs(pred_np - target_np))
        
        # Prediction residuals (prior prediction - current obs)
        if i > 0:
            prior_pred = pred_list[i-1]
            pred_residual_list.append(np.abs(prior_pred - obs_np))
        else:
            # For first timestep, no prior prediction exists
            pred_residual_list.append(None)
    
    # Compute global min/max for consistent scaling
    all_obs_data = np.concatenate([np.array(obs_list), np.array(target_list), np.array(pred_list)])
    global_vmin = all_obs_data.min()
    global_vmax = all_obs_data.max()
    
    # Create figure: 7 rows × num_timesteps columns
    fig, axes = plt.subplots(7, num_timesteps, figsize=(4*num_timesteps, 28))
    fig.suptitle(f'Rollout Episode {episode_num} - Steps {step_num-num_timesteps+1} to {step_num}', 
                 fontsize=16, fontweight='bold')
    
    for t in range(num_timesteps):
        col = t
        actual_step = step_num - num_timesteps + t + 1
        
        # Row 0: Target Residual (prediction - target)
        target_residual_img = target_residual_list[t]
        im0 = axes[0, col].imshow(target_residual_img, cmap='hot', vmin=0, vmax=target_residual_img.max())
        axes[0, col].set_title(f'Step {actual_step}\nTarget Residual (MAE={target_residual_img.mean():.6f})', fontsize=10)
        axes[0, col].axis('off')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        
        # Row 1: Prediction Residual (prior prediction - current obs)
        if pred_residual_list[t] is not None:
            pred_residual_img = pred_residual_list[t]
            im1 = axes[1, col].imshow(pred_residual_img, cmap='hot', vmin=0, vmax=pred_residual_img.max())
            axes[1, col].set_title(f'Pred Residual (MAE={pred_residual_img.mean():.6f})', fontsize=10)
            axes[1, col].axis('off')
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
        else:
            axes[1, col].text(0.5, 0.5, 'No prior\nprediction', ha='center', va='center', fontsize=10)
            axes[1, col].axis('off')
        
        # Row 2: Input observation
        obs_img = obs_list[t]
        im2 = axes[2, col].imshow(obs_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[2, col].set_title('Input Obs', fontsize=10)
        axes[2, col].axis('off')
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046)
        
        # Row 3: Action vector (as heatmap)
        action_vec = actions_window[t]
        action_2d = action_vec.reshape(1, -1)  # [1, action_dim]
        im3 = axes[3, col].imshow(action_2d, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[3, col].set_title(f'Action ({len(action_vec)} dims)', fontsize=10)
        axes[3, col].axis('off')
        plt.colorbar(im3, ax=axes[3, col], fraction=0.046)
        
        # Row 4: Hidden state (LSTM output)
        if lstm_outs_window[t] is not None:
            hidden_vec = lstm_outs_window[t].cpu().detach().numpy()
            if hidden_vec.ndim > 1:
                hidden_vec = hidden_vec.flatten()
            hidden_size = len(hidden_vec)
            # Find factors close to square root for better visualization
            h_rows = int(np.sqrt(hidden_size))
            while hidden_size % h_rows != 0 and h_rows > 1:
                h_rows -= 1
            h_cols = hidden_size // h_rows
            hidden_2d = hidden_vec[:h_rows*h_cols].reshape(h_rows, h_cols)
            
            im4 = axes[4, col].imshow(hidden_2d, cmap='coolwarm', aspect='auto')
            axes[4, col].set_title(f'Hidden State ({h_rows}x{h_cols})', fontsize=10)
            axes[4, col].axis('off')
            plt.colorbar(im4, ax=axes[4, col], fraction=0.046)
        else:
            axes[4, col].text(0.5, 0.5, 'No hidden\nstate', ha='center', va='center')
            axes[4, col].axis('off')
        
        # Row 5: Target next observation
        target_img = target_list[t]
        im5 = axes[5, col].imshow(target_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[5, col].set_title('Target Next Obs', fontsize=10)
        axes[5, col].axis('off')
        plt.colorbar(im5, ax=axes[5, col], fraction=0.046)
        
        # Row 6: Model prediction
        pred_img = pred_list[t]
        im6 = axes[6, col].imshow(pred_img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[6, col].set_title('Prediction', fontsize=10)
        axes[6, col].axis('off')
        plt.colorbar(im6, ax=axes[6, col], fraction=0.046)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Saved rollout visualization to: {save_path}")
    
    plt.close(fig)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_value(key: str, default=None, args=None, config_dict=None, cli_provided=None):
    """
    Get configuration value with priority: CLI > config file > default
    
    Args:
        key: Configuration key
        default: Default value
        args: Parsed command-line arguments
        config_dict: Configuration dictionary from file
        cli_provided: Set of CLI-provided argument names
    """
    # Priority 1: Command-line argument (if explicitly provided)
    if cli_provided and key in cli_provided and args and hasattr(args, key):
        return getattr(args, key)
    
    # Priority 2: Config file
    if config_dict and key in config_dict:
        return config_dict[key]
    
    # Priority 3: Default value (or CLI default if not in cli_provided)
    if args and hasattr(args, key):
        return getattr(args, key)
    
    return default


def main():
    """Main function for testing the WorldModelAgent"""
    parser = argparse.ArgumentParser(description="World Model Agent Rollout")
    
    # Config file
    parser.add_argument("--config", type=str, default="optomech_config.json",
                       help="Path to JSON config file")
    
    # Model settings
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained world model checkpoint")
    parser.add_argument("--action-selection", type=str, default="best_guess",
                       help="Action selection strategy: best_guess, differentiation")
    parser.add_argument("--num-action-samples", type=int, default=100,
                       help="Number of random actions to evaluate per step (best_guess mode)")
    parser.add_argument("--num-action-seeds", type=int, default=10,
                       help="Number of random action initializations to try (differentiation mode)")
    parser.add_argument("--action-opt-steps", type=int, default=50,
                       help="Number of gradient optimization steps (differentiation mode)")
    parser.add_argument("--action-learning-rate", type=float, default=0.01,
                       help="Learning rate for action optimization (differentiation mode)")
    parser.add_argument("--action-init-zero", action="store_true",
                       help="Initialize actions at zero for optimization (default: random)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                       help="Number of random action steps before using action selection strategy")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--show-model-internals", action="store_true",
                       help="Show visualization of model predictions during action selection")
    parser.add_argument("--debug-predictions", action="store_true",
                       help="Print debug information about prediction tensor values")
    
    # Environment settings
    parser.add_argument("--env-id", type=str, default="optomech-v1")
    parser.add_argument("--object-type", type=str, default="single")
    parser.add_argument("--ao-interval-ms", type=float, default=5.0)
    parser.add_argument("--control-interval-ms", type=float, default=5.0)
    parser.add_argument("--frame-interval-ms", type=float, default=5.0)
    parser.add_argument("--decision-interval-ms", type=float, default=5.0)
    parser.add_argument("--num-atmosphere-layers", type=int, default=0)
    parser.add_argument("--aperture-type", type=str, default="elf")
    parser.add_argument("--focal-plane-image-size-pixels", type=int, default=512)
    parser.add_argument("--observation-mode", type=str, default="image_only")
    parser.add_argument("--command-secondaries", action="store_true", default=False)
    parser.add_argument("--incremental-control", action="store_true", default=False)
    parser.add_argument("--init-differential-motion", action="store_true", default=False)
    parser.add_argument("--model-wind-diff-motion", action="store_true", default=False)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--reward-function", type=str, default="align")
    parser.add_argument("--dataset", action="store_true", default=False)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    
    # Rollout settings
    parser.add_argument("--num-episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                       help="Render environment")
    parser.add_argument("--save-rollout", action="store_true",
                       help="Save rollout data")
    parser.add_argument("--rollout-save-path", type=str, default="rollouts",
                       help="Path to save rollout data")
    parser.add_argument("--viz-interval", type=int, default=1,
                       help="Generate rollout visualization every N steps (when window is full)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Determine which arguments were provided via CLI
    cli_provided = set()
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            cli_provided.add(key)
    
    # Load config file if provided
    config_dict = {}
    if args.config and Path(args.config).exists():
        print(f"📋 Loading config from: {args.config}")
        config_dict = load_config(args.config)
    
    # Helper function for getting values with priority
    def get_val(key, default=None):
        return get_value(key, default, args, config_dict, cli_provided)
    
    # Create config
    config = WorldModelAgentConfig(
        model_path=get_val('model_path'),
        action_selection=get_val('action_selection', 'best_guess'),
        num_action_samples=get_val('num_action_samples', 100),
        num_action_seeds=get_val('num_action_seeds', 10),
        action_opt_steps=get_val('action_opt_steps', 50),
        action_learning_rate=get_val('action_learning_rate', 0.01),
        action_init_zero=get_val('action_init_zero', False),
        warmup_steps=get_val('warmup_steps', 0),
        device=get_val('device', 'auto'),
        show_model_internals=get_val('show_model_internals', False),
        env_id=get_val('env_id', 'optomech-v1'),
        object_type=get_val('object_type', 'single'),
        ao_interval_ms=get_val('ao_interval_ms', 5.0),
        control_interval_ms=get_val('control_interval_ms', 5.0),
        frame_interval_ms=get_val('frame_interval_ms', 5.0),
        decision_interval_ms=get_val('decision_interval_ms', 5.0),
        num_atmosphere_layers=get_val('num_atmosphere_layers', 0),
        aperture_type=get_val('aperture_type', 'elf'),
        focal_plane_image_size_pixels=get_val('focal_plane_image_size_pixels', 512),
        observation_mode=get_val('observation_mode', 'image_only'),
        command_secondaries=get_val('command_secondaries', True),
        incremental_control=get_val('incremental_control', False),
        init_differential_motion=get_val('init_differential_motion', True),
        model_wind_diff_motion=get_val('model_wind_diff_motion', True),
        num_envs=get_val('num_envs', 1),
        reward_function=get_val('reward_function', 'align'),
        dataset=get_val('dataset', True),
        max_episode_steps=get_val('max_episode_steps', 100),
        num_episodes=get_val('num_episodes', 1),
        render=get_val('render', False),
        save_rollout=get_val('save_rollout', False),
        rollout_save_path=get_val('rollout_save_path', 'rollouts'),
        viz_interval=get_val('viz_interval', 1),
        seed=get_val('seed', 42)
    )
    
    print("🔧 WorldModelAgent Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Register custom environment
    print(f"\n📝 Registering environment: {config.env_id}")
    gym.envs.registration.register(
        id=config.env_id,
        entry_point='optomech.optomech.optomech:OptomechEnv',
        max_episode_steps=config.max_episode_steps,
    )
    
    # Create environment
    print(f"🌍 Creating environment: {config.env_id}")
    
    # Use the environment config helper to get all required parameters
    from optomech.optomech.env_config import OptomechEnvConfig, create_env_args_from_config
    
    # Create base config with defaults
    env_config = OptomechEnvConfig(
        env_id=config.env_id,
        object_type=config.object_type,
        ao_interval_ms=config.ao_interval_ms,
        control_interval_ms=config.control_interval_ms,
        frame_interval_ms=config.frame_interval_ms,
        decision_interval_ms=config.decision_interval_ms,
        num_atmosphere_layers=config.num_atmosphere_layers,
        aperture_type=config.aperture_type,
        focal_plane_image_size_pixels=config.focal_plane_image_size_pixels,
        observation_mode=config.observation_mode,
        command_secondaries=config.command_secondaries,
        incremental_control=config.incremental_control,
        init_differential_motion=config.init_differential_motion,
        model_wind_diff_motion=config.model_wind_diff_motion,
        reward_function=config.reward_function,
        max_episode_steps=config.max_episode_steps
    )
    
    # Convert to namespace (includes all defaults)
    env_args = create_env_args_from_config(env_config)
    
    # Add dataset parameter (not in OptomechEnvConfig but needed by env)
    env_args.dataset = config.dataset
    
    # Create environment with all parameters
    env = gym.make(config.env_id, **vars(env_args))
    
    # Get target observation from environment
    print("\n🎯 Getting target observation from environment")
    target_observation = env.unwrapped.optical_system.target_image
    print(f"   Target shape: {target_observation.shape}")
    
    # Load checkpoint to extract input_crop_size and log_scale from training config
    print(f"\n🔧 Loading checkpoint to extract training config")
    
    # Import WorldModelConfig for unpickling
    try:
        from optomech.world_models.train_world_model import WorldModelConfig as TrainingConfig
        sys.modules['__main__'].WorldModelConfig = TrainingConfig
    except ImportError:
        pass
    
    checkpoint = torch.load(config.model_path, map_location='cpu', weights_only=False)
    training_config = checkpoint.get('config', None)
    
    # Extract preprocessing parameters from training config
    input_crop_size = None
    log_scale = False
    if training_config is not None:
        if hasattr(training_config, 'input_crop_size'):
            input_crop_size = training_config.input_crop_size
        if hasattr(training_config, 'log_scale'):
            log_scale = training_config.log_scale
    
    print(f"   Input crop size: {input_crop_size}")
    print(f"   Log scale: {log_scale}")
    
    # Create agent
    print(f"\n🤖 Creating WorldModelAgent")
    agent = WorldModelAgent(
        model_path=config.model_path,
        target_observation=target_observation,
        action_selection=config.action_selection,
        num_action_samples=config.num_action_samples,
        num_action_seeds=config.num_action_seeds,
        action_opt_steps=config.action_opt_steps,
        action_learning_rate=config.action_learning_rate,
        action_init_zero=config.action_init_zero,
        warmup_steps=config.warmup_steps,
        device=config.device,
        input_crop_size=input_crop_size,
        log_scale=log_scale,
        show_model_internals=config.show_model_internals,
        debug_predictions=config.debug_predictions
    )
    
    # Run episodes
    print(f"\n🚀 Starting rollout for {config.num_episodes} episode(s)")
    print("=" * 60)
    
    # Storage for all episodes
    all_episode_rewards = []
    all_episode_lengths = []
    all_step_rewards = []  # For plotting reward per step across all episodes
    
    for episode in range(config.num_episodes):
        print(f"\n📊 Episode {episode + 1}/{config.num_episodes}")
        
        # Reset environment and agent
        obs, info = env.reset(seed=config.seed + episode)
        agent.reset()
        
        # Debug: print action space info
        print(f"  🎯 Environment action space: {env.action_space}")
        print(f"  🎯 Agent action dim: {agent.action_dim}")
        
        episode_reward = 0
        episode_length = 0
        episode_step_rewards = []  # Rewards for this episode
        done = False
        truncated = False
        
        # Window for visualization (store last 4 steps)
        window_size = 4
        obs_window = []
        actions_window = []
        predictions_window = []
        lstm_outs_window = []
        
        while not (done or truncated):
            # Select action
            print(f"\n  Step {episode_length + 1}")
            print(f"  📸 Observation shape: {obs.shape}")
            
            # Compute state estimate with LSTM output for visualization
            state_estimate, lstm_out = agent.compute_state_estimate(obs, return_lstm_out=True)
            
            # Store current observation (preprocessed) for window
            current_obs = agent.model_input_observation.clone()
            
            # Store current observation for visualization in action selection methods
            agent.current_observation = obs
            
            # Increment step counter
            agent.step_counter += 1
            
            # During warmup: return random action without running action selection
            if agent.step_counter <= agent.warmup_steps:
                action = np.random.uniform(-1, 1, size=agent.action_dim).astype(np.float32)
                print(f"  🔥 Warmup step {agent.step_counter}/{agent.warmup_steps}: using random action")
                # For warmup, just predict with zero action for visualization
                action_tensor = torch.from_numpy(action).unsqueeze(0).to(agent.device)
                pred_next_obs = agent.predict_next_observation(state_estimate, action_tensor)
            else:
                # After warmup: use configured action selection strategy
                if agent.action_selection == "best_guess":
                    action = agent.select_action_best_guess(state_estimate)
                elif agent.action_selection == "differentiation":
                    action = agent.select_action_differentiation(state_estimate)
                else:
                    raise NotImplementedError(
                        f"Action selection strategy '{agent.action_selection}' not implemented. "
                        f"Available strategies: 'best_guess', 'differentiation'"
                    )
                
                # Get prediction for visualization
                action_tensor = torch.from_numpy(action).unsqueeze(0).to(agent.device)
                pred_next_obs = agent.predict_next_observation(state_estimate, action_tensor)
            
            print(f"  🎮 Selected action: {action}")
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Add to window (keep last window_size steps)
            obs_window.append(current_obs.squeeze(0))  # Remove batch dim
            actions_window.append(action)
            predictions_window.append(pred_next_obs)
            lstm_outs_window.append(lstm_out)
            
            if len(obs_window) > window_size:
                obs_window.pop(0)
                actions_window.pop(0)
                predictions_window.pop(0)
                lstm_outs_window.pop(0)
            
            episode_reward += reward
            episode_length += 1
            episode_step_rewards.append(reward)
            
            print(f"  💰 Reward: {reward:.4f}")
            print(f"  📈 Episode reward: {episode_reward:.4f}")
            
            # Generate visualization at specified interval (when window is full)
            if len(obs_window) == window_size and episode_length % config.viz_interval == 0:
                print(f"  🎨 Generating rollout visualization...")
                # Create target window (same as current obs for all steps since we're tracking to target)
                targets_window = [agent.target_observation] * window_size
                
                viz_path = f"rollout_viz_ep{episode+1}_step{episode_length}.png"
                try:
                    create_rollout_visualization(
                        obs_window=obs_window,
                        actions_window=actions_window,
                        predictions_window=predictions_window,
                        targets_window=targets_window,
                        lstm_outs_window=lstm_outs_window,
                        episode_num=episode + 1,
                        step_num=episode_length,
                        save_path=viz_path
                    )
                except Exception as e:
                    print(f"  ⚠️  Visualization failed: {e}")
            
            if done or truncated:
                print(f"\n  🏁 Episode finished!")
                print(f"     Total reward: {episode_reward:.4f}")
                print(f"     Episode length: {episode_length}")
                print(f"     Done: {done}, Truncated: {truncated}")
        
        # Store episode statistics
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        all_step_rewards.extend(episode_step_rewards)
    
    env.close()
    print("\n✅ Rollout complete!")
    
    # Plot rewards
    print("\n📊 Generating reward plots...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Rollout Statistics - {config.action_selection} mode', fontsize=16)
    
    # Plot 1: Cumulative reward per episode
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(all_episode_rewards) + 1), all_episode_rewards, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Total Reward per Episode', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Episode lengths
    ax2 = axes[0, 1]
    ax2.plot(range(1, len(all_episode_lengths) + 1), all_episode_lengths, marker='s', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward per step (all episodes concatenated)
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(all_step_rewards) + 1), all_step_rewards, linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Step (across all episodes)', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Reward per Step', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Summary Statistics
    {'='*40}
    
    Episodes: {len(all_episode_rewards)}
    Total Steps: {len(all_step_rewards)}
    
    Mean Episode Reward: {np.mean(all_episode_rewards):.4f}
    Std Episode Reward: {np.std(all_episode_rewards):.4f}
    Min Episode Reward: {np.min(all_episode_rewards):.4f}
    Max Episode Reward: {np.max(all_episode_rewards):.4f}
    
    Mean Episode Length: {np.mean(all_episode_lengths):.2f}
    
    Mean Step Reward: {np.mean(all_step_rewards):.4f}
    Std Step Reward: {np.std(all_step_rewards):.4f}
    
    Action Selection: {config.action_selection}
    """
    if config.action_selection == "best_guess":
        stats_text += f"    Num Action Samples: {config.num_action_samples}\n"
    elif config.action_selection == "differentiation":
        stats_text += f"    Optimization Steps: {config.action_opt_steps}\n"
        stats_text += f"    Learning Rate: {config.action_learning_rate}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    save_path = f"rollout_stats_{config.action_selection}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📁 Saved plot to: {save_path}")
    
    plt.show(block=False)
    input("\nPress Enter to close plot and exit...")
    plt.close()


if __name__ == "__main__":
    main()
