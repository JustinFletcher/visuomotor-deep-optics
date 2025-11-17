#!/usr/bin/env python3
"""
World Model Architecture

A recurrent world model that predicts next observations given current observations and actions.

Architecture:
1. Encoder: Maps observations to latent representation (from pretrained autoencoder)
2. LSTM: Processes latent representation to produce state estimate
3. Action MLP: Encodes actions to same dimensionality as state estimate
4. Fusion: Adds encoded action to state estimate
5. Decoder: Reconstructs next observation from fused representation (from pretrained autoencoder)

Supports BPTT (Backpropagation Through Time) for sequence-based training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WorldModel(nn.Module):
    """
    Recurrent World Model for next observation prediction.
    
    Architecture:
        obs_t -> Encoder -> z_t -> LSTM -> s_t
        action_t -> ActionMLP -> a_encoded
        s_t + a_encoded -> Decoder -> obs_t+1 (predicted)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        action_dim: int,
        state_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        action_hidden_dim: int = 128,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False
    ):
        """
        Args:
            encoder: Pretrained encoder module (e.g., from autoencoder)
            decoder: Pretrained decoder module (e.g., from autoencoder)
            latent_dim: Dimension of encoder output
            action_dim: Dimension of action space
            state_dim: Dimension of LSTM state representation
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            action_hidden_dim: Hidden dimension for action MLP
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
        """
        super(WorldModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: maps observations to latent representation
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # LSTM: processes latent representations to produce state estimates
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Project LSTM hidden state to state representation
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.ReLU()
        )
        
        # Action encoder: maps actions to same dimensionality as state
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, state_dim),
            nn.ReLU()
        )
        
        # Decoder: maps fused representation to next observation
        self.decoder = decoder
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def get_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get zero-initialized LSTM hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h0, c0) with shape [num_layers, batch, hidden_dim]
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0
    
    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through world model.
        
        Args:
            obs: Observations [batch, seq_len, channels, height, width]
            actions: Actions [batch, seq_len, action_dim]
            hidden: Optional initial LSTM hidden state (h0, c0)
                   Each with shape [num_layers, batch, hidden_dim]
        
        Returns:
            Tuple of:
                - next_obs_pred: Predicted next observations [batch, seq_len, channels, height, width]
                - latent: Encoded latent representations [batch, seq_len, latent_dim]
                - hidden: Final LSTM hidden state (h0, c0)
        """
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        device = obs.device
        
        # Flatten batch and sequence dimensions for encoder
        obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        
        # Encode observations to latent space
        with torch.set_grad_enabled(self.encoder.training and any(p.requires_grad for p in self.encoder.parameters())):
            latent_flat = self.encoder(obs_flat)
        
        # Reshape back to [batch, seq_len, latent_dim]
        latent = latent_flat.reshape(batch_size, seq_len, -1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_zero_hidden(batch_size, device)
        
        # Process through LSTM to get state representations
        lstm_out, hidden = self.lstm(latent, hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        
        # Project to state representation
        state = self.state_projection(lstm_out)  # [batch, seq_len, state_dim]
        
        # Encode actions
        action_encoded = self.action_encoder(actions)  # [batch, seq_len, state_dim]
        
        # Fuse state and action representations
        fused = state + action_encoded  # [batch, seq_len, state_dim]
        
        # Flatten for decoder
        fused_flat = fused.reshape(batch_size * seq_len, -1)
        
        # Decode to predict next observations
        with torch.set_grad_enabled(self.decoder.training and any(p.requires_grad for p in self.decoder.parameters())):
            next_obs_pred_flat = self.decoder(fused_flat)
        
        # Get the actual output shape from decoder (may differ from input due to cropping)
        decoder_output_shape = next_obs_pred_flat.shape[1:]  # [channels, height, width]
        
        # Reshape back to [batch, seq_len, channels, height, width]
        next_obs_pred = next_obs_pred_flat.reshape(batch_size, seq_len, *decoder_output_shape)
        
        return next_obs_pred, latent, hidden
    
    def predict_next_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next observation for a single timestep (for rollouts/evaluation).
        
        Args:
            obs: Single observation [batch, channels, height, width]
            action: Single action [batch, action_dim]
            hidden: Optional LSTM hidden state (h0, c0)
        
        Returns:
            Tuple of:
                - next_obs_pred: Predicted next observation [batch, channels, height, width]
                - hidden: Updated LSTM hidden state (h0, c0)
        """
        # Add sequence dimension
        obs_seq = obs.unsqueeze(1)  # [batch, 1, channels, height, width]
        action_seq = action.unsqueeze(1)  # [batch, 1, action_dim]
        
        # Forward pass
        next_obs_pred_seq, _, hidden = self.forward(obs_seq, action_seq, hidden)
        
        # Remove sequence dimension
        next_obs_pred = next_obs_pred_seq.squeeze(1)  # [batch, channels, height, width]
        
        return next_obs_pred, hidden


def create_world_model_from_autoencoder(
    autoencoder: nn.Module,
    action_dim: int,
    state_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 1,
    action_hidden_dim: int = 128,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False
) -> WorldModel:
    """
    Create a WorldModel using encoder and decoder from a pretrained autoencoder.
    
    Args:
        autoencoder: Pretrained autoencoder with encoder and decoder attributes
        action_dim: Dimension of action space
        state_dim: Dimension of LSTM state representation
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        action_hidden_dim: Hidden dimension for action MLP
        freeze_encoder: Whether to freeze encoder weights
        freeze_decoder: Whether to freeze decoder weights
    
    Returns:
        WorldModel instance
    """
    # Extract encoder and decoder
    # Check if this is a ResNet-style autoencoder with separate bottleneck layers
    if hasattr(autoencoder, 'encoder') and hasattr(autoencoder, 'bottleneck_encode') and hasattr(autoencoder, 'bottleneck_decode'):
        # Create wrapper modules that include the bottleneck
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder, bottleneck_encode):
                super().__init__()
                self.encoder = encoder
                self.bottleneck_encode = bottleneck_encode
            
            def forward(self, x):
                x = self.encoder(x)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = self.bottleneck_encode(x)
                return x
        
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder, bottleneck_decode):
                super().__init__()
                self.decoder = decoder
                self.bottleneck_decode = bottleneck_decode
            
            def forward(self, z):
                x = self.bottleneck_decode(z)
                # Reshape to [batch, 512, 4, 4] for decoder
                x = x.view(x.size(0), 512, 4, 4)
                x = self.decoder(x)
                return x
        
        encoder = EncoderWrapper(autoencoder.encoder, autoencoder.bottleneck_encode)
        decoder = DecoderWrapper(autoencoder.decoder, autoencoder.bottleneck_decode)
    elif hasattr(autoencoder, 'encoder') and hasattr(autoencoder, 'decoder'):
        # Simple encoder/decoder without bottleneck (e.g., CNN-based)
        encoder = autoencoder.encoder
        decoder = autoencoder.decoder
    elif hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode'):
        # Use the full encode/decode methods
        class EncoderWrapper(nn.Module):
            def __init__(self, encode_fn):
                super().__init__()
                self.encode_fn = encode_fn
            
            def forward(self, x):
                return self.encode_fn(x)
        
        class DecoderWrapper(nn.Module):
            def __init__(self, decode_fn):
                super().__init__()
                self.decode_fn = decode_fn
            
            def forward(self, z):
                return self.decode_fn(z)
        
        encoder = EncoderWrapper(autoencoder.encode)
        decoder = DecoderWrapper(autoencoder.decode)
    else:
        raise ValueError("Autoencoder must have 'encoder'/'decoder' or 'encode'/'decode' methods")
    
    # Get latent dimension
    if hasattr(autoencoder, 'latent_dim'):
        latent_dim = autoencoder.latent_dim
    else:
        # Try to infer from encoder output
        raise ValueError("Cannot determine latent_dim from autoencoder. Please specify explicitly.")
    
    return WorldModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        action_hidden_dim=action_hidden_dim,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder
    )
