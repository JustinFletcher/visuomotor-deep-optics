#!/usr/bin/env python3
"""
Behavior Cloning Model Architecture

A recurrent behavior cloning model that predicts actions given observations.
Architecture matches the world model but outputs actions instead of images.

Architecture:
1. Encoder: Maps observations to latent representation (from pretrained autoencoder)
2. LSTM Branch: Processes latent representation to produce temporal state estimate
3. Fusion MLP: Concatenates [encoder output, LSTM output] and processes with 2-layer MLP
4. Action Head: Maps fusion output to action predictions (2-layer MLP)

This architecture mirrors the world model's design for consistency.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BCModel(nn.Module):
    """
    Recurrent Behavior Cloning Model for action prediction.
    
    Architecture:
        obs_t -> Encoder -> z_t ─┬─> LSTM -> lstm_t
                                  │
                                  └─> [z_t, lstm_t] -> FusionMLP -> f_t -> ActionHead -> action_t
    
    The LSTM branch provides temporal context, while direct encoder output provides current state.
    The fusion MLP combines both sources non-linearly, and the action head predicts actions.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        fusion_hidden_dim: int = 512,
        action_head_hidden_dim: int = 128,
        freeze_encoder: bool = True
    ):
        """
        Args:
            encoder: Pretrained encoder module (e.g., from autoencoder)
            latent_dim: Dimension of encoder output
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            fusion_hidden_dim: Hidden dimension for fusion MLP
            action_head_hidden_dim: Hidden dimension for action head MLP
            freeze_encoder: Whether to freeze encoder weights
        """
        super(BCModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: maps observations to latent representation
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # LSTM Branch: processes latent representations to produce temporal state estimates
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fusion MLP: 2-layer MLP that combines [encoder_out, lstm_out]
        # Input dimension: latent_dim + hidden_dim
        fusion_input_dim = latent_dim + hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, latent_dim)  # Output dimension matches latent_dim
        )
        
        # Action Head: 2-layer MLP that maps fusion output to actions
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim, action_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_head_hidden_dim, action_dim)
        )
    
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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BC model.
        
        Args:
            obs: Observations [batch, seq_len, channels, height, width]
            hidden: Optional initial LSTM hidden state (h0, c0)
                   Each with shape [num_layers, batch, hidden_dim]
        
        Returns:
            Tuple of:
                - action_pred: Predicted actions [batch, seq_len, action_dim]
                - latent: Encoded latent representations [batch, seq_len, latent_dim]
                - hidden: Final LSTM hidden state (h0, c0)
        """
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        device = obs.device
        
        # Flatten batch and sequence dimensions for encoder
        obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        
        # Encode observations to latent space
        latent_flat = self.encoder(obs_flat)
        
        # Reshape back to [batch, seq_len, latent_dim]
        latent = latent_flat.reshape(batch_size, seq_len, -1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_zero_hidden(batch_size, device)
        
        # LSTM Branch: Process through LSTM to get temporal state representations
        lstm_out, hidden = self.lstm(latent, hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        
        # Concatenate encoder output and LSTM output
        # Shape: [batch, seq_len, latent_dim + hidden_dim]
        concat_features = torch.cat([latent, lstm_out], dim=-1)
        
        # Flatten for fusion MLP
        concat_flat = concat_features.reshape(batch_size * seq_len, -1)
        
        # Fuse with 2-layer MLP
        fused_flat = self.fusion_mlp(concat_flat)  # [batch*seq_len, latent_dim]
        
        # Predict actions with action head
        action_pred_flat = self.action_head(fused_flat)  # [batch*seq_len, action_dim]
        
        # Reshape back to [batch, seq_len, action_dim]
        action_pred = action_pred_flat.reshape(batch_size, seq_len, self.action_dim)
        
        return action_pred, latent, hidden
    
    def predict_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict action for single observation (convenience method for inference).
        
        Args:
            obs: Single observation [batch, channels, height, width]
            hidden: Optional LSTM hidden state from previous step
        
        Returns:
            Tuple of:
                - action: Predicted action [batch, action_dim]
                - hidden: Updated LSTM hidden state
        """
        # Add sequence dimension
        obs = obs.unsqueeze(1)  # [batch, 1, channels, height, width]
        
        # Forward pass
        action_pred, _, hidden = self.forward(obs, hidden)
        
        # Remove sequence dimension
        action = action_pred.squeeze(1)  # [batch, action_dim]
        
        return action, hidden
