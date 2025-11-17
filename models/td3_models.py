"""
TD3 Actor and Critic Models with LSTM Support.

Provides LSTM-based actor and critic models for TD3 training with support for:
- Visual observations (Impala-style CNN encoders)
- Low-dimensional observations
- Pretrained encoder integration
- Hidden state management for recurrent processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# Weight Initialization Utilities
# ============================================================================

def uniform_init(layer: nn.Module, lower_bound: float = -1e-4, upper_bound: float = 1e-4) -> nn.Module:
    """Initialize layer with uniform distribution."""
    nn.init.uniform_(layer.weight, a=lower_bound, b=upper_bound)
    nn.init.uniform_(layer.bias, a=lower_bound, b=upper_bound)
    return layer


def conv_init(layer: nn.Module, bias_const: float = 0.0) -> nn.Module:
    """Initialize convolutional layer with Kaiming normal."""
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def init_lstm_weights(lstm: nn.LSTM) -> nn.LSTM:
    """Initialize LSTM weights with proper initialization."""
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
            # Set forget gate bias to 1.0
            n = param.size(0)
            param.data[n//4:n//2] = 1.0
    return lstm


# ============================================================================
# TD3 Actor Model with LSTM
# ============================================================================

class ImpalaActorLSTM(nn.Module):
    """
    TD3 Actor with Impala-style CNN encoder, LSTM, and action head.
    Supports optional pretrained encoder.
    """
    
    def __init__(self,
                 envs,
                 device,
                 encoder: Optional[nn.Module] = None,
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 1,
                 channel_scale: int = 16,
                 fc_scale: int = 8,
                 action_scale: float = 1.0,
                 use_pretrained_encoder: bool = True):
        """
        Initialize TD3 Actor with LSTM.
        
        Args:
            envs: Gym vectorized environment
            device: Device to use (cuda/mps/cpu)
            encoder: Optional pretrained encoder module
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            channel_scale: CNN channel multiplier
            fc_scale: Fully connected layer size
            action_scale: Scaling factor for actions
            use_pretrained_encoder: Whether to freeze pretrained encoder
        """
        
        super().__init__()
        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        vector_action_size = envs.single_action_space.shape[0]
        obs_shape = envs.single_observation_space.shape
        
        # Check if channels-last
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]
        
        # Use pretrained encoder or create new one
        if encoder is not None:
            self.visual_encoder = encoder
            # Freeze if it's pretrained
            if use_pretrained_encoder:
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
        else:
            # Create Impala-style encoder
            self.visual_encoder = nn.Sequential(
                conv_init(nn.Conv2d(input_channels, channel_scale, kernel_size=8, stride=4)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale, channel_scale * 2, kernel_size=4, stride=2)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale * 2, channel_scale * 4, kernel_size=2, stride=2)),
                nn.ReLU(),
            )
        
        # Compute encoder output shape
        with torch.inference_mode():
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            # Move test tensor to same device as encoder
            if encoder is not None:
                try:
                    encoder_device = next(encoder.parameters()).device
                    x = x.to(encoder_device)
                except StopIteration:
                    pass  # Encoder has no parameters
            visual_output_shape = self.visual_encoder(x).shape
        
        # MLP after encoder
        mlp_output_size = fc_scale
        self.mlp = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(int(np.prod(visual_output_shape[1:])), mlp_output_size), std=np.sqrt(2.0)),
            nn.ReLU(),
        )
        
        # LSTM for temporal processing
        # Input: encoded observation + prior action + prior reward
        self.lstm = nn.LSTM(
            input_size=mlp_output_size + vector_action_size + 1,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        init_lstm_weights(self.lstm)
        
        # Action prediction head
        with torch.inference_mode():
            x = torch.zeros(1, mlp_output_size + vector_action_size + 1)
            pre_head_output_shape = self.lstm(x)[0].shape
        
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(int(np.prod(pre_head_output_shape[1:])), fc_scale), std=1e-3),
            nn.ReLU(),
            uniform_init(nn.Linear(fc_scale, int(np.prod(envs.single_action_space.shape))),
                        lower_bound=-1e-4, upper_bound=1e-4),
            nn.Tanh()
        )
        
        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))
    
    def get_zero_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero hidden state for LSTM."""
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim).detach().to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim).detach().to(self.device)
        return (h, c)
    
    def forward(self, o, a_prior, r_prior, hidden):
        """
        Forward pass through the actor.
        
        Note: Preprocessing (normalization, crop, log-scale) is now done in replay buffer
        to match the approach used in imitation learning
        """
        
        # Handle hidden state format: could be tuple or list
        # When batched from replay buffer, hidden[0] and hidden[1] are lists of tensors
        if isinstance(hidden, (list, tuple)) and len(hidden) == 2:
            h0_raw, c0_raw = hidden[0], hidden[1]
            # Concatenate if lists of tensors (each already [num_layers, 1, hidden_dim])
            if isinstance(h0_raw, list):
                h0 = torch.cat(h0_raw, dim=1)  # Concat along batch dim: [num_layers, batch, hidden_dim]
                c0 = torch.cat(c0_raw, dim=1)
            else:
                h0, c0 = h0_raw, c0_raw
        else:
            raise ValueError(f"Expected hidden to be tuple/list of length 2, got {type(hidden)}")
        
        # Detect if sequence input
        sequence_input = len(o.shape) == 6
        
        if sequence_input:
            batch_size, seq_len, frame, channels, height, width = o.shape
            # Reshape to [batch * seq, channels, height, width]
            o = o.view(-1, o.shape[3], o.shape[4], o.shape[5])
            
            # Handle hidden state: if unbatched, expand to batch
            if h0.dim() == 2:  # [num_layers, hidden_dim] -> need to expand
                h0 = h0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
                c0 = c0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            # else already [num_layers, batch, hidden_dim]
            
            a_prior = a_prior.squeeze(-2) if a_prior.dim() > 3 else a_prior  # Remove extra dim if present
        else:
            batch_size = o.shape[0]
            seq_len = 1
            # Ensure r_prior has the right shape [batch, 1]
            if r_prior.dim() == 1:
                r_prior = r_prior.unsqueeze(-1)
            
            # For non-sequence mode, we will unsqueeze to add time dimension, making input 3D
            # So hidden states must also be 3D [num_layers, batch, hidden_dim]
            if h0.dim() == 2:  # [num_layers, hidden_dim] -> need batch dim
                h0 = h0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
                c0 = c0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
        
        # Process observations through visual encoder
        if self.visual_encoder is not None:
            o = self.visual_encoder(o)
        
        # Flatten spatial dimensions if needed
        if len(o.shape) > 2:
            o = o.reshape(o.size(0), -1)
        
        # Process through MLP
        x = self.mlp(o)
        
        # Reshape for LSTM
        if sequence_input:
            x = x.view(batch_size, seq_len, -1)
        else:
            x = x.unsqueeze(1)  # Add time dimension
        
        # Concatenate with prior actions and rewards
        x = torch.cat([x, a_prior.unsqueeze(1) if a_prior.dim() == 2 else a_prior, 
                      r_prior.unsqueeze(1) if r_prior.dim() == 2 else r_prior], dim=-1)
        
        # Process through LSTM
        x, hidden_out = self.lstm(x, (h0, c0))
        
        # Get last time step if sequence
        if sequence_input:
            x = x[:, -1, :]
        else:
            x = x.squeeze(1)
        
        # Generate action
        action = self.action_head(x)
        action = action * self.action_scale + self.action_bias
        
        return action, hidden_out


# ============================================================================
# TD3 Critic Model with LSTM
# ============================================================================

class ImpalaCriticLSTM(nn.Module):
    """
    Critic model with Impala-style CNN encoder and LSTM for TD3.
    
    Takes observations, actions, and previous actions/rewards, outputs Q-values.
    Maintains LSTM hidden state for temporal processing.
    """
    
    def __init__(self,
                 envs,
                 device: torch.device,
                 encoder: Optional[nn.Module] = None,
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 1,
                 channel_scale: int = 16,
                 fc_scale: int = 64,
                 q_bias: float = 0.0,
                 use_pretrained_encoder: bool = False):
        """
        Args:
            envs: Vectorized environment for shape inference
            device: Device to run model on
            encoder: Optional pretrained encoder module
            lstm_hidden_dim: Hidden dimension of LSTM
            lstm_num_layers: Number of LSTM layers
            channel_scale: Channel multiplier for CNN
            fc_scale: Hidden dimension of MLP layers
            q_bias: Bias initialization for Q-value output
            use_pretrained_encoder: Whether using pretrained encoder
        """
        super().__init__()
        
        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_pretrained_encoder = use_pretrained_encoder
        self.q_bias = q_bias
        
        # Get environment shapes
        obs_shape = envs.single_observation_space.shape
        vector_action_size = envs.single_action_space.shape[0]
        
        # Check if channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]
        
        # Use pretrained encoder or create new one
        if encoder is not None:
            self.visual_encoder = encoder
            if use_pretrained_encoder:
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
        else:
            self.visual_encoder = nn.Sequential(
                conv_init(nn.Conv2d(input_channels, channel_scale, kernel_size=8, stride=4)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale, channel_scale * 2, kernel_size=4, stride=2)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale * 2, channel_scale * 4, kernel_size=2, stride=2)),
                nn.ReLU(),
            )
        
        # Compute encoder output shape
        with torch.inference_mode():
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            # Move test tensor to same device as encoder
            if encoder is not None:
                try:
                    encoder_device = next(encoder.parameters()).device
                    x = x.to(encoder_device)
                except StopIteration:
                    pass  # Encoder has no parameters
            visual_output_shape = self.visual_encoder(x).shape
        
        # MLP after encoder
        mlp_output_size = fc_scale
        self.mlp = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(int(np.prod(visual_output_shape[1:])), mlp_output_size), std=np.sqrt(2)),
            nn.ReLU(),
        )
        
        # LSTM for temporal processing
        # Input: encoded observation + current action + prior action + prior reward
        self.lstm = nn.LSTM(
            input_size=mlp_output_size + vector_action_size + vector_action_size + 1,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        init_lstm_weights(self.lstm)
        
        # Q-value prediction head
        with torch.inference_mode():
            x = torch.zeros(1, mlp_output_size + vector_action_size + vector_action_size + 1)
            pre_head_output_shape = self.lstm(x)[0].shape
        
        self.q_head = nn.Sequential(
            layer_init(nn.Linear(int(np.prod(pre_head_output_shape[1:])), fc_scale), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(fc_scale, 1), std=1.0, bias_const=q_bias),
        )
    
    def get_zero_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero hidden state for LSTM."""
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim).detach().to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim).detach().to(self.device)
        return (h, c)
    
    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                a_prior: torch.Tensor,
                r_prior: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through critic.
        
        Args:
            o: Observations [batch, seq_len, frame, channels, height, width] or [batch, channels, height, width]
            a: Current actions [batch, seq_len, action_dim] or [batch, action_dim]
            a_prior: Prior actions [batch, seq_len, action_dim] or [batch, action_dim]
            r_prior: Prior rewards [batch, seq_len, 1] or [batch, 1]
            hidden: LSTM hidden state tuple (h, c) - shapes [num_layers, hidden_dim] or [num_layers, batch, hidden_dim]
            
        Returns:
            Tuple of (q_values, new_hidden_state)
        """
        # Handle channels-last format
        if self.channels_last:
            if len(o.shape) == 4:
                o = o.permute(0, 3, 1, 2)
            elif len(o.shape) == 6:
                o = o.permute(0, 1, 5, 2, 3, 4)
        
        # Note: Preprocessing (normalization, crop, log-scale) is now done in replay buffer
        # to match the approach used in imitation learning
        
        # Handle hidden state format: could be tuple or list
        # When batched from replay buffer, hidden[0] and hidden[1] are lists of tensors
        if isinstance(hidden, (list, tuple)) and len(hidden) == 2:
            h0_raw, c0_raw = hidden[0], hidden[1]
            # Concatenate if lists of tensors
            if isinstance(h0_raw, list):
                # Check dimensionality of first tensor to determine how to batch
                if h0_raw[0].dim() == 2:
                    # Tensors are 2D [num_layers, hidden_dim], stack to create batch dim
                    h0 = torch.stack(h0_raw, dim=1)  # [num_layers, batch, hidden_dim]
                    c0 = torch.stack(c0_raw, dim=1)
                elif h0_raw[0].dim() == 3:
                    # Tensors are 3D [num_layers, 1, hidden_dim], concatenate along batch dim
                    h0 = torch.cat(h0_raw, dim=1)  # [num_layers, batch, hidden_dim]
                    c0 = torch.cat(c0_raw, dim=1)
                else:
                    raise ValueError(f"Unexpected hidden state dimension: {h0_raw[0].dim()}")
            else:
                h0, c0 = h0_raw, c0_raw
        else:
            raise ValueError(f"Expected hidden to be tuple/list of length 2, got {type(hidden)}")
        
        # Detect if sequence input
        sequence_input = len(o.shape) == 6
        
        if sequence_input:
            batch_size, seq_len, frame, channels, height, width = o.shape
            # Reshape to [batch * seq, channels, height, width]
            o = o.view(-1, o.shape[3], o.shape[4], o.shape[5])
            
            # Handle hidden state: if unbatched, expand to batch (h0, c0 already extracted at top of forward)
            if h0.dim() == 2:  # [num_layers, hidden_dim] -> need to expand
                h0 = h0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
                c0 = c0.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            # else already [num_layers, batch, hidden_dim]
            
            a_prior = a_prior.squeeze(-2) if a_prior.dim() > 3 else a_prior
            a = a.squeeze(-2) if a.dim() > 3 else a
        else:
            batch_size = o.shape[0]
            seq_len = 1
            # Ensure r_prior has the right shape [batch, 1]
            if r_prior.dim() == 1:
                r_prior = r_prior.unsqueeze(-1)
            
            # For Critic non-sequence mode, x remains 2D [batch, features]
            # PyTorch LSTM treats 2D input as unbatched, so hidden states must be 2D [num_layers, hidden_dim]
            # ONLY expand to 3D if already 3D (from batched replay buffer)
            # Do NOT expand if currently 2D (from get_zero_hidden during environment interaction)
            pass  # Keep h0, c0 as-is
        
        # Encode observations
        x_o = self.visual_encoder(o)
        x = self.mlp(x_o)
        
        if sequence_input:
            x = x.view(batch_size, seq_len, -1)
            # Ensure actions and rewards match [batch, seq, features] format
            # They might be 2D [batch, features] and need seq dimension added
            if a.dim() == 2:
                # Infer if this is [batch, features] or [batch*seq, features]
                if a.size(0) == batch_size:
                    # It's [batch, features], replicate across seq dimension
                    a = a.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    # It's [batch*seq, features], reshape to [batch, seq, features]
                    a = a.view(batch_size, seq_len, -1)
            if a_prior.dim() == 2:
                if a_prior.size(0) == batch_size:
                    a_prior = a_prior.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    a_prior = a_prior.view(batch_size, seq_len, -1)
            if r_prior.dim() == 2:
                if r_prior.size(0) == batch_size:
                    r_prior = r_prior.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    r_prior = r_prior.view(batch_size, seq_len, -1)
        
        # Concatenate with current action, prior action, and prior reward
        x = torch.cat([x, a, a_prior, r_prior], dim=-1)
        
        # LSTM processing
        h0 = h0.detach()
        c0 = c0.detach()
        x, new_hidden = self.lstm(x, (h0, c0))
        
        # Detach to prevent gradient backprop through hidden states
        new_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        
        # Predict Q-values
        q = self.q_head(x)
        
        return q, new_hidden
