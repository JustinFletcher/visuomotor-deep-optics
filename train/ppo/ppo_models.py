"""
PPO Actor-Critic Models with LSTM Support for POMDPs.

Provides a shared-encoder recurrent actor-critic for PPO training, plus
a deployment wrapper that matches the interface expected by existing
rollout and evaluation tools (rollout_optomech_policy, TD3Evaluator).

Architecture:
    obs -> CNN encoder -> MLP -> concat(features, prior_action, prior_reward)
        -> LSTM (shared) -> policy_head (Gaussian mean + learnable log_std)
                         -> value_head (scalar V)
"""

import os
import sys

# Ensure repo root is on sys.path for cross-package imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Optional
from torchvision.models import resnet18

from models.td3_models import layer_init, conv_init, init_lstm_weights

# Valid model types: "small" (3-layer CNN), "medium" (ResNet18 encoder)
VALID_MODEL_TYPES = ("small", "medium")


class RecurrentActorCritic(nn.Module):
    """
    PPO Actor-Critic with shared CNN encoder, shared LSTM, and separate
    policy/value heads. Designed for continuous-action POMDPs.

    The policy outputs a Gaussian distribution (learned mean + state-independent
    log_std). Actions are NOT tanh-squashed; they are clamped at environment
    boundaries during interaction, with log_prob computed on the unclamped sample.
    """

    def __init__(
        self,
        envs,
        device: torch.device,
        encoder: Optional[nn.Module] = None,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        channel_scale: int = 16,
        fc_scale: int = 64,
        action_scale: float = 1.0,
        init_log_std: float = -0.5,
        freeze_encoder: bool = False,
        model_type: str = "small",
        target_dim: int = 0,
    ):
        super().__init__()
        assert model_type in VALID_MODEL_TYPES, \
            f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'"
        self.device = device
        self.model_type = model_type
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        # Optional auxiliary target input (e.g. dark-hole geometry
        # [sin(θ), cos(θ), radius, size]). Default 0 preserves the
        # pre-existing LSTM input width so old checkpoints load cleanly.
        self.target_dim = int(target_dim)

        obs_shape = envs.single_observation_space.shape
        self.action_dim = envs.single_action_space.shape[0]

        # Detect channels-last layout
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]

        # --- Visual encoder ---
        if encoder is not None:
            self.visual_encoder = encoder
            if freeze_encoder:
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
        elif model_type == "medium":
            self.visual_encoder = self._build_resnet18_encoder(input_channels)
        else:
            # "small": lightweight 3-layer CNN
            self.visual_encoder = nn.Sequential(
                conv_init(nn.Conv2d(input_channels, channel_scale, kernel_size=8, stride=4)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale, channel_scale * 2, kernel_size=4, stride=2)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale * 2, channel_scale * 4, kernel_size=2, stride=2)),
                nn.ReLU(),
            )

        # Compute encoder output size
        with torch.inference_mode():
            dummy = torch.zeros(1, *obs_shape)
            if self.channels_last:
                dummy = dummy.permute(0, 3, 1, 2)
            if encoder is not None:
                try:
                    enc_device = next(encoder.parameters()).device
                    dummy = dummy.to(enc_device)
                except StopIteration:
                    pass
            encoder_out_shape = self.visual_encoder(dummy).shape
        encoder_out_flat = int(np.prod(encoder_out_shape[1:]))

        # --- Shared MLP after encoder ---
        mlp_out = fc_scale
        self.mlp = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(encoder_out_flat, mlp_out)),
            nn.ReLU(),
        )

        # --- Shared LSTM ---
        # features + prior_action + prior_reward (+ optional target_vec)
        lstm_input = mlp_out + self.action_dim + 1 + self.target_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        init_lstm_weights(self.lstm)

        # --- Policy head ---
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_dim, fc_scale)),
            nn.ReLU(),
            layer_init(nn.Linear(fc_scale, self.action_dim), std=0.01),
        )
        # State-independent log standard deviation
        self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std))

        # --- Value head ---
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_dim, fc_scale)),
            nn.ReLU(),
            layer_init(nn.Linear(fc_scale, 1), std=1.0),
        )

        # Action scaling and clamping
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))
        # Store environment action bounds for clamping
        action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    # ------------------------------------------------------------------
    # Encoder builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_resnet18_encoder(input_channels: int) -> nn.Module:
        """ResNet18 encoder for the 'medium' model variant.

        Takes a ResNet18 pretrained on ImageNet, replaces the first conv
        to accept the observation's channel count (e.g. 1 or 2 instead of 3),
        and strips the final FC + avgpool so it outputs a spatial feature map.
        Output: [B, 512, H', W'] where H', W' depend on input resolution.
        For 128x128 input -> 4x4 spatial -> 512*4*4 = 8192 features after flatten.
        """
        backbone = resnet18(weights=None)

        # Replace first conv to match input channels (no pretrained weights
        # since channel count differs from ImageNet's 3)
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Strip avgpool and fc — we just want the conv feature map
        encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        return encoder

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def get_zero_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero hidden state: each tensor is [num_layers, hidden_dim] (no batch dim)."""
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim, device=self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim, device=self.device)
        return (h, c)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Run observation through CNN + MLP. Handles channels-last."""
        if self.channels_last:
            if obs.dim() == 4:
                obs = obs.permute(0, 3, 1, 2)
            elif obs.dim() == 5:
                b, t = obs.shape[:2]
                obs = obs.permute(0, 1, 4, 2, 3)
        return obs

    def _prepare_hidden(
        self, hidden: Tuple[torch.Tensor, torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure hidden is [num_layers, batch, hidden_dim]."""
        h, c = hidden
        if h.dim() == 2:  # [num_layers, hidden_dim]
            h = h.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            c = c.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
        return h, c

    def _prepare_target(
        self,
        target_vec: Optional[torch.Tensor],
        B: int,
        T: Optional[int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Resolve the per-step target-vec tensor to the right shape.

        Returns None if self.target_dim is 0 (no concat needed)."""
        if self.target_dim == 0:
            return None
        if target_vec is None:
            if T is None:
                return torch.zeros(B, self.target_dim, device=device)
            return torch.zeros(B, T, self.target_dim, device=device)
        # Broadcast shape to match sequence expectation.
        if T is None:
            # [B, target_dim] expected.
            if target_vec.dim() == 3:
                raise ValueError("target_vec has time dim but obs does not")
            return target_vec
        else:
            if target_vec.dim() == 2:
                target_vec = target_vec.unsqueeze(1).expand(-1, T, -1)
            return target_vec

    def _forward_shared(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        target_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Shared forward pass through encoder -> MLP -> LSTM.

        Accepts both single-step [B, ...] and sequence [B, T, ...] inputs.
        When ``self.target_dim > 0`` and ``target_vec`` is None, a zero
        target is substituted so prior-existing callers keep working.
        """
        obs = self._encode_obs(obs)
        is_seq = obs.dim() == 5  # [B, T, C, H, W]

        if is_seq:
            B, T = obs.shape[:2]
            # Flatten batch*time for CNN
            obs_flat = obs.reshape(B * T, *obs.shape[2:])
            features = self.visual_encoder(obs_flat)
            features = self.mlp(features)  # [B*T, mlp_out]
            features = features.view(B, T, -1)

            # Ensure prior_action and prior_reward are [B, T, ...]
            if prior_action.dim() == 2:
                prior_action = prior_action.unsqueeze(1).expand(-1, T, -1)
            if prior_reward.dim() == 1:
                prior_reward = prior_reward.unsqueeze(-1)
            if prior_reward.dim() == 2 and prior_reward.shape[1] != T:
                prior_reward = prior_reward.unsqueeze(1).expand(-1, T, -1)
            elif prior_reward.dim() == 2:
                prior_reward = prior_reward.unsqueeze(-1)
            # prior_reward should be [B, T, 1]

            pieces = [features, prior_action, prior_reward]
            tv = self._prepare_target(target_vec, B, T, features.device)
            if tv is not None:
                pieces.append(tv)
            lstm_in = torch.cat(pieces, dim=-1)
        else:
            B = obs.shape[0]
            features = self.visual_encoder(obs)
            features = self.mlp(features)  # [B, mlp_out]

            if prior_reward.dim() == 1:
                prior_reward = prior_reward.unsqueeze(-1)

            pieces = [features, prior_action, prior_reward]
            tv = self._prepare_target(target_vec, B, None, features.device)
            if tv is not None:
                pieces.append(tv)
            lstm_in = torch.cat(pieces, dim=-1)
            lstm_in = lstm_in.unsqueeze(1)  # [B, 1, input]

        h, c = self._prepare_hidden(hidden, B)
        lstm_out, (h_new, c_new) = self.lstm(lstm_in, (h, c))

        if not is_seq:
            lstm_out = lstm_out.squeeze(1)  # [B, hidden]

        return lstm_out, (h_new, c_new)

    def _forward_shared_sequential(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        target_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shared forward pass that processes sequences step-by-step, resetting
        hidden states at episode boundaries.

        Args:
            obs: [B, T, C, H, W]
            prior_action: [B, T, action_dim]
            prior_reward: [B, T, 1]
            hidden: (h, c) each [num_layers, B, hidden_dim]
            episode_starts: [B, T] binary mask, 1.0 at episode starts

        Returns:
            lstm_out: [B, T, hidden_dim]
        """
        obs = self._encode_obs(obs)
        B, T = obs.shape[:2]

        # Encode all observations at once
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        features = self.visual_encoder(obs_flat)
        features = self.mlp(features).view(B, T, -1)

        if prior_reward.dim() == 2:
            prior_reward = prior_reward.unsqueeze(-1)

        pieces = [features, prior_action, prior_reward]
        tv = self._prepare_target(target_vec, B, T, features.device)
        if tv is not None:
            pieces.append(tv)
        lstm_inputs = torch.cat(pieces, dim=-1)

        h, c = hidden  # Already [num_layers, B, hidden_dim]
        outputs = []

        for t in range(T):
            # Reset hidden state at episode boundaries
            # episode_starts[:, t] is 1.0 where a new episode began
            mask = (1.0 - episode_starts[:, t]).unsqueeze(0).unsqueeze(-1)  # [1, B, 1]
            h = h * mask
            c = c * mask

            # Single LSTM step
            out, (h, c) = self.lstm(lstm_inputs[:, t : t + 1, :], (h, c))
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # [B, T, hidden_dim]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_and_clamp_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Scale a raw policy-space action and clamp to environment bounds."""
        scaled = raw_action * self.action_scale + self.action_bias
        return torch.clamp(scaled, self.action_low, self.action_high)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
        target_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action, log_prob, entropy, value, and new hidden state.

        IMPORTANT: Returns the *raw* (unscaled) action and its log_prob.
        The caller is responsible for scaling/clamping via
        scale_and_clamp_action() before stepping the environment.
        This ensures log_prob and stored actions are consistent for PPO
        ratio computation.

        If action is None, sample a new action from the policy.
        If action is provided, evaluate that action under the current policy
        (used for computing the PPO ratio during training).

        For training with sequences and episode boundary handling, pass
        episode_starts [B, T] to use the sequential forward pass.
        """
        if episode_starts is not None:
            # Sequential processing with hidden state resets
            h, c = self._prepare_hidden(hidden, obs.shape[0])
            lstm_out = self._forward_shared_sequential(
                obs, prior_action, prior_reward, (h, c), episode_starts,
                target_vec=target_vec,
            )
            # Use all timesteps for policy and value
            mean = self.policy_head(lstm_out)  # [B, T, action_dim]
            std = self.log_std.exp().expand_as(mean)
            value = self.value_head(lstm_out)  # [B, T, 1]

            dist = Normal(mean, std)
            if action is None:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # [B, T]
            entropy = dist.entropy().sum(dim=-1)  # [B, T]

            # hidden state not meaningful here (we process the full sequence),
            # return dummy hidden — caller should use stored hidden states
            new_hidden = hidden
            return action, log_prob, entropy, value.squeeze(-1), new_hidden
        else:
            # Single-step or non-sequential batch
            lstm_out, new_hidden = self._forward_shared(
                obs, prior_action, prior_reward, hidden,
                target_vec=target_vec,
            )
            mean = self.policy_head(lstm_out)
            std = self.log_std.exp().expand_as(mean)
            value = self.value_head(lstm_out)

            dist = Normal(mean, std)
            if action is None:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            return action, log_prob, entropy, value.squeeze(-1), new_hidden

    def get_value(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        target_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value estimate only (for GAE bootstrap)."""
        lstm_out, _ = self._forward_shared(
            obs, prior_action, prior_reward, hidden, target_vec=target_vec)
        return self.value_head(lstm_out).squeeze(-1)

    def get_deterministic_action(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        target_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get deterministic (mean) action, scaled and clamped for deployment."""
        lstm_out, new_hidden = self._forward_shared(
            obs, prior_action, prior_reward, hidden, target_vec=target_vec,
        )
        mean = self.policy_head(lstm_out)
        action = self.scale_and_clamp_action(mean)
        return action, new_hidden


class PPOActorWrapper(nn.Module):
    """
    Deployment wrapper that exposes the same interface as ImpalaActorLSTM.

    This allows PPO-trained models to work with:
    - rollout_optomech_policy() in optomech/rollout.py
    - TD3Evaluator in optomech/rl/td3_evaluation.py

    Interface:
        wrapper.get_zero_hidden() -> (h, c)
        wrapper(obs, prior_action, prior_reward, hidden) -> (action, new_hidden)
        wrapper.action_scale  # registered buffer
    """

    def __init__(self, actor_critic: RecurrentActorCritic):
        super().__init__()
        self.actor_critic = actor_critic
        self.register_buffer("action_scale", actor_critic.action_scale.clone())
        # Expose attributes needed by rollout tools
        self.lstm_hidden_dim = actor_critic.lstm_hidden_dim
        self.lstm_num_layers = actor_critic.lstm_num_layers
        self.device = actor_critic.device

    def get_zero_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_critic.get_zero_hidden()

    def forward(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        target_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        action, new_hidden = self.actor_critic.get_deterministic_action(
            obs, prior_action, prior_reward, hidden, target_vec=target_vec,
        )
        return action, new_hidden
