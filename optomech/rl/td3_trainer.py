"""
TD3 Training Module.

Provides training logic for TD3 (Twin Delayed DDPG) algorithm including:
- Actor and critic loss computation
- Target network soft updates
- Delayed policy updates
- Target policy smoothing
- Gradient clipping
- OU noise exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter


class OUNoise:
    """Ornstein-Uhlenbeck noise process for exploration."""
    
    def __init__(self,
                 action_dim: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma_initial: float = 0.2,
                 min_sigma: float = 0.05,
                 decay_rate: float = 0.995,
                 auto_decay: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            action_dim: Dimension of action space
            mu: Mean of noise
            theta: Rate of mean reversion
            sigma_initial: Initial standard deviation
            min_sigma: Minimum standard deviation
            decay_rate: Exponential decay rate for sigma
            auto_decay: Whether to automatically decay sigma
            device: Device for tensor operations
        """
        self.mu = mu
        self.theta = theta
        self.sigma_initial = sigma_initial
        self.sigma = sigma_initial
        self.min_sigma = min_sigma
        self.decay_rate = decay_rate
        self.auto_decay = auto_decay
        self.device = device
        self.action_dim = action_dim
        self.mu_tensor = torch.full((action_dim,), mu, dtype=torch.float32, device=device)
        self.state = self.mu_tensor.clone()
    
    def reset(self):
        """Reset internal state to mean."""
        self.state = self.mu_tensor.clone()
    
    def sample(self) -> torch.Tensor:
        """Generate noise sample and decay sigma if enabled."""
        noise = self.theta * (self.mu_tensor - self.state) + \
                self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state += noise
        if self.auto_decay:
            self.decay()
        return self.state
    
    def decay(self):
        """Apply exponential decay to sigma."""
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)


def get_grad_norm(model: nn.Module, norm_type: int = 2) -> float:
    """Compute gradient norm for model parameters."""
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class TD3Trainer:
    """
    TD3 (Twin Delayed DDPG) trainer.
    
    Implements the TD3 algorithm with LSTM support for recurrent processing.
    """
    
    def __init__(self,
                 actor: nn.Module,
                 qf1: nn.Module,
                 qf2: nn.Module,
                 target_actor: nn.Module,
                 target_qf1: nn.Module,
                 target_qf2: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 qf1_optimizer: torch.optim.Optimizer,
                 qf2_optimizer: torch.optim.Optimizer,
                 gamma: float = 0.99,
                 tau: float = 0.004,
                 policy_frequency: int = 2,
                 target_smoothing: bool = True,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 max_grad_norm: float = 1.0,
                 reward_scale: float = 1.0,
                 device: torch.device = torch.device('cpu'),
                 writer: Optional[SummaryWriter] = None,
                 writer_interval: int = 1000):
        """
        Args:
            actor: Actor network
            qf1: Critic 1 network
            qf2: Critic 2 network
            target_actor: Target actor network
            target_qf1: Target critic 1 network
            target_qf2: Target critic 2 network
            actor_optimizer: Optimizer for actor
            qf1_optimizer: Optimizer for critic 1
            qf2_optimizer: Optimizer for critic 2
            gamma: Discount factor
            tau: Soft update coefficient
            policy_frequency: Frequency of policy updates (delayed)
            target_smoothing: Whether to use target policy smoothing
            policy_noise: Noise scale for target smoothing
            noise_clip: Clip range for target smoothing noise
            max_grad_norm: Maximum gradient norm for clipping
            reward_scale: Scale factor for rewards
            device: Device for computations
            writer: Tensorboard writer
            writer_interval: Interval for logging to tensorboard
        """
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_actor = target_actor
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        
        self.actor_optimizer = actor_optimizer
        self.qf1_optimizer = qf1_optimizer
        self.qf2_optimizer = qf2_optimizer
        
        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.target_smoothing = target_smoothing
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale
        self.device = device
        self.writer = writer
        self.writer_interval = writer_interval
        
        self.update_count = 0
    
    def compute_critic_loss(self,
                           actor_hidden: Tuple,
                           qf1_hidden: Tuple,
                           qf2_hidden: Tuple,
                           observations: torch.Tensor,
                           actions: torch.Tensor,
                           prior_actions: torch.Tensor,
                           rewards: torch.Tensor,
                           prior_rewards: torch.Tensor,
                           next_observations: torch.Tensor,
                           dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute TD3 critic losses.
        
        Args:
            actor_hidden: Actor hidden states
            qf1_hidden: Critic 1 hidden states
            qf2_hidden: Critic 2 hidden states
            observations: Batch of observations
            actions: Batch of actions
            prior_actions: Batch of prior actions
            rewards: Batch of rewards
            prior_rewards: Batch of prior rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            
        Returns:
            Tuple of (qf1_loss, qf2_loss, metrics_dict)
        """
        with torch.no_grad():
            # Get next actions from target actor
            next_state_actions, _ = self.target_actor(
                next_observations,
                actions,
                rewards,
                actor_hidden
            )
            
            # Target policy smoothing
            if self.target_smoothing:
                noise = (torch.randn_like(next_state_actions) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_state_actions = (next_state_actions + noise).clamp(-1.0, 1.0)
            
            # Compute target Q-values using twin critics (take minimum)
            qf1_next_target, _ = self.target_qf1(
                next_observations,
                next_state_actions,
                actions,
                rewards,
                qf1_hidden
            )
            
            qf2_next_target, _ = self.target_qf2(
                next_observations,
                next_state_actions,
                actions,
                rewards,
                qf2_hidden
            )
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            # Compute target Q-value
            next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target
        
        # Compute current Q-values
        qf1_a_values, _ = self.qf1(observations, actions, prior_actions, prior_rewards, qf1_hidden)
        qf2_a_values, _ = self.qf2(observations, actions, prior_actions, prior_rewards, qf2_hidden)
        
        # Compute losses
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        
        metrics = {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'qf1_values_mean': qf1_a_values.mean().item(),
            'qf2_values_mean': qf2_a_values.mean().item(),
            'target_q_mean': next_q_value.mean().item(),
        }
        
        return qf1_loss, qf2_loss, metrics
    
    def compute_actor_loss(self,
                          actor_hidden: Tuple,
                          qf1_hidden: Tuple,
                          observations: torch.Tensor,
                          prior_actions: torch.Tensor,
                          prior_rewards: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute actor loss (policy gradient).
        
        Args:
            actor_hidden: Actor hidden states
            qf1_hidden: Critic 1 hidden states
            observations: Batch of observations
            prior_actions: Batch of prior actions
            prior_rewards: Batch of prior rewards
            
        Returns:
            Tuple of (actor_loss, metrics_dict)
        """
        # Get actions from current policy
        actions, _ = self.actor(observations, prior_actions, prior_rewards, actor_hidden)
        
        # Compute Q-values for these actions
        q_values, _ = self.qf1(observations, actions, prior_actions, prior_rewards, qf1_hidden)
        
        # Actor loss is negative Q-value (we want to maximize Q)
        actor_loss = -q_values.mean()
        
        metrics = {
            'actor_loss': actor_loss.item(),
            'actor_q_mean': q_values.mean().item(),
        }
        
        return actor_loss, metrics
    
    def soft_update_targets(self):
        """Soft update target networks."""
        # Update target critics
        for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update target actor
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(self,
                   batch: Tuple,
                   global_step: int) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of data from replay buffer
            global_step: Current global training step
            
        Returns:
            Dictionary of training metrics
        """
        # Unpack batch
        (actor_hidden, qf1_hidden, qf2_hidden,
         observations, actions, prior_actions,
         rewards, prior_rewards, next_observations, dones) = batch
        
        # Convert to tensors
        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        prior_actions = torch.tensor(np.array(prior_actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        prior_rewards = torch.tensor(np.array(prior_rewards), dtype=torch.float32, device=self.device)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        
        # ===== Train Critics =====
        qf1_loss, qf2_loss, critic_metrics = self.compute_critic_loss(
            actor_hidden, qf1_hidden, qf2_hidden,
            observations, actions, prior_actions,
            rewards, prior_rewards, next_observations, dones
        )
        
        # Update QF1
        if global_step % self.writer_interval == 0:
            qf1_grad = get_grad_norm(self.qf1)
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=self.max_grad_norm)
        if global_step % self.writer_interval == 0:
            qf1_grad_clipped = get_grad_norm(self.qf1)
            critic_metrics['qf1_grad'] = qf1_grad
            critic_metrics['qf1_grad_clipped'] = qf1_grad_clipped
        self.qf1_optimizer.step()
        
        # Update QF2
        if global_step % self.writer_interval == 0:
            qf2_grad = get_grad_norm(self.qf2)
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=self.max_grad_norm)
        if global_step % self.writer_interval == 0:
            qf2_grad_clipped = get_grad_norm(self.qf2)
            critic_metrics['qf2_grad'] = qf2_grad
            critic_metrics['qf2_grad_clipped'] = qf2_grad_clipped
        self.qf2_optimizer.step()
        
        # ===== Train Actor (Delayed) =====
        actor_metrics = {}
        if self.update_count % self.policy_frequency == 0:
            actor_loss, actor_metrics = self.compute_actor_loss(
                actor_hidden, qf1_hidden,
                observations, prior_actions, prior_rewards
            )
            
            # Freeze critic parameters
            for p in self.qf1.parameters():
                p.requires_grad = False
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            
            # Unfreeze critic parameters
            for p in self.qf1.parameters():
                p.requires_grad = True
            
            if global_step % self.writer_interval == 0:
                actor_grad = get_grad_norm(self.actor)
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
            
            if global_step % self.writer_interval == 0:
                actor_grad_clipped = get_grad_norm(self.actor)
                actor_metrics['actor_grad'] = actor_grad
                actor_metrics['actor_grad_clipped'] = actor_grad_clipped
            
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update_targets()
        
        # Update critic targets every step (even when actor is not updated)
        else:
            # Soft update only critic targets
            for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_count += 1
        
        # Log to tensorboard
        if self.writer and global_step % self.writer_interval == 0:
            for k, v in critic_metrics.items():
                self.writer.add_scalar(f"losses/{k}", v, global_step)
            for k, v in actor_metrics.items():
                self.writer.add_scalar(f"losses/{k}", v, global_step)
        
        # Combine metrics
        metrics = {**critic_metrics, **actor_metrics}
        
        return metrics
