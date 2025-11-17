"""
TD3 Evaluation Module.

Provides evaluation utilities for TD3 agents including:
- Episode rollouts with LSTM hidden state management
- Performance metrics tracking
- Baseline policy comparisons
- Tensorboard logging
"""

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Callable
from torch.utils.tensorboard import SummaryWriter
import time


class TD3Evaluator:
    """
    Evaluator for TD3 agents with LSTM support.
    
    Handles episode rollouts, metrics computation, and baseline comparisons.
    """
    
    def __init__(self,
                 env: gym.Env,
                 actor: torch.nn.Module,
                 device: torch.device = torch.device('cpu'),
                 render_mode: Optional[str] = None):
        """
        Args:
            env: Gymnasium environment
            actor: Actor network to evaluate
            device: Device for computations
            render_mode: Rendering mode for environment
        """
        self.env = env
        self.actor = actor
        self.device = device
        self.render_mode = render_mode
        
    def rollout_episode(self,
                       max_steps: Optional[int] = None,
                       deterministic: bool = True,
                       render: bool = False) -> Dict[str, any]:
        """
        Roll out a single episode using the actor policy.
        
        Args:
            max_steps: Maximum steps per episode (None = no limit)
            deterministic: Use deterministic policy (no exploration noise)
            render: Whether to render the episode
            
        Returns:
            Dictionary with episode metrics and trajectory data
        """
        observation, info = self.env.reset()
        
        # Initialize LSTM hidden state
        actor_hidden = self.actor.get_zero_hidden()
        if self.device:
            actor_hidden = (actor_hidden[0].to(self.device),
                          actor_hidden[1].to(self.device))
        
        # Initialize prior action and reward (zeros for first step)
        prior_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        prior_reward = np.array(0.0, dtype=np.float32)
        
        # Tracking
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            prior_action_tensor = torch.tensor(prior_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            prior_reward_tensor = torch.tensor(prior_reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get action from actor
            with torch.no_grad():
                action_tensor, actor_hidden = self.actor(
                    obs_tensor,
                    prior_action_tensor,
                    prior_reward_tensor,
                    actor_hidden
                )
                action = action_tensor.cpu().numpy()[0]
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Track trajectory
            trajectory['observations'].append(observation)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['infos'].append(info)
            
            total_reward += reward
            steps += 1
            
            # Update state
            observation = next_observation
            prior_action = action
            prior_reward = reward
            
            # Render if requested
            if render and self.render_mode:
                self.env.render()
            
            # Check max steps
            if max_steps and steps >= max_steps:
                break
        
        # Compute metrics
        metrics = {
            'episode_return': total_reward,
            'episode_length': steps,
            'trajectory': trajectory,
            'final_info': info
        }
        
        # Extract cost if available
        if 'cost' in info:
            metrics['final_cost'] = info['cost']
        
        return metrics
    
    def evaluate(self,
                num_episodes: int = 10,
                max_steps: Optional[int] = None,
                deterministic: bool = True,
                render: bool = False,
                verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate actor over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            deterministic: Use deterministic policy
            render: Whether to render episodes
            verbose: Print progress
            
        Returns:
            Dictionary of aggregated metrics
        """
        self.actor.eval()
        
        episode_returns = []
        episode_lengths = []
        final_costs = []
        
        for ep in range(num_episodes):
            if verbose:
                print(f"Evaluating episode {ep + 1}/{num_episodes}...")
            
            metrics = self.rollout_episode(
                max_steps=max_steps,
                deterministic=deterministic,
                render=render
            )
            
            episode_returns.append(metrics['episode_return'])
            episode_lengths.append(metrics['episode_length'])
            
            if 'final_cost' in metrics:
                final_costs.append(metrics['final_cost'])
        
        # Aggregate metrics
        results = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'min_return': np.min(episode_returns),
            'max_return': np.max(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }
        
        if final_costs:
            results.update({
                'mean_final_cost': np.mean(final_costs),
                'std_final_cost': np.std(final_costs),
                'min_final_cost': np.min(final_costs),
                'max_final_cost': np.max(final_costs),
            })
        
        if verbose:
            print(f"\nEvaluation Results ({num_episodes} episodes):")
            print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
            print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            if final_costs:
                print(f"  Mean Final Cost: {results['mean_final_cost']:.4f} ± {results['std_final_cost']:.4f}")
        
        return results
    
    def evaluate_with_baselines(self,
                                num_episodes: int = 10,
                                max_steps: Optional[int] = None,
                                baselines: Optional[Dict[str, Callable]] = None,
                                verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate actor and compare with baseline policies.
        
        Args:
            num_episodes: Number of episodes per policy
            max_steps: Maximum steps per episode
            baselines: Dictionary mapping baseline names to policy functions
            verbose: Print progress
            
        Returns:
            Dictionary mapping policy names to their metrics
        """
        results = {}
        
        # Evaluate main actor
        if verbose:
            print("Evaluating TD3 Actor...")
        results['td3_actor'] = self.evaluate(
            num_episodes=num_episodes,
            max_steps=max_steps,
            deterministic=True,
            render=False,
            verbose=verbose
        )
        
        # Evaluate baselines
        if baselines:
            for name, policy_fn in baselines.items():
                if verbose:
                    print(f"\nEvaluating baseline: {name}...")
                
                baseline_returns = []
                baseline_lengths = []
                
                for ep in range(num_episodes):
                    observation, info = self.env.reset()
                    total_reward = 0.0
                    steps = 0
                    done = False
                    
                    while not done:
                        action = policy_fn(observation)
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        total_reward += reward
                        steps += 1
                        
                        if max_steps and steps >= max_steps:
                            break
                    
                    baseline_returns.append(total_reward)
                    baseline_lengths.append(steps)
                
                results[name] = {
                    'mean_return': np.mean(baseline_returns),
                    'std_return': np.std(baseline_returns),
                    'mean_length': np.mean(baseline_lengths),
                    'std_length': np.std(baseline_lengths),
                }
                
                if verbose:
                    print(f"  Mean Return: {results[name]['mean_return']:.2f} ± {results[name]['std_return']:.2f}")
        
        return results
    
    def log_to_tensorboard(self,
                           writer: SummaryWriter,
                           global_step: int,
                           num_episodes: int = 5,
                           max_steps: Optional[int] = None,
                           prefix: str = "eval"):
        """
        Evaluate and log metrics to tensorboard.
        
        Args:
            writer: Tensorboard writer
            global_step: Current training step
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            prefix: Prefix for tensorboard tags
        """
        metrics = self.evaluate(
            num_episodes=num_episodes,
            max_steps=max_steps,
            deterministic=True,
            render=False,
            verbose=False
        )
        
        # Log metrics
        writer.add_scalar(f"{prefix}/mean_return", metrics['mean_return'], global_step)
        writer.add_scalar(f"{prefix}/std_return", metrics['std_return'], global_step)
        writer.add_scalar(f"{prefix}/mean_length", metrics['mean_length'], global_step)
        
        if 'mean_final_cost' in metrics:
            writer.add_scalar(f"{prefix}/mean_final_cost", metrics['mean_final_cost'], global_step)
            writer.add_scalar(f"{prefix}/std_final_cost", metrics['std_final_cost'], global_step)


def create_baseline_policies(env: gym.Env) -> Dict[str, Callable]:
    """
    Create standard baseline policies for comparison.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Dictionary mapping baseline names to policy functions
    """
    baselines = {}
    
    # Zero action baseline
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    baselines['zero_action'] = lambda obs: zero_action
    
    # Random action baseline
    baselines['random_action'] = lambda obs: env.action_space.sample()
    
    # Prior action baseline (always repeat previous action)
    prior_action = [np.zeros(env.action_space.shape, dtype=np.float32)]
    def prior_action_policy(obs):
        action = prior_action[0]
        return action
    baselines['prior_action'] = prior_action_policy
    
    return baselines
