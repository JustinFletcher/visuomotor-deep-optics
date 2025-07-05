import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


# Define the Visual Pendulum Environment (using the wrapper you already provided)
class VisualPendulumEnvWrapper(gym.Wrapper):
    def __init__(self, resolution=64, crop_size=(32, 32), num_frames=1):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.resolution = resolution
        self.crop_size = crop_size
        self.num_frames = num_frames

    def reset(self):
        return super().reset()

    def step(self, action):
        return super().step(action)

    def render(self, mode="rgb_array"):
        return super().render(mode=mode)


# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv(x / 255.0)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, action):
        x = self.conv(x / 255.0)
        x = x.view(x.size(0), -1)
        return self.fc(torch.cat([x, action], dim=1))


# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.stack(actions), np.array(rewards),
                np.stack(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, env, device, gamma=0.99, tau=0.005, lr=1e-3, buffer_size=100000):
        self.env = env
        obs_shape = (3, 32, 32)  # Example shape after cropping and preprocessing
        action_dim = env.action_space.shape[0]
        
        # Initialize networks
        self.actor = Actor(obs_shape, action_dim).to(device)
        self.actor_target = Actor(obs_shape, action_dim).to(device)
        self.critic = Critic(obs_shape, action_dim).to(device)
        self.critic_target = Critic(obs_shape, action_dim).to(device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(max_size=buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def act(self, state, noise_scale=0.1):
        # Ensure state is in the correct shape
        if isinstance(state, np.ndarray) and len(state.shape) == 3:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}, expected (channels, height, width)")

        action = self.actor(state).cpu().data.numpy().flatten()
        action += noise_scale * np.random.randn(self.env.action_space.shape[0])
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return  # Skip update if there isn't enough data
        
        # Sample from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, episodes, batch_size, render=False):
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                if render:
                    self.env.render()
                
                # Select action and take a step
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Add to replay buffer
                self.buffer.add(state, action, reward, next_state, done)
                
                # Update agent
                self.update(batch_size)
                
                # Move to the next state
                state = next_state
                episode_reward += reward
            
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
            
    def act(self, state, noise_scale=0.1):
        # Ensure state is in the correct shape
        if isinstance(state, np.ndarray) and len(state.shape) == 3:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}, expected (channels, height, width)")

        action = self.actor(state).cpu().data.numpy().flatten()
        action += noise_scale * np.random.randn(self.env.action_space.shape[0])
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    
    def train(self, episodes, batch_size, render=False):
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                if render:
                    self.env.render()
                
                # Select action and take a step
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Add to replay buffer
                self.buffer.add(state, action, reward, next_state, done)
                
                # Update agent
                self.update(batch_size)
                
                # Move to the next state
                state = next_state
                episode_reward += reward
            
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


# Instantiate and train the agent
if __name__ == "__main__":
    env = VisualPendulumEnvWrapper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the agent
    agent = DDPGAgent(env, device, lr=1e-3)
    
    # Train the agent
    agent.train(episodes=100, batch_size=64, render=False)
    
    env.close()