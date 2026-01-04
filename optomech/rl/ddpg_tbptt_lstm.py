# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
import shutil
import copy
from dataclasses import dataclass

from typing import Optional, Tuple

import math

import gymnasium as gym
# from gymnasium.envs import box2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchrl.data import ReplayBuffer
from tensordict import TensorDict

import torch.distributions as dist
from typing import Tuple
from torchrl.envs import GymWrapper, TransformedEnv
# from torchrl.envs.transforms import ToTensor
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchinfo import summary
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from anytree import Node, RenderTree

import matplotlib.pyplot as plt



# Add parent directories to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optomech.rollout import rollout_optomech_policy
from optomech.replay_buffers import *

# Import TD3 modular components
from models.td3_models import ImpalaActorLSTM, ImpalaCriticLSTM
from optomech.rl.td3_replay_buffer import TD3ReplayBufferLSTM, load_pretrained_encoder, load_pretrained_actor
from optomech.rl.sa_dataset_loader import load_sa_dataset_to_buffer, get_sa_dataset_info, get_sa_dataset_info

# Import AutoencoderConfig for unpickling saved checkpoints
try:
    from models.train_autoencoder import AutoencoderConfig
    # Register in __main__ namespace for unpickling
    sys.modules['__main__'].AutoencoderConfig = AutoencoderConfig
except ImportError:
    pass


class StepTimer:
    """
    Hierarchical timing instrumentation for training loop.
    Tracks cumulative and per-step timing for various operations.
    
    Uses a two-level hierarchy:
    - Level 1 (categories): Major phases that DON'T overlap (env_interaction, training, logging)
    - Level 2 (operations): Specific operations within each category
    
    The table shows only leaf operations that don't contain other timed operations,
    avoiding double-counting confusion.
    """
    def __init__(self, enabled: bool = True, print_interval: int = 100, writer=None):
        self.enabled = enabled
        self.print_interval = print_interval
        self.writer = writer  # TensorBoard SummaryWriter
        self.timings = {}  # name -> list of durations
        self.current_starts = {}  # name -> start time
        self.step_count = 0
        self.global_step = 0  # Track global step for TensorBoard logging
        
        # Define the hierarchy for display purposes
        # Format: category -> [operations]
        self.categories = {
            "ENV INTERACTION": [
                "action_sample_random",
                "action_sample_policy", 
                "action_add_noise",
                "hidden_state_warmup",
                "hidden_state_update",
                "env_step",
                "obs_normalize",
                "episode_data_append",
                "replay_buffer_push",
                "episode_reset",
            ],
            "TRAINING": [
                "rb_sample",
                "tensor_to_device",
                "target_actor_forward",
                "target_critic_forward", 
                "td_target_compute",
                "critic_forward",
                "critic_loss_compute",
                "critic_backward",
                "critic_grad_clip",
                "critic_optimizer_step",
                "actor_forward",
                "actor_loss_compute",
                "actor_backward",
                "actor_grad_clip",
                "actor_optimizer_step",
                "target_network_update",
            ],
            "LOGGING": [
                "tensorboard_write",
                "checkpoint_save",
            ],
        }
        
    def start(self, name: str):
        """Start timing an operation."""
        if self.enabled:
            self.current_starts[name] = time.perf_counter()
    
    def stop(self, name: str):
        """Stop timing and record duration."""
        if self.enabled and name in self.current_starts:
            duration = time.perf_counter() - self.current_starts[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            del self.current_starts[name]
            return duration
        return 0.0
    
    def step(self, global_step: int = None):
        """Increment step counter and print/log if at interval."""
        self.step_count += 1
        if global_step is not None:
            self.global_step = global_step
        if self.enabled and self.step_count % self.print_interval == 0:
            self.print_summary()
            self.log_to_tensorboard()
    
    def print_summary(self):
        """Print hierarchical timing summary."""
        if not self.timings:
            return
        
        print(f"\n{'═'*80}")
        print(f"⏱  TIMING BREAKDOWN (steps {self.step_count - self.print_interval + 1}-{self.step_count})")
        print(f"{'═'*80}")
        
        grand_total_time = 0.0
        category_totals = {}
        
        # First pass: calculate totals
        for category, operations in self.categories.items():
            category_total = 0.0
            for op in operations:
                if op in self.timings:
                    recent = self.timings[op][-self.print_interval:] if len(self.timings[op]) >= self.print_interval else self.timings[op]
                    category_total += sum(recent)
            category_totals[category] = category_total
            grand_total_time += category_total
        
        # Also check for uncategorized timings
        all_categorized = set()
        for ops in self.categories.values():
            all_categorized.update(ops)
        uncategorized_total = 0.0
        uncategorized_ops = []
        for name in self.timings:
            if name not in all_categorized:
                recent = self.timings[name][-self.print_interval:] if len(self.timings[name]) >= self.print_interval else self.timings[name]
                uncategorized_total += sum(recent)
                uncategorized_ops.append(name)
        if uncategorized_total > 0:
            category_totals["OTHER"] = uncategorized_total
            grand_total_time += uncategorized_total
        
        # Print header
        print(f"{'Operation':<40} {'Mean':>8} {'Std':>8} {'Total':>10} {'%':>6}")
        print(f"{'─'*40} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
        
        # Print each category
        for category, operations in self.categories.items():
            cat_total = category_totals.get(category, 0.0)
            if cat_total == 0:
                continue
                
            cat_pct = 100 * cat_total / grand_total_time if grand_total_time > 0 else 0
            print(f"\n{category} ({cat_total*1000:.1f}ms, {cat_pct:.1f}%)")
            print(f"{'─'*40}")
            
            # Sort operations by total time
            op_stats = []
            for op in operations:
                if op in self.timings:
                    recent = self.timings[op][-self.print_interval:] if len(self.timings[op]) >= self.print_interval else self.timings[op]
                    if recent:
                        mean_ms = np.mean(recent) * 1000
                        std_ms = np.std(recent) * 1000
                        total_ms = sum(recent) * 1000
                        op_stats.append((op, mean_ms, std_ms, total_ms, len(recent)))
            
            op_stats.sort(key=lambda x: -x[3])
            
            for op, mean_ms, std_ms, total_ms, count in op_stats:
                pct = 100 * (total_ms / 1000) / grand_total_time if grand_total_time > 0 else 0
                # Indent operations under category
                print(f"  {op:<38} {mean_ms:>7.2f} {std_ms:>7.2f} {total_ms:>9.1f} {pct:>5.1f}%")
        
        # Print uncategorized if any
        if uncategorized_ops:
            cat_total = category_totals.get("OTHER", 0.0)
            cat_pct = 100 * cat_total / grand_total_time if grand_total_time > 0 else 0
            print(f"\nOTHER ({cat_total*1000:.1f}ms, {cat_pct:.1f}%)")
            print(f"{'─'*40}")
            for op in uncategorized_ops:
                recent = self.timings[op][-self.print_interval:] if len(self.timings[op]) >= self.print_interval else self.timings[op]
                if recent:
                    mean_ms = np.mean(recent) * 1000
                    std_ms = np.std(recent) * 1000
                    total_ms = sum(recent) * 1000
                    pct = 100 * (total_ms / 1000) / grand_total_time if grand_total_time > 0 else 0
                    print(f"  {op:<38} {mean_ms:>7.2f} {std_ms:>7.2f} {total_ms:>9.1f} {pct:>5.1f}%")
        
        # Summary
        print(f"\n{'═'*80}")
        print(f"TOTAL TIMED: {grand_total_time*1000:.1f}ms over {self.print_interval} steps")
        print(f"AVG PER STEP: {grand_total_time*1000/self.print_interval:.2f}ms  |  THROUGHPUT: {self.print_interval/grand_total_time:.1f} steps/sec")
        print(f"{'═'*80}\n")
        
        # Clear old timings to prevent memory growth
        for name in self.timings:
            self.timings[name] = self.timings[name][-self.print_interval*2:]
    
    def log_to_tensorboard(self):
        """Log timing table as text to TensorBoard."""
        if not self.writer or not self.timings:
            return
        
        # Build the timing table as a markdown string
        lines = []
        lines.append(f"**TIMING BREAKDOWN (steps {self.step_count - self.print_interval + 1}-{self.step_count})**\n")
        lines.append("| Operation | Mean (ms) | Std (ms) | Total (ms) | % |")
        lines.append("|:----------|----------:|---------:|-----------:|--:|")
        
        grand_total_time = 0.0
        category_totals = {}
        
        # Calculate totals per category
        for category, operations in self.categories.items():
            category_total = 0.0
            for op in operations:
                if op in self.timings:
                    recent = self.timings[op][-self.print_interval:] if len(self.timings[op]) >= self.print_interval else self.timings[op]
                    category_total += sum(recent)
            category_totals[category] = category_total
            grand_total_time += category_total
        
        # Build table rows for each category
        for category, operations in self.categories.items():
            cat_total = category_totals.get(category, 0.0)
            if cat_total == 0:
                continue
            
            cat_pct = 100 * cat_total / grand_total_time if grand_total_time > 0 else 0
            lines.append(f"| **{category}** | | | **{cat_total*1000:.1f}** | **{cat_pct:.1f}%** |")
            
            # Sort operations by total time
            op_stats = []
            for op in operations:
                if op in self.timings:
                    recent = self.timings[op][-self.print_interval:] if len(self.timings[op]) >= self.print_interval else self.timings[op]
                    if recent:
                        mean_ms = np.mean(recent) * 1000
                        std_ms = np.std(recent) * 1000
                        total_ms = sum(recent) * 1000
                        op_stats.append((op, mean_ms, std_ms, total_ms))
            
            op_stats.sort(key=lambda x: -x[3])
            
            for op, mean_ms, std_ms, total_ms in op_stats:
                pct = 100 * (total_ms / 1000) / grand_total_time if grand_total_time > 0 else 0
                lines.append(f"| &nbsp;&nbsp;{op} | {mean_ms:.2f} | {std_ms:.2f} | {total_ms:.1f} | {pct:.1f}% |")
        
        # Add summary
        if grand_total_time > 0:
            throughput = self.print_interval / grand_total_time
            avg_step_ms = grand_total_time * 1000 / self.print_interval
            lines.append(f"| | | | | |")
            lines.append(f"| **TOTAL** | | | **{grand_total_time*1000:.1f}** | |")
            lines.append(f"\n**Avg/step:** {avg_step_ms:.2f}ms | **Throughput:** {throughput:.1f} steps/sec")
        
        # Write to TensorBoard as text
        table_text = "\n".join(lines)
        self.writer.add_text("timing/breakdown", table_text, self.global_step)
    
    def reset(self):
        """Reset all timings."""
        self.timings = {}
        self.current_starts = {}


@dataclass
class Args:
    config: Optional[str] = None
    """Path to JSON config file (overrides defaults, but is overridden by command-line args)"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if using cuda, set the gpu"""
    gpu_list: int = 0
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    num_envs: int = 1
    """The number of environments to create."""
    async_env: bool = False
    """Whether to use an AsynchronousVectorEnv"""
    subproc_env: bool = False
    """Whether to use a SubprocVectorEnv"""
    model_save_interval: int = 10_000
    """The interval between saving model weights"""
    writer_interval: int = 1000
    """The interval between recording to tensorboard"""
    num_eval_rollouts: int = 1
    """The number of rollouts to perform for each evaluation."""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    # learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    critic_learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.004
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 100
    """minimum number of sequences in replay buffer before learning starts"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    policy_noise: float = 0.2
    """std of noise added to target policy actions for TD3 target smoothing"""
    decay_rate: float = 0.995
    """Decay rate for noise decay"""
    action_scale: float = 1.0
    """The scale of the actors actions"""
    reward_scale: float = 1.0
    """The scale of the reward"""
    l2_reg: float = 0.0
    """The scale of the L2 regularization"""
    l1_reg: float = 0.0
    """The scale of the L1 regularization"""
    max_grad_norm: float = 1.0
    """The maximum gradient norm"""
    clip_gradients: bool = False
    """Whether to clip gradients and log gradient norms (expensive when writer_interval is low)"""
    prefetch_batches: bool = False
    """Use pinned memory + CUDA streams to prefetch next batch while training (CUDA only)"""
    use_q_bias: bool = False
    """If toggled, compute q bias in the critic model."""
    normalize_returns: bool = False
    """If toggled, normalize the returns in the critic model."""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    replay_buffer_load_path: str = None
    """the path to load the replay buffer from"""

    # Actor model parameters
    actor_type: str = "vanilla"
    """Type of actor model to use."""
    actor_channel_scale: int = 16
    """The scale of the actor model channels."""
    actor_fc_scale: int = 64
    """The scale of the actor model fully connected layers."""
    low_dim_actor: bool = False
    """Whether the actor model is visual."""
    use_multiscale_head: bool = False
    """If toggled and using impala, it will have a multi-scale head"""

    # QNetwork model parameters
    critic_type: str = "vanilla"
    """Type of QNetwork model to use."""
    qnetwork_channel_scale: int = 16
    """The scale of the QNetwork model channels."""
    qnetwork_fc_scale: int = 64
    """The scale of the QNetwork model fully connected layers."""
    lstm_hidden_dim: int = 128
    """The scale of the QNetwork model fully connected layers."""
    low_dim_qnetwork: bool = False
    """Whether the qnetwork model is visual."""


    # Custom Algorthim Arguments
    """Which prelearning sample strategy to use (e.g., 'scales', 'normal')"""
    prelearning_sample: str = ""
    """How many steps to optimize the q function before actor training starts"""
    actor_training_delay: int = 0
    """How many steps to wait before populating the RB"""
    experience_sampling_delay: int = 0
    """Whether or not to use target smoothing"""
    target_smoothing: bool = False
    """How long the sequence length is for the LSTM"""
    tbptt_seq_len: int = 16

    # Pretrained model arguments
    pretrained_encoder_path: Optional[str] = None
    """Path to pretrained autoencoder checkpoint"""
    freeze_encoder: bool = True
    """Whether to freeze pretrained encoder"""
    pretrained_actor_path: Optional[str] = None
    """Path to pretrained actor checkpoint"""
    freeze_actor_encoder: bool = True
    """Whether to freeze encoder in pretrained actor"""
    
    # Dataset pre-loading arguments
    dataset_path: Optional[str] = None
    """Path to trajectory dataset for pre-loading replay buffer"""
    num_preload_trajectories: Optional[int] = None
    """Number of trajectories to load from dataset (None = all)"""
    use_td3_replay_buffer: bool = False
    """Whether to use the new TD3ReplayBufferLSTM for dataset loading"""
    replay_buffer_cutoff_step: Optional[int] = None
    """Step after which no more writes are made to replay buffer (None = no cutoff)"""
    
    # SA Dataset pre-loading arguments
    sa_dataset_path: Optional[str] = None
    """Path to SA dataset directory for pre-loading replay buffer (e.g., datasets/sa_dataset_1m)"""
    sa_action_type: str = "perfect_actions"
    """Action type to use from SA dataset: 'perfect_actions', 'sa_actions', 'perfect_incremental_actions', 'sa_incremental_actions'"""
    sa_max_sequences: Optional[int] = None
    """Maximum number of sequences to load from SA dataset (None = no limit)"""
    sa_max_episodes: Optional[int] = None
    """Maximum number of episodes to load from SA dataset (None = no limit)"""

    # visual pendulum parameters
    # learning_rate: float = 3e-4
    # """the learning rate of the optimizer"""
    # buffer_size: int = int(1e6)
    # """the replay memory buffer size"""
    # gamma: float = 0.99
    # """the discount factor gamma"""
    # tau: float = 0.004
    # """target smoothing coefficient (default: 0.005)"""
    # batch_size: int = 64
    # """the batch size of sample from the reply memory"""
    # exploration_noise: float = 0.1
    # """the scale of exploration noise"""
    # learning_starts: int = 256
    # """timestep to start learning"""
    # policy_frequency: int = 4
    # """the frequency of training policy (delayed)"""
    # noise_clip: float = 0.5
    # """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Timing instrumentation
    timing_enabled: bool = True
    """Enable detailed timing instrumentation for performance profiling"""
    timing_interval: int = 100
    """How often to print timing summary (in steps)"""

    # Environment specific arguments
    """Class for holding all arguments for the script."""
    gpu_list: str = "0"
    """The list of GPUs to use."""
    render: bool = False
    """Whether to render the environment."""
    report_time: bool = False
    """Whether to report time statistics."""
    action_type: str = "none"
    """The type of action to use."""
    object_type: str = "binary"
    """The type of object to use."""
    aperture_type: str = "elf"
    """The type of aperture to use."""
    max_episode_steps: int = 100


    """Toggle to enable discrete control."""
    discrete_control: bool = False
    """The number of discrete control steps."""
    discrete_control_steps: int = 128
    """Toggle to enable incremental control."""
    incremental_control: bool = False
    """Toggle to enable agent control of tensioners."""
    command_tensioners: bool = False
    """Toggle to enable agent control of secondaries."""
    command_secondaries: bool = False
    """Toggle to enable agent control of tip/tilt for large mirrors."""
    command_tip_tilt: bool = False
    """Toggle to enable agent control of dm."""
    command_dm: bool = False



    """ The type of observation to model 'image_only' or 'image_action'."""
    observation_mode: bool = "image_only"

    """The type of aperture to use."""
    ao_loop_active: bool = False
    """The maximum number of steps per episode."""
    num_episodes: int = 1
    """The number of episodes to run."""
    num_atmosphere_layers: int = 0
    """The number of atmosphere layers."""
    reward_threshold: float = 25.0
    """The reward threshold to reach."""
    num_steps: int = 16
    """The number of steps to take."""
    silence: bool = False
    """Whether to silence the output."""
    optomech_version: str = "test"
    """The version of optomech to use."""
    reward_function: str = "strehl"
    """The reward function to use."""
    render_frequency: int = 1
    """The frequency of rendering."""
    ao_interval_ms: float = 1.0
    """The interval between AO updates."""
    control_interval_ms: float = 2.0
    """The interval between control updates."""
    init_differential_motion: bool = False
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = False
    """Whether to simulate differential motion."""
    frame_interval_ms: float = 4.0
    """The interval between frames."""
    decision_interval_ms: float = 8.0
    """The interval between decisions."""
    focal_plane_image_size_pixels: int = 256
    """The size of the focal plane image in pixels."""
    render_dpi: float = 500.0
    """The DPI for rendering."""
    record_env_state_info: bool = False
    """Whether to record environment state information."""
    write_env_state_info: bool = False
    """Whether to write environment state information."""
    state_info_save_dir: str = "./tmp/"
    """The directory to save state information."""
    randomize_dm: bool = False
    """Whether to randomize the DM."""
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    """The file for the extended object image."""
    extended_object_distance: str = None
    """The distance to the extended object."""
    extended_object_extent: str = None
    """The extent of the extended object."""
    observation_window_size: int = 2**1
    """The size of the observation window."""
    num_tensioners: int = 16
    """The number of tensioners."""
    model_wind_diff_motion: bool = False
    """Whether to model wind differential motion."""
    model_gravity_diff_motion: bool = False
    """Whether to model gravity differential motion."""
    model_temp_diff_motion: bool = False
    """Whether to model temperature differential motion."""




# Deprecated: ImpalaActor class removed (now imported from models.td3_models)
# Deprecated: ImpalaCritic class removed (now imported from models.td3_models)
# Deprecated: Weight initialization functions removed (now in models.td3_models)


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


def log_weights_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        logger.add_histogram(tag + "/weight", value.cpu(), step)


def sample_normal_action(action_space, std_dev=0.1):
    """
    Samples an action from a normal distribution, clipped to fit within the action space bounds.
    
    Parameters:
        action_space (gymnasium.Space): The action space of the environment.

    Returns:
        np.ndarray: A sampled action within the bounds of the action space.
    """
    # Get action space bounds and shape
    low, high = action_space.low, action_space.high
    shape = action_space.shape

    # Calculate mean and standard deviation for the normal distribution
    mean = 0
    # Sample actions and clip to action space bounds
    epsilon = 1e-7
    action = np.clip(np.random.normal(loc=mean, scale=std_dev, size=shape), action_space.low + epsilon, action_space.high - epsilon)
    action = action.astype(action_space.dtype)
    return action

def make_env(env_id, idx, capture_video, run_name, flags):

    if env_id == "optomech-v1":

        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", **vars(flags))
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **vars(flags))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.action_space.seed(seed)
            return env

        return thunk

    else:

        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.action_space.seed(seed)
            return env

        return thunk
    
class OUNoiseTorch:
    def __init__(self, 
                 action_dim, 
                 mu=0.0, 
                 theta=0.15, 
                 sigma_initial=0.2, 
                 min_sigma=0.05, 
                 decay_rate=0.995,
                 auto_decay=True,
                 device='cpu'):
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
        """Reset the internal state to mean `mu`."""
        self.state = self.mu_tensor.clone()

    def sample(self):
        """Generate a noise sample and decay sigma if auto_decay is enabled."""
        noise = self.theta * (self.mu_tensor - self.state) + \
                self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state += noise
        if self.auto_decay:
            self.decay()
        return self.state

    def decay(self):
        """Apply exponential decay to sigma."""
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

if __name__ == "__main__":


    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    

    args = tyro.cli(Args)
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Get command-line overrides (non-default values)
        import sys
        cli_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                cli_args.add(arg.split('=')[0].replace('--', '').replace('-', '_'))
        
        # Apply config values, but don't override command-line args
        num_applied = 0
        for key, value in config.items():
            if hasattr(args, key) and key not in cli_args and key != 'config':
                current_value = getattr(args, key)
                # Only override if it's still the default value
                setattr(args, key, value)
                num_applied += 1
        print(f"✓ Loaded {num_applied} config values from {args.config}")
    
    
    # Register our custom optomech environment.
    gym.envs.registration.register(
        id='optomech-v1',
        entry_point='optomech.optomech:OptomechEnv',
        # max_episode_steps=4,
        # reward_threshold=flags.reward_threshold,
    )

    # Register our custom VisualPendulum environment.
    gym.envs.registration.register(
        id='VisualPendulum-v1',
        entry_point='visual_pendulum:VisualPendulumEnv',
        max_episode_steps=args.max_episode_steps,
    )


    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Add a summary writer.
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Write the arguements of this run to a file.
    args_store_path = f"./runs/{run_name}/args.json"
    with open(args_store_path, "w") as f:
        json.dump(vars(args), f)

    # Check if MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(int(args.gpu_list))
        print(f"✓ Device: CUDA (GPU {args.gpu_list})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Device: MPS")
    else:
        device = torch.device("cpu")
        print("✓ Device: CPU")

    # env setup
    if args.subproc_env:
        envs = gym.vector.SubprocVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
        print(f"✓ Environment: SubprocVectorEnv ({args.num_envs} envs)")
    if args.async_env:
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
        print(f"✓ Environment: AsyncVectorEnv ({args.num_envs} envs)")
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
        print(f"✓ Environment: SyncVectorEnv ({args.num_envs} envs)")
    
    if args.actor_type == "impala":
        
        # Load pretrained encoder if specified
        base_encoder = None
        if args.pretrained_encoder_path:
            base_encoder = load_pretrained_encoder(
                args.pretrained_encoder_path,
                None,  # TODO: Pass autoencoder class
                device=device,
                freeze=args.freeze_encoder
            )
            print(f"✓ Loaded pretrained encoder ({'frozen' if args.freeze_encoder else 'trainable'})")
        
        # Deep copy encoder for actor (or None if no pretrained encoder)
        actor_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        
        # Initialize actor with TD3 model
        # Note: Preprocessing now handled in replay buffer, not in model
        actor = ImpalaActorLSTM(
            envs=envs,
            device=device,
            encoder=actor_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale
        ).to(device)
        
        # Load pretrained actor if specified
        if args.pretrained_actor_path:
            checkpoint = torch.load(args.pretrained_actor_path, map_location=device, weights_only=False)
            actor.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            print(f"✓ Loaded pretrained actor")
            
            # Optionally freeze encoder
            if args.freeze_actor_encoder and actor.visual_encoder is not None:
                for param in actor.visual_encoder.parameters():
                    param.requires_grad = False
                print("  └─ Encoder frozen")
        
        # Deep copy encoder for target actor
        target_actor_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        
        # Create target actor
        target_actor = ImpalaActorLSTM(
            envs=envs,
            device=device,
            encoder=target_actor_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale
        ).to(device)
        
        # Skip TorchScript when using pretrained encoder (complex models may not be JIT-compatible)
        if args.pretrained_encoder_path:
            scripted_actor = actor
            print("⚠ Skipping TorchScript compilation (pretrained encoder not compatible)")
        else:
            scripted_actor = torch.jit.script(actor)

    # Potential-based reward shaping https://arxiv.org/pdf/2502.01307
    if args.use_q_bias:
        reward_sample_episodes = 2
        episode_rewards = []
        # Sample some random rewards to compute the q bias.
        # This is a hacky way to get the expected reward.
        # We sample 10 episodes of random actions and average the rewards.
        print("Sampling random rewards to compute Q-bias...")
        for _ in range(reward_sample_episodes):
            obs, _ = envs.reset(seed=args.seed)
            episode_reward_sum = 0.0
            for t in range(args.max_episode_steps):
                first_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                _, rewards, _, _, _ = envs.step(first_actions)
                episode_reward_sum = np.mean(rewards) + args.gamma * episode_reward_sum
            episode_rewards.append(episode_reward_sum)
        expected_reward = np.mean(episode_rewards) * args.reward_scale
        # expected_reward = -13.5 * args.reward_scale
        q_bias = expected_reward * ((1 - (args.gamma ** args.max_episode_steps)) / (1 - args.gamma))
        print(f"✓ Q-bias computed: {q_bias:.2f} (expected reward: {expected_reward:.2f})")
        # args.normalize_returns = False
        if args.normalize_returns:
            # If we are normalizing returns, we need to scale the q bias.
            args.reward_scale = 1.0 / q_bias
            q_bias = 0.0
    else:
        q_bias = 0.0



    
    if args.critic_type  == "impala":
        
        # Deep copy encoders for each critic (4 total: qf1, qf1_target, qf2, qf2_target)
        qf1_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        qf1_target_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        qf2_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        qf2_target_encoder = copy.deepcopy(base_encoder) if base_encoder is not None else None
        
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf1 = ImpalaCriticLSTM(
            envs=envs,
            device=device,
            encoder=qf1_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            q_bias=q_bias
        ).to(device)
        qf1_target = ImpalaCriticLSTM(
            envs=envs,
            device=device,
            encoder=qf1_target_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            q_bias=q_bias
        ).to(device)
        
        _ = torch.randn(1000)
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        _ = torch.randn(1000)
        qf2 = ImpalaCriticLSTM(
            envs=envs,
            device=device,
            encoder=qf2_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            q_bias=q_bias
        ).to(device)
        qf2_target = ImpalaCriticLSTM(
            envs=envs,
            device=device,
            encoder=qf2_target_encoder,
            lstm_hidden_dim=args.lstm_hidden_dim,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            q_bias=q_bias
        ).to(device)
        
        for p in qf2.parameters():
            p.data += 1e-3 * torch.randn_like(p)
        for p in qf2_target.parameters():
            p.data += 1e-3 * torch.randn_like(p)

    else:
        raise ValueError("Invalid critic type specified.")

    actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in qf1.parameters() if p.requires_grad)
    print(f"✓ Models initialized: Actor ({actor_params:,} params), Critic ({critic_params:,} params)")
    summary(actor, )
    summary(qf1,)

    # Test if qf1 and qf2 have any identical parameters.
    any_different = False
    for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
        if not(p1.data.equal(p2.data)):
            any_different = True
            break

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    qf1_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_learning_rate)
    qf2_optimizer = optim.Adam(list(qf2.parameters()), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    envs.single_observation_space.dtype = np.float32

    # Initialize replay buffer - use TD3 version if flag is set
    if args.use_td3_replay_buffer:
        # Enable preprocessing if using pretrained encoder
        preprocess_obs = args.pretrained_encoder_path is not None
        rb = TD3ReplayBufferLSTM(
            capacity=args.buffer_size,
            preprocess_observations=preprocess_obs,
            normalize_obs=True,  # Always normalize uint16 data
            obs_crop_size=256 if preprocess_obs else None,  # Crop to 256 to match autoencoder
            obs_log_scale=preprocess_obs  # Log-scale if using pretrained encoder
        )
        
        print(f"✓ Replay buffer: TD3ReplayBufferLSTM (capacity={args.buffer_size:,})")
        if preprocess_obs:
            print(f"  └─ Preprocessing enabled: normalize=True, crop=256, log_scale=True")
        
        # Pre-load from dataset if specified (legacy format)
        if args.dataset_path:
            print(f"  └─ Pre-loading from {args.dataset_path}...")
            rb.load_from_dataset(
                dataset_path=args.dataset_path,
                actor_model=actor,
                qf1_model=qf1,
                qf2_model=qf2,
                num_trajectories=args.num_preload_trajectories,
                sequence_length=args.tbptt_seq_len,
                device=device,
                verbose=True
            )
            print(f"  └─ Pre-loaded {len(rb)} sequences")
        
        # Pre-load from SA dataset if specified
        if args.sa_dataset_path:
            print(f"\n📂 Pre-loading from SA dataset: {args.sa_dataset_path}")
            
            # Show dataset info first
            try:
                sa_info = get_sa_dataset_info(args.sa_dataset_path)
                print(f"  └─ Dataset contains {sa_info['total_samples']:,} samples from {sa_info['num_episodes']:,} episodes")
                print(f"  └─ Observation shape: {sa_info['observation_shape']}")
                print(f"  └─ Action shape: {sa_info['action_shape']}")
            except Exception as e:
                print(f"  └─ Warning: Could not get dataset info: {e}")
            
            # Load SA dataset into replay buffer
            sequences_added = load_sa_dataset_to_buffer(
                replay_buffer=rb,
                dataset_path=args.sa_dataset_path,
                actor_model=actor,
                qf1_model=qf1,
                qf2_model=qf2,
                sequence_length=args.tbptt_seq_len,
                action_type=args.sa_action_type,
                device=device,
                max_sequences=args.sa_max_sequences,
                max_episodes=args.sa_max_episodes,
                verbose=True
            )
            print(f"✅ Pre-loaded {sequences_added} sequences from SA dataset")
            print(f"   Total buffer size: {len(rb)} sequences\n")
            
    else:
        rb = ReplayBufferWithHiddenStatesBPTT(args.buffer_size)
        print(f"✓ Replay buffer: ReplayBufferWithHiddenStatesBPTT (capacity={args.buffer_size:,})")

    if args.replay_buffer_load_path:
        if not os.path.exists(args.replay_buffer_load_path):
            raise FileNotFoundError(f"Replay buffer load path {args.replay_buffer_load_path} does not exist.")
        if os.path.isdir(args.replay_buffer_load_path):
            # If the path is a directory, load the replay buffer from the latest file in the directory.
            files = [f for f in os.listdir(args.replay_buffer_load_path) if f.endswith('.pt')]
            if not files:
                raise FileNotFoundError(f"No .pt files found in {args.replay_buffer_load_path}.")

            rb.restore(args.replay_buffer_load_path)
            print(f"  └─ Loaded {len(rb.buffer):,} existing elements")

    eval_dict = dict()
    for eval_rollout in range(args.num_eval_rollouts):

        eval_dict[eval_rollout] = dict()
        rollout_seed = np.random.randint(0, 999999)
        eval_dict[eval_rollout]["seed"] = rollout_seed
        env_kwargs = {"seed": rollout_seed}
        zero_policy_returns = rollout_optomech_policy(
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            env_kwargs=env_kwargs,
                            prelearning_sample="zeros"
                        )
        eval_dict[eval_rollout]["zero_policy_returns"] = zero_policy_returns
        random_policy_returns = rollout_optomech_policy(
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            env_kwargs=env_kwargs,
                        )
        eval_dict[eval_rollout]["random_policy_returns"] = random_policy_returns
        eval_dict[eval_rollout]["on_policy_returns"] = dict()

    # TODO: Add a dud check here.

    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed)
    
    # DIAGNOSTIC: Check observation data type and range
    print("\n" + "="*60)
    print("🔍 OBSERVATION DIAGNOSTICS")
    print("="*60)
    obs_array = np.array(obs)
    print(f"Observation shape: {obs_array.shape}")
    print(f"Observation dtype: {obs_array.dtype}")
    print(f"Observation min: {obs_array.min()}")
    print(f"Observation max: {obs_array.max()}")
    print(f"Observation mean: {obs_array.mean():.4f}")
    print(f"First pixel values (sample): {obs_array.flatten()[:10]}")
    
    # Check if uint16
    if obs_array.dtype == np.uint16:
        print("✓ Detected uint16 data (will normalize by 65535)")
    elif obs_array.dtype == np.uint8:
        print("✓ Detected uint8 data (will normalize by 255)")
    elif obs_array.dtype in [np.float32, np.float64]:
        if obs_array.max() > 256:
            print("⚠️  Float data with max > 256 (treating as unnormalized uint16)")
        elif obs_array.max() > 1.0:
            print("⚠️  Float data with max > 1.0 (treating as unnormalized uint8)")
        else:
            print("✓ Float data already normalized to [0,1]")
    print("="*60 + "\n")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # We're going to accumulate the trajectory data in these lists.
    episode_state = list()
    episode_action = list()
    episode_last_action = list()
    episode_reward = list()
    episode_last_reward = list()
    episode_next_state = list()
    episode_done = list()

    # Initialize the global step and episode return lists.
    global_step = 0
    best_train_episode_return = -np.inf
    train_episodic_return_list = list()
    best_eval_episode_return = -np.inf

    # During this phase, we run exactly full pass through the env and models.
    first_actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])
    obs, rewards, _, _, _ = envs.step(first_actions)

    # Convert uint8/uint16 images to float32 - check numpy dtype directly
    if obs.dtype == np.uint8:
        obs = (obs / 255.0).astype(np.float32)
    elif obs.dtype == np.uint16:
        obs = (obs / 65535.0).astype(np.float32)
    rewards = args.reward_scale * rewards

    prior_actions = first_actions
    prior_rewards = rewards
    first_step_reward = rewards
    
    # Convert to tensors once
    obs_tensor = torch.from_numpy(obs).to(device, dtype=torch.float32)
    prior_actions_tensor = torch.from_numpy(prior_actions).to(device, dtype=torch.float32)
    prior_rewards_tensor = torch.from_numpy(prior_rewards).to(device, dtype=torch.float32)
    
    actions, new_actor_hidden = actor(
            obs_tensor,
            prior_actions_tensor,
            prior_rewards_tensor,
            actor.get_zero_hidden()
        )
    actor_hidden = new_actor_hidden
    
    first_actions_tensor = torch.from_numpy(first_actions).to(device, dtype=torch.float32)
    
    qf1_a_values, qf1_hidden = qf1(
        obs_tensor,
        actions.to(device),
        first_actions_tensor,
        prior_rewards_tensor,
        qf1.get_zero_hidden()
    )
    qf2_a_values, qf2_hidden = qf2(
        obs_tensor,
        actions.to(device),
        first_actions_tensor,
        prior_rewards_tensor,
        qf2.get_zero_hidden()
    )

    # Store the initial hidden states for the actor and critics for the first step.
    initial_actor_hidden = (new_actor_hidden[0].detach().clone(),
                            new_actor_hidden[1].detach().clone())
    initial_qf1_hidden = (qf1_hidden[0].detach().clone(),
                          qf1_hidden[1].detach().clone())
    initial_qf2_hidden = (qf2_hidden[0].detach().clone(),
                          qf2_hidden[1].detach().clone())

    # Initialize the exploration noise generator.
    noise_generator = OUNoiseTorch(
                 action_dim=envs.single_action_space.shape[0], 
                 mu=0.0, 
                 theta=0.15, 
                 sigma_initial=args.exploration_noise, 
                 min_sigma=args.exploration_noise * 0.01, 
                 decay_rate=args.decay_rate,
                 auto_decay=False,
                 device='cpu')

    # ============ Batch Prefetching Setup ============
    # Helper function to prepare batch tensors from replay buffer sample
    def prepare_batch_tensors(rb_sample, device, non_blocking=False):
        """Convert replay buffer sample to GPU tensors, optionally with non-blocking transfer."""
        (actor_hidden_batch, qf1_hidden_batch, qf2_hidden_batch,
         observations_batch, actions_batch_batch, prior_actions_batch,
         rewards_batch, prior_rewards_batch, next_observations_batch, dones_batch) = rb_sample
        
        # Convert to tensors - use pin_memory for async transfer when prefetching
        if non_blocking:
            observations_batch = torch.from_numpy(np.asarray(observations_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            actions_batch_batch = torch.from_numpy(np.asarray(actions_batch_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            prior_actions_batch = torch.from_numpy(np.asarray(prior_actions_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            rewards_batch = torch.from_numpy(np.asarray(rewards_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            prior_rewards_batch = torch.from_numpy(np.asarray(prior_rewards_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            next_observations_batch = torch.from_numpy(np.asarray(next_observations_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            dones_batch = torch.from_numpy(np.asarray(dones_batch)).pin_memory().to(device, dtype=torch.float32, non_blocking=True)
        else:
            observations_batch = torch.from_numpy(np.asarray(observations_batch)).to(device, dtype=torch.float32)
            actions_batch_batch = torch.from_numpy(np.asarray(actions_batch_batch)).to(device, dtype=torch.float32)
            prior_actions_batch = torch.from_numpy(np.asarray(prior_actions_batch)).to(device, dtype=torch.float32)
            rewards_batch = torch.from_numpy(np.asarray(rewards_batch)).to(device, dtype=torch.float32)
            prior_rewards_batch = torch.from_numpy(np.asarray(prior_rewards_batch)).to(device, dtype=torch.float32)
            next_observations_batch = torch.from_numpy(np.asarray(next_observations_batch)).to(device, dtype=torch.float32)
            dones_batch = torch.from_numpy(np.asarray(dones_batch)).to(device, dtype=torch.float32)
        
        return (actor_hidden_batch, qf1_hidden_batch, qf2_hidden_batch,
                observations_batch, actions_batch_batch, prior_actions_batch,
                rewards_batch, prior_rewards_batch, next_observations_batch, dones_batch)

    # Initialize prefetch stream for CUDA async data loading
    prefetch_stream = None
    prefetched_batch = None
    if args.prefetch_batches and device.type == 'cuda':
        prefetch_stream = torch.cuda.Stream()
        print(f"✓ Batch prefetching enabled (CUDA stream created)")
    elif args.prefetch_batches:
        print(f"⚠ Batch prefetching requested but not available on {device.type} (requires CUDA)")

    # Initialize timing instrumentation
    timer = StepTimer(enabled=args.timing_enabled, print_interval=args.timing_interval, writer=writer)
    if args.timing_enabled:
        print(f"✓ Timing instrumentation enabled (interval: {args.timing_interval} steps)")

    print(f"\n{'='*60}")
    print(f"Starting training: {args.total_timesteps:,} steps")
    print(f"{'='*60}\n")

    # Track when learning actually starts (in terms of global_step).
    # This is set when the replay buffer first has enough sequences.
    learning_started_at_step = None
    for iteration in range(args.total_timesteps):

        step_time = time.time()

        # First, check to see if we need to save the model.
        if args.save_model:
            # Check that we've completed at least one post-learning episode.
            # Use learning_started_at_step to know when learning actually began.
            if learning_started_at_step is not None and global_step > (learning_started_at_step + args.max_episode_steps):

                # If mean reward of recent episodes improved, save the model.
                if np.mean(train_episodic_return_list) > best_train_episode_return:

                    # Update the best mean reward and save the model.
                    best_train_episode_return = np.mean(train_episodic_return_list)

                    # First, delete the prior best model if it exists...
                    best_models_save_path = f"./runs/{run_name}/best_models"
                    model_path = f"{best_models_save_path}/best_train_policy.pt"
                    if os.path.exists(model_path):
                        os.remove(model_path)


                    # ...then save the best train model.
                    if args.pretrained_encoder_path:
                        # Save regular checkpoint instead of scripted model
                        torch.save({
                            'model_state_dict': actor.state_dict(),
                            'encoder_frozen': args.freeze_encoder
                        }, model_path)
                    else:
                        scripted_actor = torch.jit.script(actor)
                        scripted_actor.save(model_path)
                    print(f"✓ Saved best model (mean return: {best_train_episode_return:.2f}) [step {global_step}]")

                    # Now, load the model you just saved to roll it out, ...
                    model = torch.load(model_path, weights_only=False)
                    model.eval()
                    from rollout import rollout_optomech_policy

                    # ...delete any prior rollouts...
                    rollouts_path = f"{best_models_save_path}/rollouts"
                    if os.path.exists(rollouts_path):
                        shutil.rmtree(rollouts_path)

                    # ...and create a new directory for the rollouts...
                    Path(rollouts_path).mkdir(parents=True, exist_ok=True)

                    # ...then roll out the model, accumulating returns.
                    episodic_returns_list = list()
                    zero_policy_returns_list = list()
                    random_policy_returns_list = list()

                    for i, eval_rollout_dict in eval_dict.items():

                        print(f"Rollout episode: {i}")

                        eval_save_path = f"{rollouts_path}/"

                        env_kwargs = {"seed": eval_rollout_dict["seed"]}
                        
                        episodic_returns = rollout_optomech_policy(
                            model_path,
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            eval_save_path=eval_save_path,
                            env_kwargs=env_kwargs,
                        )

                        eval_rollout_dict["on_policy_returns"][iteration] = episodic_returns

                        episodic_returns_list.append(episodic_returns)
                        zero_policy_returns = eval_rollout_dict["zero_policy_returns"]
                        zero_policy_returns_list.append(zero_policy_returns)
                        random_policy_returns = eval_rollout_dict["random_policy_returns"]
                        random_policy_returns_list.append(random_policy_returns)

                    episodic_returns_array = np.array(episodic_returns_list).flatten()
                    zero_policy_returns_array = np.array(zero_policy_returns_list).flatten()
                    random_policy_returns_array = np.array(random_policy_returns_list).flatten()

                    mean_eval_episode_return = np.mean(episodic_returns_array)
                    mean_zero_return_advantage = np.mean(episodic_returns_array / zero_policy_returns_array)
                    mean_random_return_advantage = np.mean(episodic_returns_array / random_policy_returns_array)

                    writer.add_scalar("eval/best_policy_mean_returns", mean_eval_episode_return, iteration)
                    writer.add_scalar("eval/best_policy_zero_return_advantage", mean_zero_return_advantage, iteration)
                    writer.add_scalar("eval/best_policy_random_return_advantage", mean_random_return_advantage, iteration)


                    print(f"Mean eval episodic return: {mean_eval_episode_return}")

                # If we didn't just save them model, but it's time to save the model, do so.
                elif iteration % args.model_save_interval == 0:

                    use_torchsctipt = not args.pretrained_encoder_path  # Disable TorchScript for pretrained models
                    eval_save_path = f"runs/{run_name}/eval_{args.exp_name}_{str(iteration)}"
                    Path(eval_save_path).mkdir(parents=True, exist_ok=True)

                    if use_torchsctipt:

                        print("Saving model.")
                        model_path = f"{eval_save_path}/{args.exp_name}_{str(iteration)}_policy.pt"
                        scripted_actor = torch.jit.script(actor)
                        scripted_actor.save(model_path)
                        print(f"Torchscript model saved to {model_path}.")
                    else:

                        model_path = f"{eval_save_path}/{args.exp_name}_{str(iteration)}_policy.pth"
                        torch.save(actor, model_path)
                        print(f"Pytorch model saved to {model_path}.")

                    print("Loading model.")

                    if use_torchsctipt:
                        scripted_model = torch.jit.load(model_path)
                        scripted_model.eval()

                    else:
                        
                        model = torch.load(model_path, weights_only=False)
                        model.eval()

                    print(f"Evaluating model (iteration {iteration})...")

                    eval_save_path = f"runs/{run_name}/eval_{args.exp_name}_{str(iteration)}"
                    # ...then roll out the model, accumulating returns.
                    episodic_returns_list = list()
                    zero_policy_returns_list = list()
                    random_policy_returns_list = list()

                    for i, eval_rollout_dict in eval_dict.items():

                        env_kwargs = {"seed": eval_rollout_dict["seed"]}
                        
                        episodic_returns = rollout_optomech_policy(
                            model_path,
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            env_kwargs=env_kwargs,
                        )
                        eval_rollout_dict["on_policy_returns"][iteration] = episodic_returns
                        episodic_returns_list.append(episodic_returns)

                        zero_policy_returns = eval_rollout_dict["zero_policy_returns"]
                        zero_policy_returns_list.append(zero_policy_returns)

                        random_policy_returns = eval_rollout_dict["random_policy_returns"]
                        random_policy_returns_list.append(random_policy_returns)

                    episodic_returns_array = np.array(episodic_returns_list).flatten()
                    zero_policy_returns_array = np.array(zero_policy_returns_list).flatten()
                    random_policy_returns_array = np.array(random_policy_returns_list).flatten()

                    mean_eval_episode_return = np.mean(episodic_returns_array)
                    mean_zero_return_advantage = np.mean(episodic_returns_array / zero_policy_returns_array)
                    mean_random_return_advantage = np.mean(episodic_returns_array / random_policy_returns_array)

                    writer.add_scalar("eval/mean_returns", mean_eval_episode_return, iteration)
                    writer.add_scalar("eval/zero_return_advantage", mean_zero_return_advantage, iteration)
                    writer.add_scalar("eval/random_return_advantage", mean_random_return_advantage, iteration)
                
        # Actions are about to be replaced, so now we make the current actions the prior actions.
        if isinstance(actions, torch.Tensor):
            prior_actions = actions.detach().clone().cpu().numpy()
        else:
            prior_actions = actions.copy()

        # if isinstance(prior_actions, torch.Tensor):
        #     prior_actions = prior_actions.cpu().numpy()
        # else:
        #     prior_actions = prior_actions.copy()

    
        # ALGO LOGIC: put action logic here
        # If during prelearning, sample actions using the specified method.

        # Wait until the experience sampling delay has passed before sampling actions.
        if global_step >= args.experience_sampling_delay:

            # If there is a prelearning phase, use it.
            # Use exploration sampling until learning has started AND actor training delay has passed.
            prelearning_active = (learning_started_at_step is None) or (global_step < learning_started_at_step + args.actor_training_delay)
            if prelearning_active:

                timer.start("action_sample_random")
                # Cache action_scale on CPU to avoid repeated GPU transfers
                action_scale_np = actor.action_scale.cpu().numpy()

                if args.prelearning_sample == "scales":

                    if (iteration % args.max_episode_steps) == 0:

                        scales = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
                        action_scale = np.random.choice(scales)

                    # actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
                
                    actions = np.array([(action_scale_np * action_scale * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                    
                elif args.prelearning_sample == "normal":
                    
                    actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
                
                else:
                    
                    actions = np.array([(action_scale_np * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                timer.stop("action_sample_random")

                # Now we'll get the hidden states for the actor and critics, but ignore the outputs.
                timer.start("hidden_state_warmup")
                with torch.no_grad():

                    # Convert to tensors once and reuse
                    obs_tensor = torch.from_numpy(obs).to(device, dtype=torch.float32)
                    actions_tensor = torch.from_numpy(actions).to(device, dtype=torch.float32)
                    prior_actions_tensor = torch.from_numpy(prior_actions).to(device, dtype=torch.float32)
                    prior_rewards_tensor = torch.from_numpy(prior_rewards).to(device, dtype=torch.float32)

                    # If the actor takes prior actions and rewards as input, pass them in.
                    _, new_actor_hidden = actor(
                        obs_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        actor_hidden
                    )
                    
                    _, new_qf1_hidden = qf1(
                        obs_tensor,
                        actions_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        qf1_hidden
                    )
                    _, new_qf2_hidden = qf2(
                        obs_tensor,
                        actions_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        qf2_hidden
                    )

                # Update hidden states for next timestep
                actor_hidden = new_actor_hidden
                qf1_hidden = new_qf1_hidden
                qf2_hidden = new_qf2_hidden
                timer.stop("hidden_state_warmup")


            # Once prelearning is complete, sample actions using the actor.
            else:

                with torch.no_grad():

                    timer.start("action_sample_policy")
                    # Convert to tensors once and reuse
                    obs_tensor = torch.from_numpy(obs).to(device, dtype=torch.float32)
                    prior_actions_tensor = torch.from_numpy(prior_actions).to(device, dtype=torch.float32)
                    prior_rewards_tensor = torch.from_numpy(prior_rewards).to(device, dtype=torch.float32)

                    # If the actor takes prior actions and rewards as input, pass them in.
                    actions, new_actor_hidden = actor(
                        obs_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        actor_hidden
                    )
                    timer.stop("action_sample_policy")
                    
                    timer.start("action_add_noise")
                    # Sample and add noise to the actions, then clip to action space bounds.
                    noise = noise_generator.sample().to(device)
                    actions = actions + noise
                    
                    # Cache action_scale on CPU to avoid repeated transfers
                    action_scale_np = actor.action_scale.cpu().numpy()
                    actions = actions.cpu().numpy().clip(
                        action_scale_np * envs.single_action_space.low,
                        action_scale_np * envs.single_action_space.high)
                    timer.stop("action_add_noise")
                    
                    timer.start("hidden_state_update")
                    # Convert actions to tensor once for critic calls
                    actions_tensor = torch.from_numpy(actions).to(device, dtype=torch.float32)
                    
                    qf1_a_values, new_qf1_hidden = qf1(
                        obs_tensor,
                        actions_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        qf1_hidden
                    )
                    qf2_a_values, new_qf2_hidden = qf2(
                        obs_tensor,
                        actions_tensor,
                        prior_actions_tensor,
                        prior_rewards_tensor,
                        qf2_hidden
                    )
                    timer.stop("hidden_state_update")

                # Update hidden states for next timestep
                actor_hidden = new_actor_hidden
                qf1_hidden = new_qf1_hidden
                qf2_hidden = new_qf2_hidden


            # If using BPTT, we need to store the initial hidden states.
            # if initial_actor_hidden_in is None:
            #     initial_actor_hidden_in = actor.get_zero_hidden()
            #     initial_actor_hidden_out = initial_actor_hidden_in
            
            # Store the current rewards before generating a new transition.
            prior_rewards = rewards.copy()

            # TRY NOT TO MODIFY: execute the game and log data.
            timer.start("env_step")
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            timer.stop("env_step")

            timer.start("obs_normalize")
            # Convert uint8/uint16 images to float32 - check numpy dtype directly
            if obs.dtype == np.uint8:
                obs = (obs / 255.0).astype(np.float32)
            elif obs.dtype == np.uint16:
                obs = (obs / 65535.0).astype(np.float32)
                
            if next_obs.dtype == np.uint8:
                next_obs = (next_obs / 255.0).astype(np.float32)
            elif next_obs.dtype == np.uint16:
                next_obs = (next_obs / 65535.0).astype(np.float32)
            timer.stop("obs_normalize")
                
            # Rescale the rewards. By default, this is 1.0 and does nothing.
            rewards = args.reward_scale * rewards

            # If this is the first step of the episode, store the rewards.
            if first_step_reward is None:
                first_step_reward = rewards

            # Determine if we're past the learning start + actor training delay
            past_prelearning = (learning_started_at_step is not None) and (global_step > learning_started_at_step + args.actor_training_delay)

            # If we've hit a writing interval, log the data.
            if iteration % args.writer_interval == 0:
                if past_prelearning:
                    writer.add_scalar("online/action_mean", actions.mean().item(), global_step)
                    writer.add_scalar("online/action_std", actions.std().item(), global_step)
                    # writer.add_scalar("online/actions_l2", torch.norm(actions, p=2), global_step)
                    writer.add_scalar("online/reward_mean/", np.mean(rewards), global_step)
                    writer.add_scalar("online/reward_gain/", np.mean(rewards) - np.mean(first_step_reward), global_step)
                    writer.add_scalar("online/first_step_reward/", np.mean(first_step_reward), global_step)
                    writer.add_scalar("online/reward_std/", np.std(rewards), global_step)
                    writer.add_scalar("online/noise_l2/", torch.norm(noise, p=2), global_step)
                    for i, action in enumerate(actions):
                        for j, action_element in enumerate(action):
                            action_label = f"online/action_{i}_{j}"
                            writer.add_scalar(action_label, action_element, global_step)
                else:
                    writer.add_scalar("prelearning/action_mean", actions.mean().item(), global_step)
                    writer.add_scalar("prelearning/action_std", actions.std().item(), global_step)
                    # writer.add_scalar("prelearning/action_l2", torch.norm(actions, p=2), global_step)
                    writer.add_scalar("prelearning/reward_mean/", np.mean(rewards), global_step)
                    writer.add_scalar("prelearning/reward_gain/", np.mean(rewards) - np.mean(first_step_reward), global_step)
                    writer.add_scalar("prelearning/first_step_reward/", np.mean(first_step_reward), global_step)
                    writer.add_scalar("prelearning/reward_std/", np.std(rewards), global_step)
                    for i, action in enumerate(actions):
                        for j, action_element in enumerate(action):
                            action_label = f"prelearning/action_{i}_{j}"
                            writer.add_scalar(action_label, action_element, global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            
            # for idx, trunc in enumerate(truncations):
            #     print(idx)
            #     if trunc:
            #         real_next_obs = infos["final_observation"]

            # If using BPTT, we store the trajectory, but only push it to the replay buffer when the episode ends.

            # Everything pushed to the replay buffer must be a numpy array.
            # Some objects are tensors, while others are numpy arrays.
            # Some of the tensors may be on the GPU, so we need to move them to the CPU.
            # If the observations are tensors, convert them to numpy arrays.
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().detach().numpy()
            if isinstance(next_obs, torch.Tensor):
                real_next_obs = real_next_obs.cpu().detach().numpy()
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().detach().numpy()
            if isinstance(prior_actions, torch.Tensor):
                prior_actions = prior_actions.cpu().detach().numpy()
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().detach().numpy()
            if isinstance(prior_rewards, torch.Tensor):
                prior_rewards = prior_rewards.cpu().detach().numpy()

            # Convert the observations, actions, and rewards to numpy arrays.
            obs = np.array(obs, dtype=np.float32)
            real_next_obs = np.array(real_next_obs, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            prior_actions = np.array(prior_actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            prior_rewards = np.array(prior_rewards, dtype=np.float32)

            # Store the trajectory data.
            timer.start("episode_data_append")
            episode_state.append(obs)
            episode_action.append(actions)
            episode_last_action.append(prior_actions)
            episode_reward.append(rewards)
            episode_last_reward.append(prior_rewards)
            episode_next_state.append(real_next_obs)
            episode_done.append(terminations)
            timer.stop("episode_data_append")
            
            if len(episode_state) == args.tbptt_seq_len:

                # Check if we should add to replay buffer (respect cutoff if set)
                should_add_to_buffer = True
                if args.replay_buffer_cutoff_step is not None:
                    should_add_to_buffer = global_step < args.replay_buffer_cutoff_step
                
                if should_add_to_buffer:
                    timer.start("replay_buffer_push")
                    # Add batch dim to hidden states: [num_layers, hidden_dim] -> [num_layers, 1, hidden_dim]
                    # This ensures consistency with SA dataset hidden states
                    def add_batch_dim(hidden):
                        return (hidden[0].unsqueeze(1), hidden[1].unsqueeze(1))
                    rb.push(add_batch_dim(initial_actor_hidden),
                            add_batch_dim(initial_qf1_hidden),
                            add_batch_dim(initial_qf2_hidden),
                            episode_state,
                            episode_action,
                            episode_last_action,
                            episode_reward,
                            episode_last_reward,
                            episode_next_state,
                            episode_done)
                    timer.stop("replay_buffer_push")
                    
                    # Log replay buffer metrics to TensorBoard
                    buffer_size = len(rb)
                    writer.add_scalar("buffer/size", buffer_size, global_step)
                    writer.add_scalar("buffer/capacity_utilization", buffer_size / rb.capacity, global_step)
                    
                    # Print buffer status periodically
                    if len(rb) % 10 == 0 or len(rb) == 1:
                        print(f"  Replay buffer: {len(rb)} sequences | Step {global_step}")
                else:
                    # Still print when cutoff is reached for debugging
                    if global_step == args.replay_buffer_cutoff_step:
                        print(f"✓ Replay buffer cutoff reached at step {global_step} (buffer size: {len(rb)} sequences)")
                    
                    # Log buffer size even when cutoff reached (will show plateau)
                    buffer_size = len(rb)
                    writer.add_scalar("buffer/size", buffer_size, global_step)
                    writer.add_scalar("buffer/capacity_utilization", buffer_size / rb.capacity, global_step)

                episode_state = list()
                episode_action = list()
                episode_last_action = list()
                episode_reward = list()
                episode_last_reward = list()
                episode_next_state = list()
                episode_done = list()

                # Set the initial hidden states to the current value.
                initial_actor_hidden = (new_actor_hidden[0].detach().clone(),
                                        new_actor_hidden[1].detach().clone())
                initial_qf1_hidden = (new_qf1_hidden[0].detach().clone(),
                                      new_qf1_hidden[1].detach().clone())
                initial_qf2_hidden = (new_qf2_hidden[0].detach().clone(),
                                      new_qf2_hidden[1].detach().clone())
                
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if infos:

                for info in infos["final_info"]:

                    writer.add_scalar("episode/episodic_return", info["episode"]["r"], global_step)
                    episodic_return_gain = info["episode"]["r"] - (args.max_episode_steps * np.mean(first_step_reward) / args.reward_scale)
                    writer.add_scalar("episode/episodic_return_gain/", episodic_return_gain, global_step)
                    writer.add_scalar("episode/episodic_length", info["episode"]["l"], global_step)
                    
                    # Convert to scalars for printing
                    ep_len = int(info['episode']['l']) if hasattr(info['episode']['l'], '__iter__') else info['episode']['l']
                    ep_return = float(info['episode']['r']) if hasattr(info['episode']['r'], '__iter__') else info['episode']['r']
                    ep_gain = float(episodic_return_gain) if hasattr(episodic_return_gain, '__iter__') else episodic_return_gain
                    print(f"Episode complete: steps={ep_len}, return={ep_return:.2f}, gain={ep_gain:.2f} (step {global_step})")

                    first_step_reward = None
                    # Add this episodes return to the list...
                    train_episodic_return_list.append(episodic_return_gain)

                    # ...and if the list is now too long, pop the first element.
                    if len(train_episodic_return_list) > 100:
                        train_episodic_return_list.pop(0)

                    # while len(episode_state) < args.tbptt_seq_len:
                    #      # If the episode is shorter than the TBPTT sequence length, pad it with the last transition.
                    #     episode_state.append(obs)
                    #     episode_action.append(actions)
                    #     episode_last_action.append(prior_actions)
                    #     episode_reward.append(rewards)
                    #     episode_last_reward.append(prior_rewards)
                    #     episode_next_state.append(real_next_obs)
                    #     episode_done.append(terminations)

                    # rb.push(initial_actor_hidden,
                    #         initial_qf1_hidden,
                    #         initial_qf2_hidden,
                    #         episode_state,
                    #         episode_action,
                    #         episode_last_action,
                    #         episode_reward,
                    #         episode_last_reward,
                    #         episode_next_state,
                    #         episode_done)

                    episode_state = list()
                    episode_action = list()
                    episode_last_action = list()
                    episode_reward = list()
                    episode_last_reward = list()
                    episode_next_state = list()
                    episode_done = list()

                    # Set the initial hidden states to the current value.
                    # initial_actor_hidden = (new_actor_hidden[0].detach().clone(),
                    #                         new_actor_hidden[1].detach().clone())
                    # initial_qf1_hidden = (qf1_hidden[0].detach().clone(),
                    #                     qf1_hidden[1].detach().clone())
                    # initial_qf2_hidden = (qf2_hidden[0].detach().clone(),
                    #                     qf2_hidden[1].detach().clone())

                    actor_hidden = actor.get_zero_hidden()
                    qf1_hidden = qf1.get_zero_hidden()
                    qf2_hidden = qf2.get_zero_hidden()
                    initial_actor_hidden = actor_hidden
                    initial_qf1_hidden = qf1_hidden
                    initial_qf2_hidden = qf2_hidden

                    noise_generator.reset()
                    if len(rb) >= args.learning_starts:
                        noise_generator.decay()
                    
                    # Warmup phase: take 16 on-policy steps to initialize hidden states
                    # This helps the LSTM get context before we start storing transitions
                    warmup_steps = 16
                    warmup_obs = obs.copy() if isinstance(obs, np.ndarray) else obs.cpu().numpy().copy()
                    warmup_prior_actions = prior_actions.copy() if isinstance(prior_actions, np.ndarray) else prior_actions
                    warmup_prior_rewards = prior_rewards.copy() if isinstance(prior_rewards, np.ndarray) else prior_rewards
                    
                    with torch.no_grad():
                        for _ in range(warmup_steps):
                            # Convert to tensors
                            warmup_obs_tensor = torch.from_numpy(warmup_obs).to(device, dtype=torch.float32)
                            warmup_prior_actions_tensor = torch.from_numpy(warmup_prior_actions).to(device, dtype=torch.float32)
                            warmup_prior_rewards_tensor = torch.from_numpy(warmup_prior_rewards).to(device, dtype=torch.float32)
                            
                            # Get actions and update hidden states
                            warmup_actions, actor_hidden = actor(
                                warmup_obs_tensor,
                                warmup_prior_actions_tensor,
                                warmup_prior_rewards_tensor,
                                actor_hidden
                            )
                            
                            warmup_actions_tensor = warmup_actions if isinstance(warmup_actions, torch.Tensor) else torch.from_numpy(warmup_actions).to(device, dtype=torch.float32)
                            
                            _, qf1_hidden = qf1(
                                warmup_obs_tensor,
                                warmup_actions_tensor,
                                warmup_prior_actions_tensor,
                                warmup_prior_rewards_tensor,
                                qf1_hidden
                            )
                            _, qf2_hidden = qf2(
                                warmup_obs_tensor,
                                warmup_actions_tensor,
                                warmup_prior_actions_tensor,
                                warmup_prior_rewards_tensor,
                                qf2_hidden
                            )
                            
                            # Step the environment (but don't store transitions)
                            warmup_actions_np = warmup_actions.cpu().numpy() if isinstance(warmup_actions, torch.Tensor) else warmup_actions
                            warmup_next_obs, warmup_rewards, warmup_term, warmup_trunc, _ = envs.step(warmup_actions_np)
                            
                            # Normalize observations
                            if warmup_next_obs.dtype == np.uint8:
                                warmup_next_obs = (warmup_next_obs / 255.0).astype(np.float32)
                            elif warmup_next_obs.dtype == np.uint16:
                                warmup_next_obs = (warmup_next_obs / 65535.0).astype(np.float32)
                            
                            # Update for next warmup step
                            warmup_prior_actions = warmup_actions_np
                            warmup_prior_rewards = warmup_rewards * args.reward_scale
                            warmup_obs = warmup_next_obs
                            
                            # If episode ends during warmup, reset env AND hidden states
                            if any(warmup_term) or any(warmup_trunc):
                                warmup_obs, _ = envs.reset()
                                if warmup_obs.dtype == np.uint8:
                                    warmup_obs = (warmup_obs / 255.0).astype(np.float32)
                                elif warmup_obs.dtype == np.uint16:
                                    warmup_obs = (warmup_obs / 65535.0).astype(np.float32)
                                # Reset hidden states since episode ended
                                actor_hidden = actor.get_zero_hidden()
                                qf1_hidden = qf1.get_zero_hidden()
                                qf2_hidden = qf2.get_zero_hidden()
                                # Reset prior actions/rewards for new episode
                                warmup_prior_actions = np.zeros_like(warmup_prior_actions)
                                warmup_prior_rewards = np.zeros_like(warmup_prior_rewards)
                    
                    # After warmup, update initial hidden states for the next sequence
                    initial_actor_hidden = (actor_hidden[0].detach().clone(),
                                            actor_hidden[1].detach().clone())
                    initial_qf1_hidden = (qf1_hidden[0].detach().clone(),
                                          qf1_hidden[1].detach().clone())
                    initial_qf2_hidden = (qf2_hidden[0].detach().clone(),
                                          qf2_hidden[1].detach().clone())
                    
                    # Update obs and prior values to continue from warmup state
                    obs = warmup_obs
                    prior_actions = warmup_prior_actions
                    prior_rewards = warmup_prior_rewards
                        
                    break


            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            # initial_actor_hidden = (new_actor_hidden[0].detach().clone(),
            #                         new_actor_hidden[1].detach().clone())
            # initial_qf1_hidden = (qf1_hidden[0].detach().clone(),
            #                       qf1_hidden[1].detach().clone())
            # initial_qf2_hidden = (qf2_hidden[0].detach().clone(),
            #                       qf2_hidden[1].detach().clone())

        # ALGO LOGIC: training.
        # If it is time to train, then train.
        if len(rb) >= args.learning_starts and len(rb) >= args.batch_size:
            
            # Record when learning actually started (in terms of global_step)
            if learning_started_at_step is None:
                learning_started_at_step = global_step
                print(f"\n{'='*60}")
                print(f"✓ TRAINING STARTED at step {global_step}")
                print(f"  Buffer size: {len(rb)} sequences")
                print(f"  Batch size: {args.batch_size}")
                print(f"{'='*60}\n")
            
            # Periodic progress update
            if iteration % 1000 == 0:
                progress_pct = 100 * global_step / args.total_timesteps
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                eta_seconds = (args.total_timesteps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                print(f"Progress: {progress_pct:.1f}% | Step {global_step:,}/{args.total_timesteps:,} | {steps_per_sec:.1f} SPS | ETA: {eta_hours:.1f}h")

            # ============ Batch Loading (with optional prefetching) ============
            if prefetch_stream is not None:
                # Use prefetched batch if available, otherwise sync-load first batch
                if prefetched_batch is not None:
                    timer.start("prefetch_sync")
                    prefetch_stream.synchronize()  # Wait for prefetch to complete
                    timer.stop("prefetch_sync")
                    
                    (actor_hidden_batch, qf1_hidden_batch, qf2_hidden_batch,
                     observations_batch, actions_batch_batch, prior_actions_batch,
                     rewards_batch, prior_rewards_batch, next_observations_batch, dones_batch) = prefetched_batch
                else:
                    # First iteration: no prefetched batch yet, do sync load
                    timer.start("rb_sample")
                    rb_sample = rb.sample(args.batch_size, device=device)
                    timer.stop("rb_sample")
                    timer.start("tensor_to_device")
                    (actor_hidden_batch, qf1_hidden_batch, qf2_hidden_batch,
                     observations_batch, actions_batch_batch, prior_actions_batch,
                     rewards_batch, prior_rewards_batch, next_observations_batch, dones_batch) = prepare_batch_tensors(rb_sample, device, non_blocking=False)
                    timer.stop("tensor_to_device")
                
                # Prefetch next batch asynchronously in the background stream
                timer.start("prefetch_launch")
                with torch.cuda.stream(prefetch_stream):
                    next_rb_sample = rb.sample(args.batch_size, device=device)
                    prefetched_batch = prepare_batch_tensors(next_rb_sample, device, non_blocking=True)
                timer.stop("prefetch_launch")
            else:
                # No prefetching: original sync loading path
                timer.start("rb_sample")
                (actor_hidden_batch,
                 qf1_hidden_batch,
                 qf2_hidden_batch,
                 observations_batch,
                 actions_batch_batch,
                 prior_actions_batch,
                 rewards_batch,
                 prior_rewards_batch,
                 next_observations_batch,
                 dones_batch) = rb.sample(args.batch_size,
                                          device=device,)
                timer.stop("rb_sample")
                
                # Convert to tensors efficiently - use torch.from_numpy if already numpy, avoid double wrapping
                timer.start("tensor_to_device")
                observations_batch =      torch.from_numpy(np.asarray(observations_batch)).to(device, dtype=torch.float32)
                actions_batch_batch =     torch.from_numpy(np.asarray(actions_batch_batch)).to(device, dtype=torch.float32)
                prior_actions_batch =     torch.from_numpy(np.asarray(prior_actions_batch)).to(device, dtype=torch.float32)
                rewards_batch =           torch.from_numpy(np.asarray(rewards_batch)).to(device, dtype=torch.float32)
                prior_rewards_batch =     torch.from_numpy(np.asarray(prior_rewards_batch)).to(device, dtype=torch.float32)
                next_observations_batch = torch.from_numpy(np.asarray(next_observations_batch)).to(device, dtype=torch.float32)
                dones_batch =             torch.from_numpy(np.asarray(dones_batch)).to(device, dtype=torch.float32)
                timer.stop("tensor_to_device")

            # Debug: print hidden state shape on first training step
            if learning_started_at_step == global_step:
                print(f"\n{'='*60}")
                print(f"DEBUG: Hidden state shapes at first training step")
                print(f"{'='*60}")
                print(f"actor_hidden_batch structure:")
                print(f"  type: {type(actor_hidden_batch)}")
                print(f"  len: {len(actor_hidden_batch)}")
                print(f"  actor_hidden_batch[0] type: {type(actor_hidden_batch[0])}")
                print(f"  actor_hidden_batch[1] type: {type(actor_hidden_batch[1])}")
                if isinstance(actor_hidden_batch[0], list):
                    print(f"  actor_hidden_batch[0] len: {len(actor_hidden_batch[0])}")
                    print(f"  actor_hidden_batch[1] len: {len(actor_hidden_batch[1])}")
                    print(f"  Individual h tensor shapes:")
                    for i, h in enumerate(actor_hidden_batch[0][:3]):  # First 3
                        print(f"    h[{i}].shape: {h.shape}, dim: {h.dim()}")
                    print(f"  Individual c tensor shapes:")
                    for i, c in enumerate(actor_hidden_batch[1][:3]):  # First 3
                        print(f"    c[{i}].shape: {c.shape}, dim: {c.dim()}")
                else:
                    print(f"  actor_hidden_batch[0].shape: {actor_hidden_batch[0].shape}")
                    print(f"  actor_hidden_batch[1].shape: {actor_hidden_batch[1].shape}")
                print(f"\nModel config:")
                print(f"  actor.lstm_hidden_dim: {actor.lstm_hidden_dim}")
                print(f"  actor.lstm_num_layers: {actor.lstm_num_layers}")
                print(f"  batch_size (args): {args.batch_size}")
                print(f"\nObservation batch shape: {next_observations_batch.shape}")
                print(f"{'='*60}\n")

            # Enable debug output on first training step
            debug_first_step = (learning_started_at_step == global_step)

            # Stop the gradients from flowing through the actor and target actor.
            with torch.no_grad():

                timer.start("target_actor_forward")
                # print(actor_hidden_batch)
                # print(len(actor_hidden_batch), len(actor_hidden_batch[0]), len(actor_hidden_batch[1]))
                # TODO: investigate the time-ordering of inputs herte. Should actions, not prior actions. Might not matter
                # Introduce target smoothing by adding noise to the next state actions.
                next_state_actions_batch, _ = target_actor(
                        next_observations_batch,
                        actions_batch_batch,
                        rewards_batch,
                        actor_hidden_batch,
                    )

                # TD3 target policy smoothing: add noise to target actions
                noise = (torch.randn_like(next_state_actions_batch) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)

                noisy_next_action = (
                        next_state_actions_batch + noise
                    ).clamp(
                            float(envs.single_action_space.low[0]),
                            float(envs.single_action_space.high[0])
                        )
                
                if args.target_smoothing:
                    next_state_actions_batch = noisy_next_action
                timer.stop("target_actor_forward")

                timer.start("target_critic_forward")
                qf1_next_target_batch, _ = qf1_target(
                        next_observations_batch,
                        next_state_actions_batch,
                        actions_batch_batch,
                        rewards_batch,
                        qf1_hidden_batch
                    )
                
                qf2_next_target_batch, _ = qf2_target(
                        next_observations_batch,
                        next_state_actions_batch,
                        actions_batch_batch,
                        rewards_batch,
                        qf2_hidden_batch
                    )
                timer.stop("target_critic_forward")
                
                timer.start("td_target_compute")
                qf1_next_target_batch = torch.min(
                        qf1_next_target_batch,
                        qf2_next_target_batch
                    )
                
                # next_q_value_batch = rewards_batch.flatten() + (1 - dones_batch.flatten()) * args.gamma * (qf1_next_target_batch).view(-1)
                # print("rewards_batch shape:", rewards_batch.shape)
                # print("dones_batch shape:", dones_batch.shape)
                # print("qf1_next_target_batch shape:", qf1_next_target_batch.shape)
                # Take the last reward and done from the batch, since we are using TBPTT.
                # TODO: If this doens't work, try using the full sequence - it was definitely broken before becuase of the silent broadcasting.
                # next_q_value_batch = rewards_batch[:, -1, :].flatten() + (1 - dones_batch[:, -1, :].flatten()) * args.gamma * (qf1_next_target_batch[:, -1, :].flatten())
                next_q_value_batch = rewards_batch + (1 - dones_batch) * args.gamma * (qf1_next_target_batch)
                # next_q_value_batch = rewards_batch


                # rewards_batch shape: torch.Size([8, 2, 1])
                # dones_batch shape: torch.Size([8, 2, 1])
                # qf1_next_target_batch shape: torch.Size([8, 2, 1])
                # next_q_value_batch shape: torch.Size([8, 2, 16])
                timer.stop("td_target_compute")

            # TODO: Should I remove the view here?
            # print("next_q_value_batch shape:", next_q_value_batch.shape)

            timer.start("critic_forward")
            qf1_a_values_batch, _ = qf1(
                observations_batch,
                actions_batch_batch,
                prior_actions_batch,
                prior_rewards_batch,
                qf1_hidden_batch
            )
            qf2_a_values_batch, _ = qf2(
                observations_batch,
                actions_batch_batch,
                prior_actions_batch,
                prior_rewards_batch,
                qf2_hidden_batch
            )
            timer.stop("critic_forward")

            clip_gradients = args.clip_gradients
            
            # Compute and apply the critic losses.
            # (input, target)
            # Both critics should use the same loss computation for consistency
            timer.start("critic_loss_compute")
            qf1_loss = F.mse_loss(qf1_a_values_batch.view(-1), next_q_value_batch.view(-1))
            qf2_loss = F.mse_loss(qf2_a_values_batch.view(-1), next_q_value_batch.view(-1))
            timer.stop("critic_loss_compute")
            
            timer.start("critic_backward")
            qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=False)
            qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=False)
            timer.stop("critic_backward")
            
            timer.start("critic_grad_clip")
            if clip_gradients:
                if iteration % args.writer_interval == 0:
                    qf1_grad = get_grad_norm(qf1)
                    qf2_grad = get_grad_norm(qf2)
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=args.max_grad_norm)
                if iteration % args.writer_interval == 0:
                    qf1_grad_clipped = get_grad_norm(qf1)
                    qf2_grad_clipped = get_grad_norm(qf2)
                    writer.add_scalar("grads/qf1_grad", qf1_grad, global_step)
                    writer.add_scalar("grads/qf1_grad_clipped", qf1_grad_clipped, global_step)
                    writer.add_scalar("grads/qf2_grad", qf2_grad, global_step)
                    writer.add_scalar("grads/qf2_grad_clipped", qf2_grad_clipped, global_step)
            timer.stop("critic_grad_clip")
            
            timer.start("critic_optimizer_step")
            qf1_optimizer.step()
            qf2_optimizer.step()
            timer.stop("critic_optimizer_step")

            # If it a policy update step and we're past the actor training delay, update the actor.
            # learning_started_at_step is guaranteed to be set here since we're inside the training block.
            if (global_step > learning_started_at_step + args.actor_training_delay) and (global_step % args.policy_frequency == 0):
                
                timer.start("actor_forward")
                loss_actions, _ = actor(
                        observations_batch,
                        prior_actions_batch,
                        prior_rewards_batch,
                        actor_hidden_batch
                    )
                timer.stop("actor_forward")
                
                timer.start("actor_loss_compute")
                loss_qvalues, _ = qf1(
                    observations_batch,
                    loss_actions,
                    prior_actions_batch,
                    prior_rewards_batch,
                    qf1_hidden_batch
                )
                actor_loss = -loss_qvalues.mean()
                timer.stop("actor_loss_compute")
                
                timer.start("actor_backward")
                actor_optimizer.zero_grad()

                for p in qf1.parameters():
                    p.requires_grad = False
                actor_loss.backward(retain_graph=False)

                for p in qf1.parameters():
                    p.requires_grad = True
                timer.stop("actor_backward")

                timer.start("actor_grad_clip")
                if clip_gradients:
                    if iteration % args.writer_interval == 0:
                        actor_grad = get_grad_norm(actor)

                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)
                
                    if iteration % args.writer_interval == 0:
                        actor_grad_clipped = get_grad_norm(actor)
                        writer.add_scalar("grads/actor_grad", actor_grad, global_step)
                        writer.add_scalar("grads/actor_grad_clipped", actor_grad_clipped, global_step)
                timer.stop("actor_grad_clip")

                timer.start("actor_optimizer_step")
                actor_optimizer.step()
                timer.stop("actor_optimizer_step")

                if iteration % args.writer_interval == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    
                # update the target actor network
                timer.start("target_network_update")
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                timer.stop("target_network_update")

            # Soft update the target critic networks.
            timer.start("target_network_update")
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            timer.stop("target_network_update")

            if iteration % args.writer_interval == 0:
                writer.add_scalar("losses/qf1_a_values", qf1_a_values_batch.mean().item(), global_step)
                writer.add_scalar("losses/qf2_a_values", qf2_a_values_batch.mean().item(), global_step)

                # Write the l2 distance between the two qf1 and qf2 outputs.
                writer.add_scalar("losses/qf1_qf2_l2", torch.linalg.vector_norm(qf1_a_values_batch - qf2_a_values_batch).item(), global_step)

                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)

        # End of step - trigger timer summary and TensorBoard logging
        timer.step(global_step=global_step)

        if iteration % args.writer_interval == 0:
            writer.add_scalar("charts/step_length", (time.time() - step_time), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/step_SPS", (args.num_envs / (time.time() - step_time)), global_step)

        global_step += args.num_envs

    envs.close()
    writer.close()
    print("Done.")


