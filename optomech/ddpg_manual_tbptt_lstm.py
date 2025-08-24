# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
import shutil
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



from rollout import rollout_optomech_policy
from replay_buffers import *


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    save_dir: str = "./runs/"
    """the directory to save the experiment results"""
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
    learning_starts: int = 256
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    decay_rate: float = 0.00001
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
    use_q_bias: bool = False
    """If toggled, compute q bias in the critic model."""
    normalize_returns: bool = False
    """If toggled, normalize the returns in the critic model."""
    clip_gradients: bool = False
    """If toggled, clip gradients during training."""

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
    actor_training_delay: int = 10_000
    """How many steps to wait before populating the RB"""
    experience_sampling_delay: int = 10_000
    """Whether or not to use target smoothing"""
    target_smoothing: bool = False
    """How long the sequence length is for the LSTM"""
    tbptt_seq_len: int = 16

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




class ImpalaActor(nn.Module):

    def __init__(self,
                 envs,
                 device,
                 lstm_hidden_dim=128,
                 lstm_num_layers=1,
                 channel_scale=16,
                 fc_scale=8,
                 action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        self.channels_last = obs_shape[-1] == 1
        if self.channels_last:
            input_channels = obs_shape[-1]
            input_shape = obs_shape[:-1]
        else:
            input_channels = obs_shape[0]
            input_shape = obs_shape[1:]

        self.debug = True

        if self.debug:
            
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(input_channels, channel_scale, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(channel_scale, channel_scale * 2, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.inference_mode():

                # Handle channels-last environments.
                x = torch.zeros(1, *obs_shape)
                if self.channels_last:
                    x = x.permute(0, 3, 1, 2)
                visual_output_shape = self.visual_encoder(x).shape

            # Debug LSTM
            self.debugnet = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(visual_output_shape[1:])), fc_scale),
                nn.ReLU(),
                nn.Linear(fc_scale, fc_scale // 2),
                nn.ReLU(),
            )

            self.debuglstm = nn.LSTM(
                input_size=fc_scale // 2,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                batch_first=True
            )   

            self.action_head = nn.Sequential(
                nn.Linear(fc_scale // 2 + self.lstm_hidden_dim, fc_scale // 2),
                nn.ReLU(),
                nn.Linear(fc_scale // 2, int(np.prod(envs.single_action_space.shape))),
                nn.Tanh(),
            )


        else:

            # Define the visual encoder, so that we know the output shape.
            self.visual_encoder = nn.Sequential(
                conv_init(
                    nn.Conv2d(input_channels, 
                            channel_scale,
                            kernel_size=8,
                            stride=4)),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(channel_scale,
                            channel_scale * 2,
                            kernel_size=4,
                            stride=2)),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(channel_scale * 2,
                            channel_scale * 4,
                            kernel_size=2,
                            stride=2)),
                nn.ReLU(),
            )

            # self.visual_encoder = nn.Sequential(
            #     nn.Flatten(),
            # )
            # self.visual_encoder = nn.Sequential(
            #     conv_init(
            #         nn.Conv2d(input_channels, 
            #                   channel_scale,
            #                   kernel_size=8,
            #                   stride=4)),
            #     nn.ReLU(),
            #     conv_init(
            #         nn.Conv2d(channel_scale,
            #                   channel_scale * 2,
            #                   kernel_size=4,
            #                   stride=2)),
            #     nn.ReLU(),
            # )

            # Get the output shape of the visual encoder
            with torch.inference_mode():

                # Handle channels-last environments.
                x = torch.zeros(1, *obs_shape)
                if self.channels_last:
                    x = x.permute(0, 3, 1, 2)
                visual_output_shape = self.visual_encoder(x).shape
        
            mlp_output_size = fc_scale
            self.mlp = nn.Sequential(
                nn.Flatten(),
                layer_init(
                    nn.Linear(
                        int(np.prod(visual_output_shape[1:])),
                        mlp_output_size),
                    std=np.sqrt(2)
                ),
                # nn.LayerNorm(mlp_output_size),
                nn.ReLU(),
            )


            self.lstm = init_lstm_weights(nn.LSTM(
                input_size=mlp_output_size + vector_action_size + 1,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                batch_first=True
            ))

            # Get the output shape of the LSTM
            with torch.inference_mode():            
            
                x = torch.zeros(1, mlp_output_size + vector_action_size + 1)

                pre_head_output_shape = self.lstm(x)[0].shape

            # Build the action head following the convolutional LSTM
            self.action_head = nn.Sequential(
                layer_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        fc_scale,
                            ),
                    std=np.sqrt(2)
                ),
                # nn.LayerNorm(fc_scale),
                nn.ReLU(),
                uniform_init(
                    nn.Linear(
                        fc_scale,
                        int(np.prod(envs.single_action_space.shape))
                            ),
                    lower_bound=-1e-4,
                    upper_bound=1e-4
                ),
                nn.Tanh()
            )
                                    
        # action rescaling
        self.register_buffer(
            # "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
            "action_scale", torch.tensor(action_scale, dtype=torch.float32)
        )
        self.register_buffer(
            # "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
            "action_bias", torch.tensor(0.0, dtype=torch.float32)
        )


    def get_zero_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        h = torch.zeros(1, self.lstm_hidden_dim,).detach().to(self.device)
        c = torch.zeros(1, self.lstm_hidden_dim,).detach().to(self.device)
        return (h, c)


    def forward(self,
                o,
                a_prior,
                r_prior,
                hidden: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_input = len(o.shape) == 5

        # Handle the messy differences in our setup
        if batch_input:
            a_prior = a_prior.squeeze(1)  
        else:
            r_prior = r_prior.unsqueeze(-1)

        # Handle channels-last environments.
        if self.channels_last:
            # print(f"[o] shape before permute: {o.shape}")
            if len(o.shape) == 4:
                o = o.permute(0, 3, 1, 2)
            elif len(o.shape) == 5:
                o = o.permute(0, 1, 4, 2, 3)
                o = o.squeeze(1)  # Remove the sequence dimension
            elif len(o.shape) == 6:
                o = o.permute(0, 1, 5, 2, 3, 4)  
            # print(f"[o] shape after permute: {o.shape}")

        x = self.visual_encoder(o)
        x = self.debugnet(x) 
        if batch_input:
            x = x.unsqueeze(1)
        x_lstm, new_hidden = self.debuglstm(x, hidden)
        x = torch.cat([x_lstm, x], dim=-1)
        a = self.action_head(x)
        # if batch_input:
        #     a = a.unsqueeze(1)
        # new_hidden = self.get_zero_hidden()

        # else:

        #     x_o = self.visual_encoder(o)
        #     x = self.mlp(x_o)


        #     # print(f"[x] shape before concat: {x.shape}")
        #     # print(f"[a_prior] shape: {a_prior.shape}")
        #     # print(f"[r_prior] shape: {r_prior.shape}")
        #     x = torch.cat([x, a_prior, r_prior], dim=-1)
            
        #     if batch_input:
        #         x = x.unsqueeze(1)  # Add sequence dimension

        #     h0 = hidden[0]
        #     c0 = hidden[1]
        #     x, new_hidden = self.lstm(x, (h0, c0))

        #     a = self.action_head(x)

        a = (a * self.action_scale + self.action_bias)

        return a, new_hidden

class ImpalaCritic(nn.Module):
    
    def __init__(self,
                 envs,
                 device,
                 lstm_hidden_dim=128,
                 lstm_num_layers=1,
                 channel_scale=16,
                 fc_scale=8,
                 q_bias=0.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.use_lstm = True
        self.device = device

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.q_bias = q_bias

        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        self.channels_last = obs_shape[-1] == 1
        if self.channels_last:
            input_channels = obs_shape[-1]
            input_shape = obs_shape[:-1]
        else:
            input_channels = obs_shape[0]
            input_shape = obs_shape[1:]

        self.debug = True

        if self.debug:

            self.visual_encoder = nn.Sequential(
                nn.Conv2d(input_channels, channel_scale, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(channel_scale, channel_scale * 2, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.inference_mode():

                # Handle channels-last environments.
                x = torch.zeros(1, *obs_shape)
                if self.channels_last:
                    x = x.permute(0, 3, 1, 2)
                visual_output_shape = self.visual_encoder(x).shape

            # Debug LSTM
            self.debugnet = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(visual_output_shape[1:])), fc_scale),
                nn.ReLU(),
                nn.Linear(fc_scale, fc_scale // 2),
                nn.ReLU(),
            )

            self.debuglstm = nn.LSTM(
                input_size=fc_scale // 2,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                batch_first=True
            )
            
            self.q_head = nn.Sequential(

                nn.Linear((fc_scale // 2) + self.lstm_hidden_dim + 1, fc_scale // 2),
                nn.ReLU(),
                layer_init(
                    nn.Linear(fc_scale // 2,
                            1
                            ),
                    std=1.0,
                    bias_const=q_bias
                ),
            )


        else:


            # Define the visual encoder, so that we know the output shape.
            self.visual_encoder = nn.Sequential(
                conv_init(
                    nn.Conv2d(input_channels, 
                            channel_scale,
                            kernel_size=8,
                            stride=4),
                    ),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(channel_scale,
                            channel_scale * 2,
                            kernel_size=4,
                            stride=2)
                    ),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(channel_scale * 2,
                            channel_scale * 4,
                            kernel_size=2,
                            stride=2)),
                nn.ReLU(),
            )

            # self.visual_encoder = nn.Sequential(
            #     nn.Flatten(),
            # )


            # self.visual_encoder = nn.Sequential(
            #     conv_init(
            #         nn.Conv2d(input_channels, 
            #                   channel_scale,
            #                   kernel_size=8,
            #                   stride=4)),
            #     nn.ReLU(),
            #     conv_init(
            #         nn.Conv2d(channel_scale,
            #                   channel_scale * 2,
            #                   kernel_size=4,
            #                   stride=2)),
            #     nn.ReLU(),
            # )

            # Get the output shape of the visual encoder
            with torch.inference_mode():

                # Handle channels-last environments.
                x = torch.zeros(1, *obs_shape)
                if self.channels_last:
                    x = x.permute(0, 3, 1, 2)
                visual_output_shape = self.visual_encoder(x).shape

            mlp_output_size = fc_scale
            self.mlp = nn.Sequential(
                nn.Flatten(),
                layer_init(
                    nn.Linear(
                        int(np.prod(visual_output_shape[1:])),
                        mlp_output_size),
                    std=np.sqrt(2)
                ),
                # nn.LayerNorm(mlp_output_size),
                nn.ReLU(),
            )

            self.lstm = init_lstm_weights(nn.LSTM(
                input_size=mlp_output_size + vector_action_size + vector_action_size + 1,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                batch_first=True
            ))

            # Get the output shape of the LSTM
            with torch.inference_mode():
                x = torch.zeros(
                    1,
                    mlp_output_size + vector_action_size + vector_action_size + 1
                )
                pre_head_output_shape = self.lstm(x)[0].shape

            # Build the q head following the convolutional LSTM
            self.q_head = nn.Sequential(
                layer_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        fc_scale
                        ),
                    std=1.0
                ),
                # nn.LayerNorm(fc_scale),
                nn.ReLU(),
                layer_init(
                    nn.Linear(fc_scale,
                            1
                            ),
                    std=1.0,
                    bias_const=q_bias
                ),
            )

    def get_zero_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        h = torch.zeros(1, self.lstm_hidden_dim,).detach().to(self.device)
        c = torch.zeros(1, self.lstm_hidden_dim,).detach().to(self.device)
        return (h, c)

    def forward(self, o, a, a_prior, r_prior, hidden=None):

        # print(f"forward-input [o] shape: {o.shape}")
        # print(f"forward-input [a] shape: {a.shape}")
        # print(f"forward-input [a_prior] shape: {a_prior.shape}")
        # print(f"forward-input [r_prior] shape: {r_prior.shape}")

        batch_input = len(o.shape) == 5

        # Handle the messy differences in our setup
        if batch_input:
            a_prior = a_prior.squeeze(1)
            a = a.squeeze(1)
            # die
        else:
            r_prior = r_prior.unsqueeze(-1)

        # Handle channels-last environments.
        if self.channels_last:
            # print(f"[o] shape before permute: {o.shape}")
            if len(o.shape) == 4:
                o = o.permute(0, 3, 1, 2)
            elif len(o.shape) == 5:
                o = o.permute(0, 1, 4, 2, 3)
                o = o.squeeze(1)  # Remove the sequence dimension
            elif len(o.shape) == 6:
                o = o.permute(0, 1, 5, 2, 3, 4)  
            # print(f"[o] shape after permute: {o.shape}")


        # Debug only
        # x = self.debugnet(o)  # Add a dimension for the q value
        # x = torch.cat([x, a], dim=-1)
        # q = self.q_head(x).unsqueeze(-1)
        # new_hidden = self.get_zero_hidden()

        # Debug LSTM
        x = self.visual_encoder(o)
        x = self.debugnet(x)  # Add a dimension for the q value
        if batch_input:
            x = x.unsqueeze(1)
            a = a.unsqueeze(1)
        x_lstm, new_hidden = self.debuglstm(x, hidden)
        # print(f"[x] shape before concat: {x.shape}")
        # print(f"[a] shape: {a.shape}")
        # print(f"[x_lstm] shape before concat: {x_lstm.shape}")
        x = torch.cat([x, x_lstm, a], dim=-1)
        q = self.q_head(x)
        

        # x_o = self.visual_encoder(o)
        # x = self.mlp(x_o)


        # print(f"[x] shape before concat: {x.shape}")
        # print(f"[a] shape: {a.shape}")
        # print(f"[a_prior] shape: {a_prior.shape}")
        # print(f"[r_prior] shape: {r_prior.shape}")
        # x = torch.cat([x, a, a_prior, r_prior], dim=-1)

        
        # if batch_input:
        #     x = x.unsqueeze(1)  # Add sequence dimension

        # h0 = hidden[0]
        # c0 = hidden[1]

        # x, new_hidden = self.lstm(x, (h0, c0))

        # # Apply action prediciton head and activation function
        # q = self.q_head(x)

        # # print(f"[q] shape: {q.shape}")
        return q, new_hidden
    


def uniform_init(layer, lower_bound=-1e-4, upper_bound=1e-4):

    # init this layer to have weights and biases drawn uniformly from bounds.
    nn.init.uniform_(layer.weight, a=lower_bound, b=upper_bound)
    nn.init.uniform_(layer.bias, a=lower_bound, b=upper_bound)
    return layer

def conv_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    # return layer

def init_lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
            # Set forget gate bias to 1.0
            n = param.size(0)
            param.data[n//4:n//2] = 1.0  # forget gate bias
    return lstm



def weight_regularization(model: torch.nn.Module, l1_scale=0.0, l2_scale=0.0):
    l1_loss = 0.0
    l2_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            if l1_scale > 0:
                l1_loss += torch.sum(torch.abs(param))
            if l2_scale > 0:
                l2_loss += torch.sum(param ** 2)
    return l1_scale * l1_loss + l2_scale * l2_loss


def log_gradients_in_model(model, logger, step, prefix="layer_grads"):
    """
    Log the gradients of all parameters in the actor model to TensorBoard.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            logger.add_scalar(f"{prefix}/{name}_grad_norm", grad_norm, step)
            logger.add_histogram(f"{prefix}/{name}_hist", param.grad.data.cpu().numpy(), step)
        else:
            logger.add_scalar(f"{prefix}/{name}_grad_norm", 0.0, step)

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
        total_norm = total_norm + param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True) 
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    

    args = tyro.cli(Args)
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
    writer = SummaryWriter(f"{args.save_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Write the arguements of this run to a file.
    args_store_path = f"{args.save_dir}/{run_name}/args.json"
    with open(args_store_path, "w") as f:
        json.dump(vars(args), f)

    # Check if MPS is available
    if torch.cuda.is_available():
        print("Running with CUDA")
        device = torch.device("cuda")
        torch.cuda.set_device(int(args.gpu_list))
    elif torch.backends.mps.is_available():
        print("Running with MSP")
        device = torch.device("mps")
    else:
        print("Running with CPU")
        device = torch.device("cpu")

    # env setup
    if args.subproc_env:
        print("Initializing SubprocVectorEnv")
        envs = gym.vector.SubprocVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
    if args.async_env:
        print("Initializing AsyncVectorEnv")
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
    else:
        print("Initializing SyncVectorEnv")
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
        )
    
    if args.actor_type == "impala":

        actor = ImpalaActor(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            action_scale=args.action_scale,).to(device)
        target_actor = ImpalaActor(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            action_scale=args.action_scale).to(device)
        
        scripted_actor = torch.jit.script(actor)



    # Potential-based reward shaping https://arxiv.org/pdf/2502.01307
    if args.use_q_bias:
        reward_sample_episodes = 100
        episode_rewards = []
        # Sample some random rewards to compute the q bias.
        # This is a hacky way to get the expected reward.
        # We sample 10 episodes of random actions and average the rewards.
        print("Sampling random rewards to compute q bias...")
        for _ in range(reward_sample_episodes):
            obs, _ = envs.reset(seed=args.seed)
            episode_reward_sum = 0.0
            for t in range(args.max_episode_steps):
                first_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                _, rewards, _, _, _ = envs.step(first_actions)
                episode_reward_sum += np.mean(rewards)
            episode_rewards.append(episode_reward_sum)
        expected_reward = np.mean(episode_rewards) * args.reward_scale
        print(f"Expected reward: {expected_reward}")
        # expected_reward = -13.5 * args.reward_scale
        q_bias = expected_reward * ((1 - (args.gamma ** args.max_episode_steps)) / (1 - args.gamma))
        # args.normalize_returns = False
        if args.normalize_returns:
            # If we are normalizing returns, we need to scale the q bias.
            args.reward_scale = 1.0 / q_bias
            q_bias = 0.0
    else:
        q_bias = 0.0



    
    if args.critic_type  == "impala":

        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf1 = ImpalaCritic(
            envs,
            device,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            q_bias=q_bias).to(device
        )
        qf1_target = ImpalaCritic(
            envs,
            device,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            q_bias=q_bias).to(device
        )
        
        _ = torch.randn(1000)
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        _ = torch.randn(1000)
        qf2 = ImpalaCritic(
            envs,
            device,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            q_bias=q_bias).to(device)
        qf2_target = ImpalaCritic(
            envs,
            device,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            lstm_hidden_dim=args.lstm_hidden_dim,
            q_bias=q_bias).to(device)
        
        for p in qf2.parameters():
            p.data += 1e-3 * torch.randn_like(p)
        for p in qf2_target.parameters():
            p.data += 1e-3 * torch.randn_like(p)

    else:
        raise ValueError("Invalid critic type specified.")

    print("Actor and Critic parameter counts:")
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print(sum(p.numel() for p in qf1.parameters() if p.requires_grad))
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

    rb = ReplayBufferWithHiddenStatesBPTT(args.buffer_size)

    if args.replay_buffer_load_path:
        print(f"Loading replay buffer from {args.replay_buffer_load_path}")
        # Search for replay_buffer_load_path in the current directory and its subdirectories.
        if not os.path.exists(args.replay_buffer_load_path):
            raise FileNotFoundError(f"Replay buffer load path {args.replay_buffer_load_path} does not exist.")
        if os.path.isdir(args.replay_buffer_load_path):
            # If the path is a directory, load the replay buffer from the latest file in the directory.
            files = [f for f in os.listdir(args.replay_buffer_load_path) if f.endswith('.pt')]
            if not files:
                raise FileNotFoundError(f"No .pt files found in {args.replay_buffer_load_path}.")

            rb.restore(args.replay_buffer_load_path)
        print(f"Replay buffer loaded with {str(len(rb.buffer))} elements.")

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
    print("Resetting Environments.")
    obs, info = envs.reset(seed=args.seed)
    print("Environments Reset.")

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


    episode_state_ring = list()
    episode_action_ring = list()
    episode_last_action_ring = list()
    episode_reward_ring = list()
    episode_last_reward_ring = list()
    episode_next_state_ring = list()
    episode_done_ring = list()

    # Initialize the global step and episode return lists.
    global_step = 0
    best_train_episode_return = -np.inf
    train_episodic_return_list = list()
    best_eval_episode_return = -np.inf

    # During this phase, we run exactly full pass through the env and models.
    first_actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])
    obs, rewards, _, _, _ = envs.step(first_actions)
    # print the obs, rewards, and first_actions shapes.
    print(f"Obs shape: {obs.shape}, Rewards shape: {rewards.shape}, First actions shape: {first_actions.shape}")

    # If the obs are 8bit images, convert to float32 and rescale.
    if torch.tensor(obs).dtype != torch.float32:
        obs = np.array((obs / 255.0).astype(np.float32))
    rewards = args.reward_scale * rewards

    prior_actions = first_actions
    prior_rewards = rewards
    first_step_reward = rewards
    actions, new_actor_hidden = actor(
            torch.tensor(obs).to(device),
            torch.tensor(prior_actions).to(device),
            torch.tensor(prior_rewards).to(torch.float32).to(device),
            actor.get_zero_hidden()
        )
    actor_hidden = new_actor_hidden
    
    # ...Everybody's a critic! (except the actor, of course)
    # Note to humans: ChatGPT made ^^^this^^^ joke, unprompted. -jrf
    qf1_a_values, qf1_hidden = qf1(
        torch.tensor(obs).to(device),
        actions.to(device),
        torch.tensor(first_actions).to(device),
        torch.tensor(rewards.astype(np.float32)).to(device),
        qf1.get_zero_hidden()
    )
    qf2_a_values, qf2_hidden = qf2(
        torch.tensor(obs).to(device),
        actions.to(device),
        torch.tensor(first_actions).to(device),
        torch.tensor(rewards.astype(np.float32)).to(device),
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


    for iteration in range(args.total_timesteps):

        step_time = time.time()

        if not args.silence:
            print(f"Iteration: {iteration} Global Step: {global_step}")

        # First, check to see if we need to save the model.
        if args.save_model:
            # Check that we've complete at least one post-learning episode.
            if global_step > (args.learning_starts + args.max_episode_steps):

                # If mean reward of recent episodes improved, save the model.
                if np.mean(train_episodic_return_list) > best_train_episode_return:

                    # Update the best mean reward and save the model.
                    best_train_episode_return = np.mean(train_episodic_return_list)

                    # First, delete the prior best model if it exists...
                    best_models_save_path = f"{args.save_dir}/{run_name}/best_models"
                    model_path = f"{best_models_save_path}/best_train_policy.pt"
                    if os.path.exists(model_path):
                        os.remove(model_path)


                    # ...then save the best train model.
                    scripted_actor = torch.jit.script(actor)
                    # Check if the directory exists, if not, create it.
                    Path(best_models_save_path).mkdir(parents=True, exist_ok=True)
                    scripted_actor.save(model_path)
                    print(f"Torchscript model saved to {model_path}.")

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
                    mean_zero_return_advantage = np.mean(zero_policy_returns_array - episodic_returns_array)
                    mean_random_return_advantage = np.mean(random_policy_returns_array - episodic_returns_array)

                    writer.add_scalar("eval/best_policy_mean_returns", mean_eval_episode_return, iteration)
                    writer.add_scalar("eval/best_policy_zero_return_advantage", mean_zero_return_advantage, iteration)
                    writer.add_scalar("eval/best_policy_random_return_advantage", mean_random_return_advantage, iteration)


                    print(f"Mean eval episodic return: {mean_eval_episode_return}")

                # If we didn't just save them model, but it's time to save the model, do so.
                elif iteration % args.model_save_interval == 0:

                    use_torchsctipt = True
                    eval_save_path = f"{args.save_dir}/{run_name}/eval_{args.exp_name}_{str(iteration)}"
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
                        print("Torchscript model loaded.")

                    else:
                        
                        model = torch.load(model_path, weights_only=False)
                        model.eval()
                        print("Pytorch model loaded.")

                    print("Evaluating model.")

                    eval_save_path = f"{args.save_dir}/{run_name}/eval_{args.exp_name}_{str(iteration)}"
                    # ...then roll out the model, accumulating returns.
                    episodic_returns_list = list()
                    zero_policy_returns_list = list()
                    random_policy_returns_list = list()

                    for i, eval_rollout_dict in eval_dict.items():

                        print(f"Rollout episode: {i}")

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


        # ALGO LOGIC: put action logic here
        # If during prelearning, sample actions using the specified method.

        # Wait until the experience sampling delay has passed before sampling actions.
        if global_step >= args.experience_sampling_delay:

            # If there is a prelearning phase, use it.
            if global_step < (args.learning_starts + args.actor_training_delay):

                if args.prelearning_sample == "scales":

                    if (iteration % args.max_episode_steps) == 0:
                        
                        print("Resetting scales.")

                        scales = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
                        action_scale = np.random.choice(scales)

                    # actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
                
                    actions = np.array([(actor.action_scale.cpu() * action_scale * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                    
                elif args.prelearning_sample == "normal":
                    
                    actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
                
                else:
                    
                    actions = np.array([(actor.action_scale.cpu() * envs.single_action_space.sample()) for _ in range(envs.num_envs)])

                # Now we'll get the hidden states for the actor and critics, but ignore the outputs.
                with torch.no_grad():

                    # If the actor takes prior actions and rewards as input, pass them in.
                    _, new_actor_hidden = actor(
                        torch.tensor(obs).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        actor_hidden
                    )
                    
                    _, new_qf1_hidden = qf1(
                        torch.tensor(obs).to(device),
                        torch.tensor(actions).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        qf1_hidden
                    )
                    _, new_qf2_hidden = qf2(
                        torch.tensor(obs).to(device),
                        torch.tensor(actions).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        qf2_hidden
                    )


            # Once prelearning is complete, sample actions using the actor.
            else:

                with torch.no_grad():

                    # If the actor takes prior actions and rewards as input, pass them in.
                    actions, new_actor_hidden = actor(
                        torch.tensor(obs).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        actor_hidden
                    )
                    
                    # Sample and add noise to the actions, then clip to action space bounds.
                    noise = noise_generator.sample().to(device)
                    actions = actions + noise
                    actions = actions.cpu().numpy().clip(
                        actor.action_scale.cpu().numpy() * envs.single_action_space.low,
                        actor.action_scale.cpu().numpy() * envs.single_action_space.high)
                    
                    qf1_a_values, new_qf1_hidden = qf1(
                        torch.tensor(obs).to(device),
                        torch.tensor(actions).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        qf1_hidden
                    )
                    qf2_a_values, new_qf2_hidden = qf2(
                        torch.tensor(obs).to(device),
                        torch.tensor(actions).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).to(torch.float32).to(device),
                        qf2_hidden
                    )


            # If using BPTT, we need to store the initial hidden states.
            # if initial_actor_hidden_in is None:
            #     initial_actor_hidden_in = actor.get_zero_hidden()
            #     initial_actor_hidden_out = initial_actor_hidden_in
            
            # Store the current rewards before generating a new transition.
            prior_rewards = rewards.copy()
            actor_hidden = (new_actor_hidden[0].detach().clone(),
                            new_actor_hidden[1].detach().clone())
            qf1_hidden = (new_qf1_hidden[0].detach().clone(),
                          new_qf1_hidden[1].detach().clone())
            qf2_hidden = (new_qf2_hidden[0].detach().clone(),
                          new_qf2_hidden[1].detach().clone())

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # If the obs are 8bit images, convert to float32 and rescale.
            if torch.tensor(obs).dtype != torch.float32:
                obs = np.array((obs / 255.0).astype(np.float32))
            # If the next obs are 8bit images, convert to float32 and rescale.
            if torch.tensor(next_obs).dtype != torch.float32:
                next_obs = np.array((next_obs / 255.0).astype(np.float32))
            # Rescale the rewards. By default, this is 1.0 and does nothing.
            rewards = args.reward_scale * rewards

            # If this is the first step of the episode, store the rewards.
            if first_step_reward is None:
                first_step_reward = rewards

            # If we've hit a writing interval, log the data.
            if iteration % args.writer_interval == 0 and (global_step > args.learning_starts + args.actor_training_delay):
                if global_step > args.learning_starts + args.actor_training_delay:
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
            episode_state.append(obs)
            episode_action.append(actions)
            episode_last_action.append(prior_actions)
            episode_reward.append(rewards)
            episode_last_reward.append(prior_rewards)
            episode_next_state.append(real_next_obs)
            episode_done.append(terminations)

            episode_state_ring.append(obs)
            episode_action_ring.append(actions)
            episode_last_action_ring.append(prior_actions)
            episode_reward_ring.append(rewards)
            episode_last_reward_ring.append(prior_rewards)
            episode_next_state_ring.append(real_next_obs)
            episode_done_ring.append(terminations)

            if len(episode_state_ring) > args.tbptt_seq_len:

                # If the ring buffer is full, we need to pop the oldest element.
                episode_state_ring.pop(0)
                episode_action_ring.pop(0)
                episode_last_action_ring.pop(0)
                episode_reward_ring.pop(0)
                episode_last_reward_ring.pop(0)
                episode_next_state_ring.pop(0)
                episode_done_ring.pop(0)

            if len(episode_state) == args.tbptt_seq_len:

                rb.push(initial_actor_hidden,
                        initial_qf1_hidden,
                        initial_qf2_hidden,
                        episode_state,
                        episode_action,
                        episode_last_action,
                        episode_reward,
                        episode_last_reward,
                        episode_next_state,
                        episode_done)
                

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

                    # terminations = [True]

                    print(f"\n\nglobal_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("episode/episodic_return", info["episode"]["r"], global_step)
                    episodic_return_gain = info["episode"]["r"] - (args.max_episode_steps * np.mean(first_step_reward) / args.reward_scale)
                    writer.add_scalar("episode/episodic_return_gain/", episodic_return_gain, global_step)
                    writer.add_scalar("episode/episodic_length", info["episode"]["l"], global_step)
                    print("Episode %d has ended with %d steps." % (global_step, info["episode"]["l"]))
                    print("Episode %d has ended with %d reward." % (global_step, info["episode"]["r"]))

                    first_step_reward = None
                    # Add this episodes return to the list...
                    train_episodic_return_list.append(episodic_return_gain)

                    # ...and if the list is now too long, pop the first element.
                    if len(train_episodic_return_list) > 100:
                        train_episodic_return_list.pop(0)

                    # episode_state_ring.append(obs)
                    # episode_action_ring.append(actions)
                    # episode_last_action_ring.append(prior_actions)
                    # episode_reward_ring.append(rewards)
                    # episode_last_reward_ring.append(prior_rewards)
                    # episode_next_state_ring.append(real_next_obs)
                    # episode_done_ring.append(terminations)

                    # Set the most recently pushed termination to True; the Env will not.
                    terminations[-1] = [True]


                    # If the ring buffer is full, we need to pop the oldest element.
                    # episode_state_ring.pop(0)
                    # episode_action_ring.pop(0)
                    # episode_last_action_ring.pop(0)
                    # episode_reward_ring.pop(0)
                    # episode_last_reward_ring.pop(0)
                    # episode_next_state_ring.pop(0)
                    # episode_done_ring.pop(0)

                    # TODO: Pop earliest from last and append done transition
                    # while len(episode_state) < args.tbptt_seq_len:
                    #      # If the episode is shorter than the TBPTT sequence length, pad it with the last transition.
                    #     episode_state.append(obs)
                    #     episode_action.append(actions)
                    #     episode_last_action.append(prior_actions)
                    #     episode_reward.append(rewards)
                    #     episode_last_reward.append(prior_rewards)
                    #     episode_next_state.append(real_next_obs)
                    #     episode_done.append(terminations)

                    rb.push(initial_actor_hidden,
                            initial_qf1_hidden,
                            initial_qf2_hidden,
                            episode_state_ring,
                            episode_action_ring,
                            episode_last_action_ring,
                            episode_reward_ring,
                            episode_last_reward_ring,
                            episode_next_state_ring,
                            episode_done_ring)

                    episode_state_ring = list()
                    episode_action_ring = list()
                    episode_last_action_ring = list()
                    episode_reward_ring = list()
                    episode_last_reward_ring = list()
                    episode_next_state_ring = list()
                    episode_done_ring = list()

                    # Set the initial hidden states to the current value.
                    # initial_actor_hidden = (new_actor_hidden[0].detach().clone(),
                    #                         new_actor_hidden[1].detach().clone())
                    # initial_qf1_hidden = (qf1_hidden[0].detach().clone(),
                    #                     qf1_hidden[1].detach().clone())
                    # initial_qf2_hidden = (qf2_hidden[0].detach().clone(),
                    #                     qf2_hidden[1].detach().clone())

                    print("Resetting Model Hidden State")
                    actor_hidden = actor.get_zero_hidden()
                    qf1_hidden = qf1.get_zero_hidden()
                    qf2_hidden = qf2.get_zero_hidden()

                    print("Warming up the hidden state")
                    num_warmup_steps = 10
                    actions = np.array([(actor.action_scale.cpu() * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                    obs, rewards, _, _, _ = envs.step(actions)

                    # If the obs are 8bit images, convert to float32 and rescale.
                    if torch.tensor(obs).dtype != torch.float32:
                        obs = np.array((obs / 255.0).astype(np.float32))
                    rewards = args.reward_scale * rewards
                    for _ in range(num_warmup_steps):

                        prior_actions = actions
                        prior_rewards = rewards
                        actions, actor_hidden = actor(
                            torch.tensor(obs).to(device),
                            torch.tensor(prior_actions).to(device),
                            torch.tensor(prior_rewards).to(torch.float32).to(device),
                            actor_hidden
                        )

                        actions = actions.cpu().numpy().clip(
                            actor.action_scale.cpu().numpy() * envs.single_action_space.low,
                            actor.action_scale.cpu().numpy() * envs.single_action_space.high)
                        
                        
                        _, qf1_hidden = qf1(
                            torch.tensor(obs).to(device),
                            torch.tensor(actions).to(device),
                            torch.tensor(prior_actions).to(device),
                            torch.tensor(prior_rewards).to(torch.float32).to(device),
                            qf1_hidden
                        )
                        _, qf2_hidden = qf2(
                            torch.tensor(obs).to(device),
                            torch.tensor(actions).to(device),
                            torch.tensor(prior_actions).to(device),
                            torch.tensor(prior_rewards).to(torch.float32).to(device),
                            qf2_hidden
                        )

                        obs, rewards, _, _, _ = envs.step(actions)
                        # If the obs are 8bit images, convert to float32 and rescale.
                        if torch.tensor(obs).dtype != torch.float32:
                            obs = np.array((obs / 255.0).astype(np.float32))
                        rewards = args.reward_scale * rewards

                    initial_actor_hidden = actor_hidden
                    initial_qf1_hidden = qf1_hidden
                    initial_qf2_hidden = qf2_hidden

                    print("Resetting Noise Generator")
                    noise_generator.reset()
                    if global_step > args.learning_starts:
                        noise_generator.decay()


                    break


            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # ALGO LOGIC: training.
        # If it is time to train, then train.
        if global_step > args.learning_starts:

            # Unpack data
            (actor_hidden_batch,
             qf1_hidden_batch,
             qf2_hidden_batch,
             observations_seq_batch,
             actions_batch_seq_batch,
             prior_actions_seq_batch,
             rewards_seq_batch,
             prior_rewards_seq_batch,
             next_observations_seq_batch,
             dones_seq_batch) = rb.sample(args.batch_size,
                                      device=device,)


            actor_hidden_batch = (torch.stack(actor_hidden_batch[0], dim=1).detach(),
                                  torch.stack(actor_hidden_batch[1], dim=1).detach())
            qf1_hidden_batch = (torch.stack(qf1_hidden_batch[0], dim=1).detach(),
                                torch.stack(qf1_hidden_batch[1], dim=1).detach())
            qf2_hidden_batch = (torch.stack(qf2_hidden_batch[0], dim=1).detach(),
                                torch.stack(qf2_hidden_batch[1], dim=1).detach())

            # print(f"Shape of observations_seq_batch: {np.array(observations_seq_batch).shape}")
            # print(f"Shape of actions_batch_seq_batch: {np.array(actions_batch_seq_batch).shape}")
            # print(f"Shape of prior_actions_seq_batch: {np.array(prior_actions_seq_batch).shape}")
            # print(f"Shape of rewards_seq_batch: {np.array(rewards_seq_batch).shape}")
            # print(f"Shape of prior_rewards_seq_batch: {np.array(prior_rewards_seq_batch).shape}")
            # print(f"Shape of next_observations_seq_batch: {np.array(next_observations_seq_batch).shape}")
            # print(f"Shape of dones_seq_batch: {np.array(dones_seq_batch).shape}")


            seq_of_observations_batchs = np.array(observations_seq_batch).transpose(1, 0, 2, 3, 4, 5)
            seq_of_actions_batchs = np.array(actions_batch_seq_batch).transpose(1, 0, 2, 3)
            seq_of_prior_actions_batchs = np.array(prior_actions_seq_batch).transpose(1, 0, 2, 3)
            seq_of_rewards_batchs = np.array(rewards_seq_batch).transpose(1, 0, 2)
            seq_of_prior_rewards_batchs = np.array(prior_rewards_seq_batch).transpose(1, 0, 2)
            seq_of_next_observations_batchs = np.array(next_observations_seq_batch).transpose(1, 0, 2, 3, 4, 5)
            seq_of_dones_batchs = np.array(dones_seq_batch).transpose(1, 0, 2)

            # print(f"Shape of seq_of_observations_batchs: {seq_of_observations_batchs.shape}")
            # print(f"Shape of seq_of_actions_batchs: {seq_of_actions_batchs.shape}")
            # print(f"Shape of seq_of_prior_actions_batchs: {seq_of_prior_actions_batchs.shape}")
            # print(f"Shape of seq_of_rewards_batchs: {seq_of_rewards_batchs.shape}")
            # print(f"Shape of seq_of_prior_rewards_batchs: {seq_of_prior_rewards_batchs.shape}")
            # print(f"Shape of seq_of_next_observations_batchs: {seq_of_next_observations_batchs.shape}")
            # print(f"Shape of seq_of_dones_batchs: {seq_of_dones_batchs.shape}")

            qf1_loss_total = 0.0
            qf2_loss_total = 0.0
            actor_loss_total = 0.0
            
            qf1_optimizer.zero_grad()
            qf2_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            for seq_idx, (observations_batch,
                          actions_batch,
                          prior_actions_batch,
                          rewards_batch,
                          prior_rewards_batch,
                          next_observations_batch,
                          dones_batch) in enumerate(
                                            zip(seq_of_observations_batchs,
                                                seq_of_actions_batchs,
                                                seq_of_prior_actions_batchs,
                                                seq_of_rewards_batchs,
                                                seq_of_prior_rewards_batchs,
                                                seq_of_next_observations_batchs,
                                                seq_of_dones_batchs)):
                
                print(f"Training on sequence {seq_idx + 1} of {len(seq_of_observations_batchs)}")
                

                # Print the hidden state shapes.
                # Actor hidden batch is a batch-length list of tuples, each tuple containing two tensors. 
                # print(f"actor_hidden_batch : {actor_hidden_batch}")
                # print(f"actor_hidden_batch length: {len(actor_hidden_batch)}")
                # print(f"actor_hidden_batch[0] : {actor_hidden_batch[0]}")
                # print(f"len(actor_hidden_batch[0]) : {len(actor_hidden_batch[0])}")
                # print(f"actor_hidden_batch[0][0] shape: {actor_hidden_batch[0][0].shape}")
                # print(f"actor_hidden_batch shape: {len(actor_hidden_batch)}, {actor_hidden_batch[0][0].shape}, {actor_hidden_batch[0][1].shape}")
                # print(f"qf1_hidden_batch shape: {len(qf1_hidden_batch)}, {qf1_hidden_batch[0][0].shape}, {qf1_hidden_batch[0][1].shape}")
                # print(f"qf2_hidden_batch shape: {len(qf2_hidden_batch)}, {qf2_hidden_batch[0][0].shape}, {qf2_hidden_batch[0][1].shape}")

            
                # retain_graph = (seq_idx < (len(seq_of_observations_batchs) - 1))


                # Convert to tensors and unsqueeze to add a sequence dimension.
                observations_batch =      torch.tensor(observations_batch, dtype=torch.float32, device=device)
                actions_batch =           torch.tensor(actions_batch, dtype=torch.float32, device=device)
                prior_actions_batch =     torch.tensor(prior_actions_batch, dtype=torch.float32, device=device)
                rewards_batch =           torch.tensor(rewards_batch, dtype=torch.float32, device=device)
                prior_rewards_batch =     torch.tensor(prior_rewards_batch, dtype=torch.float32, device=device)
                next_observations_batch = torch.tensor(next_observations_batch, dtype=torch.float32, device=device)
                dones_batch =             torch.tensor(dones_batch, dtype=torch.float32, device=device)

                # print("observations_batch shape:", observations_batch.shape)
                # print("actions_batch shape:", actions_batch.shape)
                # print("prior_actions_batch shape:", prior_actions_batch.shape)
                # print("rewards_batch shape:", rewards_batch.shape)
                # print("prior_rewards_batch shape:", prior_rewards_batch.shape)
                # print("next_observations_batch shape:", next_observations_batch.shape)
                # print("dones_batch shape:", dones_batch.shape)

                # Stop the gradients from flowing through the actor and target actor.
                with torch.no_grad():

    
                    # print("next_observations_batch shape:", next_observations_batch.shape)
                    # print("actions_batch shape:", actions_batch.shape)
                    # print("rewards_batch shape:", rewards_batch.shape)

                    next_state_actions_batch, _ = target_actor(
                            next_observations_batch.to(device),
                            actions_batch.to(device),
                            rewards_batch.to(device),
                            actor_hidden_batch,
                        )

                    # Next state action batch comes out without a sequence dimension, so we need to add it.
                    # next_state_actions_batch = next_state_actions_batch.unsqueeze(1)
                    # TODO: Two possible interventions here: remove noise or ensure the hidden batches are correct and not off by one
                    # TODO: Also it's unclear if dones really matter but I know I'm not handling them correctly.
                    # The next_q_value_batch never sees an update where it's just the reward becuase dones is never tru
                    policy_noise = 0.2

                    noise = (torch.randn_like(next_state_actions_batch) * policy_noise).clamp(-args.noise_clip, args.noise_clip)

                    noisy_next_action = (
                            next_state_actions_batch + noise
                        ).clamp(
                                float(envs.single_action_space.low[0]),
                                float(envs.single_action_space.high[0])
                            )
                    
                    if args.target_smoothing:
                        next_state_actions_batch = noisy_next_action

                    qf1_next_target_batch, _ = qf1_target(
                            next_observations_batch.to(device),
                            next_state_actions_batch.to(device),
                            actions_batch.to(device),
                            rewards_batch.to(device),
                            qf1_hidden_batch
                        )
                    
                    qf2_next_target_batch, _ = qf2_target(
                            next_observations_batch.to(device),
                            next_state_actions_batch.to(device),
                            actions_batch.to(device),
                            rewards_batch.to(device),
                            qf2_hidden_batch
                        )
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
                    # Unsqueeze the last dim of rewards and dones to add a length-1 sequence dim.
                    next_q_value_batch = rewards_batch.unsqueeze(-1) + (1 - dones_batch.unsqueeze(-1)) * args.gamma * (qf1_next_target_batch)
                    # next_q_value_batch = rewards_batch
                    # print(dones_batch)

                    # rewards_batch shape: torch.Size([8, 2, 1])
                    # dones_batch shape: torch.Size([8, 2, 1])
                    # qf1_next_target_batch shape: torch.Size([8, 2, 1])
                    # next_q_value_batch shape: torch.Size([8, 2, 16])


                # TODO: Why does this specific pass of action batch have a different shape than the seam call above when it's used as a_prior?
                # print("preinput actions_batch shape:", actions_batch.shape)

                # print(f"qf1_hidden_batch length: {len(qf1_hidden_batch)}")
                # print(f"len(qf1_hidden_batch[0]) : {len(qf1_hidden_batch[0])}")
                # print(f"qf1_hidden_batch[0][0] shape: {qf1_hidden_batch[0][0].shape}")

                qf1_a_values_batch, _ = qf1(
                    observations_batch.to(device),
                    actions_batch.to(device),
                    prior_actions_batch.to(device),
                    prior_rewards_batch.to(device),
                    qf1_hidden_batch
                )

                # print(f"qf1_hidden_batch length: {len(qf1_hidden_batch)}")
                # print(f"len(qf1_hidden_batch[0]) : {len(qf1_hidden_batch[0])}")
                # print(f"qf1_hidden_batch[0][0] shape: {qf1_hidden_batch[0][0].shape}")

                qf2_a_values_batch, _ = qf2(
                    observations_batch.to(device),
                    actions_batch.to(device),
                    prior_actions_batch.to(device),
                    prior_rewards_batch.to(device),
                    qf2_hidden_batch
                )

                # print("qf1_a_values_batch shape:", qf1_a_values_batch.shape)

                # print("qf2_a_values_batch shape:", qf2_a_values_batch.shape)
                # print("next_q_value_batch shape:", next_q_value_batch.shape)

                # print("qf1_a_values_batch:", qf1_a_values_batch)
                # print("qf2_a_values_batch:", qf2_a_values_batch)
                # print("next_q_value_batch:", next_q_value_batch)
                # TODO: There is an unresoelved broadcatcasting issue here.


                # Test that the loss inputs are the same shape, throw an error if they are not.
                if qf1_a_values_batch.shape != next_q_value_batch.shape:
                    raise ValueError(f"qf1_a_values_batch shape {qf1_a_values_batch.shape} does not match next_q_value_batch shape {next_q_value_batch.shape}")
                if qf2_a_values_batch.shape != next_q_value_batch.shape:
                    raise ValueError(f"qf2_a_values_batch shape {qf2_a_values_batch.shape} does not match next_q_value_batch shape {next_q_value_batch.shape}")



                qf1_loss = F.mse_loss(qf1_a_values_batch, next_q_value_batch)
                qf1_loss_total = qf1_loss_total + qf1_loss
                # qf1_loss = F.mse_loss(qf1_a_values_batch.view(-1), next_q_value_batch.view(-1))
                # qf1_loss = F.mse_loss(qf1_a_values_batch[:, -1, :], next_q_value_batch[:, -1, :])


                # qf2_loss = F.mse_loss(qf2_a_values_batch, next_q_value_batch)
                qf2_loss = F.mse_loss(qf2_a_values_batch, next_q_value_batch)
                qf2_loss_total = qf2_loss_total + qf2_loss


                if iteration % args.writer_interval == 0:
                    writer.add_scalar("losses/qf1_a_values", qf1_a_values_batch.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_a_values", qf2_a_values_batch.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_qf2_l2", torch.linalg.vector_norm(qf1_a_values_batch - qf2_a_values_batch).item(), global_step)
                    

                # If it a policy update step and we're past the actor training delay, update the actor.
                if (global_step > args.actor_training_delay + (args.learning_starts)) and (global_step % args.policy_frequency == 0):

                    loss_actions, actor_hidden_batch = actor(
                            observations_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device),
                            actor_hidden_batch
                        )
                    loss_qvalues, qf1_hidden_batch = qf1(
                        observations_batch.to(device),
                        loss_actions,
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device),
                        qf1_hidden_batch
                    )

                    # I added this because it makes sense this way.
                    _, qf2_hidden_batch = qf2(
                        observations_batch.to(device),
                        loss_actions,
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device),
                        qf2_hidden_batch
                    )

                    # actor_loss = -loss_qvalues.mean() + torch.linalg.vector_norm(loss_actions).item()
                    actor_loss = -loss_qvalues.mean()
                    actor_loss_total = actor_loss_total + actor_loss
        

            # Detach the hidden states to prevent backpropagation through the entire sequence.
            actor_hidden_batch = (actor_hidden_batch[0].detach(), actor_hidden_batch[1].detach())
            qf1_hidden_batch = (qf1_hidden_batch[0].detach(), qf1_hidden_batch[1].detach())
            qf2_hidden_batch = (qf2_hidden_batch[0].detach(), qf2_hidden_batch[1].detach())
            # Measure the time taken for this step.
            update_time = time.time()

            # ACTOR OPTIMIZATION BLOCK
            if (global_step > args.actor_training_delay + (args.learning_starts)) and (global_step % args.policy_frequency == 0):

                for p in qf1.parameters():
                    p.requires_grad = False
                # reg = weight_regularization(actor, l2_scale=args.l2_reg, l1_scale=args.l1_reg)
                # reg = 0.0
                # total_actor_loss = actor_loss_total
                (actor_loss_total / args.tbptt_seq_len).backward(retain_graph=True)
                # if iteration % args.writer_interval == 0:
                #     log_gradients_in_model(actor, writer, global_step, prefix="actor_grads")
                for p in qf1.parameters():
                    p.requires_grad = True
                if iteration % args.writer_interval == 0:
                    actor_grad = get_grad_norm(actor)
                    writer.add_scalar("grads/actor_grad", actor_grad, global_step)
                if args.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)
                    if iteration % args.writer_interval == 0:
                        actor_grad_clipped = get_grad_norm(actor)
                        writer.add_scalar("grads/actor_grad_clipped", actor_grad_clipped, global_step)
                actor_optimizer.step()
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # QF1 OPTIMIZATION BLOCK
            (qf1_loss_total / args.tbptt_seq_len).backward()
            # if iteration % args.writer_interval == 0:
            #     log_gradients_in_model(qf1, writer, global_step, prefix="qf1_grads")
            if iteration % args.writer_interval == 0:
                qf1_grad = get_grad_norm(qf1)
                writer.add_scalar("grads/qf1_grad", qf1_grad, global_step)
            if args.clip_gradients:
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=args.max_grad_norm)
                if iteration % args.writer_interval == 0:
                    qf1_grad_clipped = get_grad_norm(qf1)
                    writer.add_scalar("grads/qf1_grad_clipped", qf1_grad_clipped, global_step)
            qf1_optimizer.step()
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # QF2 OPTIMIZATION BLOCK
            (qf2_loss_total / args.tbptt_seq_len).backward()
            # if iteration % args.writer_interval == 0:
            #     log_gradients_in_model(qf2, writer, global_step, prefix="qf2_grads")
            if iteration % args.writer_interval == 0:
                qf2_grad = get_grad_norm(qf2)
                writer.add_scalar("grads/qf2_grad", qf2_grad, global_step)
            if args.clip_gradients:
                torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=args.max_grad_norm)
                if iteration % args.writer_interval == 0:
                    qf2_grad_clipped = get_grad_norm(qf2)
                    writer.add_scalar("grads/qf2_grad_clipped", qf2_grad_clipped, global_step)
            qf2_optimizer.step()
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)



            # Log the time taken for this step.
            update_time = time.time() - update_time

            if iteration % args.writer_interval == 0:
                writer.add_scalar("charts/update_time", update_time, global_step)

            # LOSS LOGGING BLOCK
            if iteration % args.writer_interval == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf1_loss_total", qf1_loss_total.item(), global_step)
                writer.add_scalar("losses/qf2_loss_total", qf2_loss_total.item(), global_step)

                if (global_step > args.actor_training_delay + (args.learning_starts)) and (global_step % args.policy_frequency == 0):
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss_total", actor_loss_total.item(), global_step)

        # STEP TIME LOGGING BLOCK
        if iteration % args.writer_interval == 0:
            print("Step time:", (time.time() - step_time) / args.num_envs)
            writer.add_scalar("charts/step_length", (time.time() - step_time), global_step)
            # writer.add_scalar("charts/update_time", update_time, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/step_SPS", (args.num_envs / (time.time() - step_time)), global_step)

        global_step += args.num_envs

    envs.close()
    writer.close()
    print("Done.")


