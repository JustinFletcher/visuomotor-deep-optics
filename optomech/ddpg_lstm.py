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



@dataclass
class Args:
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
    model_save_interval: int = 100
    """The interval between saving model weights"""
    writer_interval: int = 1000
    """The interval between recording to tensorboard"""

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

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

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
    low_dim_qnetwork: bool = False
    """Whether the qnetwork model is visual."""


    # Custom Algorthim Arguments
    """Which prelearning sample strategy to use (e.g., 'scales', 'normal')"""
    prelearning_sample: str = ""
    """How many steps to optimize the q function before actor training starts"""
    actor_training_delay: int = 10_000
    """Whether or not to use target smoothing"""
    target_smoothing: bool = False

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


class ReplayBufferWithHiddenStates:
    """
    A replay buffer that stores transitions including actor and critic hidden states.
    Each entry is a tuple:
    (actor_hidden, state, action, last_action, reward, last_reward, next_state, done)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, actor_hidden, state, action, last_action, reward, last_reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (actor_hidden, state, action, last_action, reward, last_reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        actor_hidden_list = []
        s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst = [], [], [], [], [], [], []
        for transition in batch:
            actor_hidden, state, action, last_action, reward, last_reward, next_state, done = transition
            actor_hidden_list.append(actor_hidden)
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            lr_lst.append(last_reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        actor_h = torch.cat([h for (h, c) in actor_hidden_list], dim=0)
        actor_c = torch.cat([c for (h, c) in actor_hidden_list], dim=0)
        actor_hidden_stacked = (actor_h, actor_c)

        return actor_hidden_stacked, s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst

    def __len__(self):
        return len(self.buffer)

class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, last_reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, last_reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, last_reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            lr_lst.append(last_reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class VanillaCritic(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True, action_scale=1.0):
        super().__init__()


        self.use_lstm = False

        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape
     
        vector_action_size = envs.single_action_space.shape[0]

        # Handle channels-last environments.
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = envs.single_observation_space.shape[-1]
        else:
            self.channels_last = False
            input_channels = envs.single_observation_space.shape[0]
        
        self.visual = not(low_dim)

        self.o_conv = nn.Sequential(
                # nn.MaxPool2d(4),
                conv_init(
                    nn.Conv2d(
                        input_channels, 
                        channel_scale, 
                        kernel_size=8, 
                        stride=4)
                        ),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(
                        channel_scale, 
                        channel_scale // 2, 
                        kernel_size=4, 
                        stride=2)
                        ),
                nn.ReLU(),
                conv_init(
                    nn.Conv2d(
                        channel_scale // 2,
                        channel_scale // 4,
                        kernel_size=3,
                        stride=1)
                    ),
                nn.Flatten(),
            )

        with torch.inference_mode():

            # Handle channels-last environments.
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            output_dim = self.o_conv(x).shape[1]

        if self.visual:
            self.merge_fc1 = uniform_init(
                nn.Linear(output_dim + vector_action_size, fc_scale),
                lower_bound=-1/np.sqrt(output_dim + vector_action_size),
                upper_bound=1/np.sqrt(output_dim + vector_action_size))
        
        else:
            self.merge_fc1 = uniform_init(
                nn.Linear(vector_action_size, fc_scale),
                lower_bound=-1/np.sqrt(vector_action_size),
                upper_bound=1/np.sqrt(vector_action_size))
            
        self.merge_fc2 = uniform_init(
            nn.Linear(fc_scale, fc_scale // 2),
            lower_bound=-1/np.sqrt(fc_scale),
            upper_bound=1/np.sqrt(fc_scale))

        self.merge_fc3 = uniform_init(
            nn.Linear(fc_scale // 2, fc_scale // 4),
            lower_bound=-1/np.sqrt(fc_scale // 2),
            upper_bound=1/np.sqrt(fc_scale // 2))

        self.fc_q = nn.Linear(fc_scale // 4, 1)

    def forward(self, o, a):

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        if self.visual:
            o = o / 255.0
            x_o = F.relu(self.o_conv(o.to(torch.float32)))
            x = torch.cat([x_o, a], 1)

        else: 
            x = a 

        x = F.relu(self.merge_fc1(x))
        x = F.relu(self.merge_fc2(x))
        x = F.relu(self.merge_fc3(x))
        q_vals = self.fc_q(x)

        return q_vals

class VanillaActor(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True, action_scale=1.0):
        super().__init__()

        self.use_lstm = False
        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape

        vector_action_size = envs.single_action_space.shape[0]

        # Check if this is a channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = envs.single_observation_space.shape[-1]
        else:
            self.channels_last = False
            input_channels = envs.single_observation_space.shape[0]

        self.visual = not(low_dim)

        self.conv = nn.Sequential(
                # nn.MaxPool2d(4),
                conv_init(
                    nn.Conv2d(input_channels, 
                              channel_scale,
                              kernel_size=8,
                              stride=4)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale,
                                    channel_scale // 2,
                                    kernel_size=4,
                                    stride=2)),
                nn.ReLU(),
                conv_init(nn.Conv2d(channel_scale // 2,
                                    channel_scale // 4,
                                    kernel_size=3,
                                    stride=1)),
                nn.Flatten(),
            )
        
        with torch.inference_mode():

            # Handle channels-last environments.
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            output_dim = self.conv(x).shape[1]

        self.ones_output = torch.ones(1, output_dim)

        self.fc1 = uniform_init(
            nn.Linear(output_dim, fc_scale // 2),
            lower_bound=-1/np.sqrt(output_dim),
            upper_bound=1/np.sqrt(output_dim)
            )
        self.fc2 = uniform_init(
            nn.Linear(fc_scale // 2, fc_scale // 4),
            lower_bound=-1/np.sqrt(fc_scale // 2),
            upper_bound=1/np.sqrt(fc_scale // 2)
            )
        # self.fc3 = uniform_init(
        #     nn.Linear(
        #         fc_scale,
        #         int(np.prod(envs.single_action_space.shape))
        #         ),
        #     lower_bound=-3e-4,
        #     upper_bound=3e-4
        #     )
        self.fc3 = uniform_init(
            nn.Linear(fc_scale // 4, int(np.prod(envs.single_action_space.shape))),
            lower_bound=-1/np.sqrt(fc_scale // 4),
            upper_bound=1/np.sqrt(fc_scale // 4)
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

    def forward(self, x):

        # Handle channels-last environments.
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)

        if self.visual:
            x = x / 255.0
            print(x.dtype)
            x = F.relu(self.conv(x.to(torch.float32)))
        else:
            x = self.ones_output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # x = F.tanh(self.fc1(x))

        a = (x * self.action_scale + self.action_bias)
        return a
    

class MultiscaleActionModule(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=128):
        super(MultiscaleActionModule, self).__init__()

        self.head0 = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, out_size),
                std=1.0
            ),
            nn.Tanh()
        )
        self.head0_logits = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1),
                std=1.0
            ),
        )

        self.head1 = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, out_size),
                std=1.0
            ),
            nn.Tanh()
        )
        self.head1_logits = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1),
                std=1.0
            ),
        )

        self.head2 = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, out_size),
                std=1.0
            ),
            nn.Tanh()
        )
        self.head2_logits = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1),
                std=1.0
            ),
        )

        self.head3 = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, out_size),
                std=1.0
            ),
            nn.Tanh()
        )
        self.head3_logits = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1),
                std=1.0
            ),
        )

        self.head4 = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, out_size),
                std=1.0
            ),
            nn.Tanh()
        )
        self.head4_logits = nn.Sequential(
            layer_init(nn.Linear(in_size, hidden_size),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1),
                std=1.0
            ),
        )


    def forward(self, x):

        # Process input through each path
        out0 = (1 / (10 ** 0)) * self.head0(x)
        active0 = (self.head0_logits(x) > 0.5).float()

        out1 = (1 / (10 ** 1)) * self.head1(x)
        active1 = (self.head1_logits(x) > 0.5).float()

        out2 = (1 / (10 ** 2)) * self.head2(x)
        active2 = (self.head2_logits(x) > 0.5).float()

        out3 = (1 / (10 ** 3)) * self.head3(x)
        active3 = (self.head3_logits(x) > 0.5).float()

        out4 = (1 / (10 ** 4)) * self.head4(x)
        active4 = (self.head4_logits(x) > 0.5).float()

        # Concatenate along the channel dimension
        out = (out0 * active0) + (out1 * active1) + (out2 * active2) + (out3 * active3) + (out4 * active4)
        return out

class ImpalaActor(nn.Module):

    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, use_multiscale_head=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.device = device
        self.use_lstm = True
        self.bptt = bptt
        self.lstm_hidden_dim = fc_scale
        self.lstm_num_layers = lstm_num_layers
        self.use_multiscale_head = use_multiscale_head

        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]

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
        )

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
                std=np.sqrt(2.0)
            ),
            nn.LayerNorm(mlp_output_size),
            nn.ReLU(),
        )


        self.lstm = nn.LSTM(
            input_size=mlp_output_size + vector_action_size + 1,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        # Get the output shape of the LSTM
        with torch.inference_mode():            
        
            x = torch.zeros(1, mlp_output_size + vector_action_size + 1)

            pre_head_output_shape = self.lstm(x)[0].shape

        if self.use_multiscale_head:

            self.action_head = nn.Sequential(
                MultiscaleActionModule(
                        int(np.prod(pre_head_output_shape[1:])),
                        int(np.prod(envs.single_action_space.shape)))
                        )

        else:
            # Build the action head following the convolutional LSTM
            self.action_head = nn.Sequential(
                layer_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        fc_scale // 2,
                            ),
                    std=1e-3
                ),
                nn.LayerNorm(fc_scale // 2),
                nn.Tanh(),
                layer_init(
                    nn.Linear(
                        fc_scale // 2,
                        int(np.prod(envs.single_action_space.shape))
                            ),
                    std=1e-3
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

        # Important: Set the hidden state to None initially.

        # self.hidden = self.init_hidden()

    def get_zero_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        return (h, c)


    def forward(self, o, a_prior, r_prior, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        

        if hidden is None:
            h_0 = torch.zeros(self.lstm_num_layers, o.size(0), self.lstm_hidden_dim, dtype=o.dtype, device=o.device)
            c_0 = torch.zeros(self.lstm_num_layers, o.size(0), self.lstm_hidden_dim, dtype=o.dtype, device=o.device)
            hidden = (h_0, c_0)
        # batch_size, seq_len, channels, height, width = o.shape

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        # TODO: here, reshape the input to allow the visual encoder to apply to all images.
        # x_reshaped = x.view(batch_size * seq_len, channels, height, width) 
        x_o = self.visual_encoder(o)

        # TODO: here, reshape is back to get a sequence again before the LSTM
        # conv_out = conv_out.view(batch_size, seq_len, 16, height, width)
        x = self.mlp(x_o)

        h0, c0 = hidden
        if h0.ndim == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        else:
            h0 = h0.permute(1, 0, 2)
            c0 = c0.permute(1, 0, 2)
        hidden = (h0, c0)
        x = torch.cat([x, a_prior, r_prior], 1)
        # TODO: this will require a total refactor for BPTT
        # Add a sequence dimension to the input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, new_hidden = self.lstm(x, hidden)
        # Remove the sequence dimension if it was added
        if x.dim() == 3:
            x = x.squeeze(1)
        #     new_hidden = (new_hidden[0].squeeze(1), new_hidden[1].squeeze(1))
        if new_hidden[0].ndim == 3:

            new_hidden = (new_hidden[0].squeeze(0), new_hidden[1].squeeze(0))

        # Apply action prediciton head and activation function
        a = self.action_head(x)

        a = (a * self.action_scale + self.action_bias)

        if new_hidden[0].ndim == 3:
            raise ValueError("LSTM hidden state has 3 dimensions, expected 2.")
        if new_hidden[1].ndim == 3:
            raise ValueError("LSTM cell state has 3 dimensions, expected 2.")
        return a, new_hidden

class ImpalaCritic(nn.Module):
    
    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.use_lstm = True
        self.device = device
        self.bptt = bptt

        self.lstm_hidden_dim = fc_scale
        self.lstm_num_layers = lstm_num_layers


        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]

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
        )

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
            nn.LayerNorm(mlp_output_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=mlp_output_size + vector_action_size + vector_action_size + 1,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        # Get the output shape of the LSTM
        with torch.inference_mode():
            x = torch.zeros(1, mlp_output_size + vector_action_size + vector_action_size + 1)
            pre_head_output_shape = self.lstm(x)[0].shape

        # Build the q head following the convolutional LSTM
        self.q_head = nn.Sequential(
            layer_init(
                nn.Linear(
                    int(np.prod(pre_head_output_shape[1:])),
                    fc_scale // 2
                    ),
                std=1.0
            ),
            nn.LayerNorm(fc_scale // 2),
            nn.Tanh(),
            layer_init(
                nn.Linear(fc_scale // 2,
                          1
                          ),
                std=1.0
            ),
        )


    def get_zero_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        return (h, c)

    def forward(self, o, a, a_prior, r_prior, hidden=None):

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        # Extract visual feature maps
        x_o = self.visual_encoder(o)
        x = self.mlp(x_o)

        h0, c0 = hidden
        if h0.ndim == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        else:
            h0 = h0.permute(1, 0, 2)
            c0 = c0.permute(1, 0, 2)
        hidden = (h0, c0)
        x = torch.cat([x, a, a_prior, r_prior], 1)

        # TODO: this will require a total refactor for BPTT
        # Add a sequence dimension to the input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, new_hidden = self.lstm(x, hidden)
        # Remove the sequence dimension if it was added
        if x.dim() == 3:
            x = x.squeeze(1)
            new_hidden = (new_hidden[0].squeeze(1), new_hidden[1].squeeze(1))

        # Apply action prediciton head and activation function
        q = self.q_head(x)
        return q, new_hidden
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    

class ImpalaLargeActor(nn.Module):

    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, use_multiscale_head=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.device = device
        self.use_lstm = True
        self.bptt = bptt
        self.lstm_hidden_dim = fc_scale
        self.lstm_num_layers = lstm_num_layers
        self.use_multiscale_head = use_multiscale_head

        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]

        # Build a visual encoder comprising two standard convs followed by two residual blocks.

        encoder_block_1 = nn.Sequential(
            conv_init(
                nn.Conv2d(input_channels, 
                          16,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(16, 16),
            nn.ReLU(),
            ResidualBlock(16, 16),
        )

        encoder_block_2 = nn.Sequential(
            conv_init(
                nn.Conv2d(16, 
                          32,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(32, 32),
            nn.ReLU(),
            ResidualBlock(32, 32),
        )


        encoder_block_3 = nn.Sequential(
            conv_init(
                nn.Conv2d(32, 
                          32,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(32, 32),
            nn.ReLU(),
            ResidualBlock(32, 32),
        )


        self.visual_encoder = nn.Sequential(
            encoder_block_1,
            encoder_block_2,
            encoder_block_3,
            nn.ReLU(),
        )
    

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

        if self.use_multiscale_head:

            self.action_head = nn.Sequential(
                MultiscaleActionModule(
                        int(np.prod(pre_head_output_shape[1:])),
                        int(np.prod(envs.single_action_space.shape)))
                        )

        else:
            # Build the action head following the convolutional LSTM
            self.action_head = nn.Sequential(
                layer_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        fc_scale
                        ),
                    std=1.0
                ),
                nn.Tanh(),
                layer_init(
                    nn.Linear(
                        fc_scale,
                        int(np.prod(envs.single_action_space.shape))
                        ),
                    std=1.0
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

        # Important: Set the hidden state to None initially.
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        return (h, c)

    def reset_hidden(self, ):
        """
        Reset the agent's internal hidden state to zeros.
        """
        self.hidden = self.init_hidden()

    def forward(self, o, a_prior, r_prior):


        # batch_size, seq_len, channels, height, width = o.shape

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)


        # TODO: here, reshape the input to allow the visual encoder to apply to all images.
        # x_reshaped = x.view(batch_size * seq_len, channels, height, width) 
        x_o = self.visual_encoder(o)

        # TODO: here, reshape is back to get a sequence again before the LSTM
        # conv_out = conv_out.view(batch_size, seq_len, 16, height, width)
        x = self.mlp(x_o)
        x, new_hidden = self.lstm(torch.cat([x, a_prior, r_prior], 1), self.hidden)


        # new_hidden is a tuple (h, c) after processing x
        if self.bptt:
            detached_hidden = new_hidden[0], new_hidden[1]
        else:
            detached_hidden = new_hidden[0].detach().to(self.device), new_hidden[1].detach().to(self.device)
        self.hidden = detached_hidden

        # Apply action prediciton head and activation function
        a = self.action_head(x)

        a = (a * self.action_scale + self.action_bias)
        return a


class ImpalaLargeCritic(nn.Module):
    
    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.use_lstm = True
        self.device = device
        self.bptt = bptt

        self.lstm_hidden_dim = fc_scale
        self.lstm_num_layers = lstm_num_layers


        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image

        obs_shape = envs.single_observation_space.shape

        # Check if this is a channels-last environment
        if obs_shape[-1] < obs_shape[0]:
            self.channels_last = True
            input_channels = obs_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_shape[0]

        # Define the visual encoder, so that we know the output shape.
        encoder_block_1 = nn.Sequential(
            conv_init(
                nn.Conv2d(input_channels, 
                          16,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(16, 16),
            nn.ReLU(),
            ResidualBlock(16, 16),
        )

        encoder_block_2 = nn.Sequential(
            conv_init(
                nn.Conv2d(16, 
                          32,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(32, 32),
            nn.ReLU(),
            ResidualBlock(32, 32),
        )


        encoder_block_3 = nn.Sequential(
            conv_init(
                nn.Conv2d(32, 
                          32,
                          kernel_size=3,
                          stride=1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(32, 32),
            nn.ReLU(),
            ResidualBlock(32, 32),
        )


        self.visual_encoder = nn.Sequential(
            encoder_block_1,
            encoder_block_2,
            encoder_block_3,
            nn.ReLU(),
        )
    
    
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
            x = torch.zeros(1, mlp_output_size + vector_action_size + vector_action_size + 1)
            pre_head_output_shape = self.lstm(x)[0].shape

        # Build the q head following the convolutional LSTM
        self.q_head = nn.Sequential(
            layer_init(
                nn.Linear(int(np.prod(pre_head_output_shape[1:])), fc_scale),
                std=np.sqrt(2)
            ),
            nn.ReLU(),
            layer_init(
                nn.Linear(fc_scale, 1),
                std=1.0
            ),
        )

        # Important: Set the hidden state to None initially.
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Create a fresh hidden state of zeros (h, c) for a single-layer LSTM.
        Shapes:
          h, c: [num_layers=1, batch_size, hidden_dim]
        """
        # TODO: Consider requires_grad_() here
        h = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.lstm_hidden_dim,).to(self.device) 
        return (h, c)

    def reset_hidden(self):
        """
        Reset the agent's internal hidden state to zeros.
        """
        self.hidden = self.init_hidden()


    def forward(self, o, a, a_prior, r_prior):
    

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        # Extract visual feature maps
        x_o = self.visual_encoder(o)
        x = self.mlp(x_o)

        x, new_hidden = self.lstm(torch.cat([x, a, a_prior, r_prior], 1), self.hidden)
        # new_hidden is a tuple (h, c) after processing x
        if self.bptt:
            detached_hidden = new_hidden[0], new_hidden[1]
        else:
            detached_hidden = new_hidden[0].detach().to(self.device), new_hidden[1].detach().to(self.device)
        self.hidden = detached_hidden

        # Apply action prediciton head and activation function
        q = self.q_head(x)
        return q


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


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


def log_weights_in_model(model, logger, step):
    for tag, value in model.named_parameters():
            logger.add_histogram(tag + "/grad", value.cpu(), step)


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
    
    bptt = False
    

    if args.actor_type == "impala" or args.actor_type == "impalalarge":
        prior_state_models = True
    else:
        prior_state_models = False

    if args.actor_type == "vanilla":
    
        # actor = Actor(envs).to(device)
        actor = VanillaActor(envs,
                    channel_scale=args.actor_channel_scale,
                    fc_scale=args.actor_fc_scale,
                    low_dim=args.low_dim_actor,
                           action_scale=args.action_scale).to(device)
        
        # target_actor = Actor(envs).to(device)
        target_actor = VanillaActor(envs,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                            low_dim=args.low_dim_actor,
                           action_scale=args.action_scale).to(device)
        
    elif args.actor_type == "impala":

        actor = ImpalaActor(envs,
                            device,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                           action_scale=args.action_scale,
                           use_multiscale_head=args.use_multiscale_head).to(device)
        target_actor = ImpalaActor(envs,
                                   device,
                                   channel_scale=args.actor_channel_scale,
                                   fc_scale=args.actor_fc_scale,
                           use_multiscale_head=args.use_multiscale_head,
                           action_scale=args.action_scale).to(device)
    
        

    elif args.actor_type == "impalalarge":
        
        actor = ImpalaLargeActor(envs,
                            device,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                           action_scale=args.action_scale,
                           use_multiscale_head=args.use_multiscale_head).to(device)
        target_actor = ImpalaLargeActor(envs,
                                   device,
                                   channel_scale=args.actor_channel_scale,
                                   fc_scale=args.actor_fc_scale,
                           use_multiscale_head=args.use_multiscale_head,
                           action_scale=args.action_scale).to(device)


    else:

        raise ValueError("Invalid actor type specified.")
    

    if args.critic_type == "vanilla":

        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf1 = VanillaCritic(
            envs,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            low_dim=args.low_dim_qnetwork,
            action_scale=args.action_scale).to(device)
        qf1_target = VanillaCritic(
            envs,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            low_dim=args.low_dim_qnetwork,
            action_scale=args.action_scale).to(device)
        
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf2 = VanillaCritic(
            envs,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            low_dim=args.low_dim_qnetwork,
            action_scale=args.action_scale).to(device)
        qf2_target = VanillaCritic(
            envs,
            channel_scale=args.qnetwork_channel_scale,
            fc_scale=args.qnetwork_fc_scale,
            low_dim=args.low_dim_qnetwork,
            action_scale=args.action_scale).to(device)

    elif args.critic_type == "impala":

        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf1 = ImpalaCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device
        )
        qf1_target = ImpalaCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device
        )
        
        _ = torch.randn(1000)
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        _ = torch.randn(1000)
        qf2 = ImpalaCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)
        qf2_target = ImpalaCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)
        
        for p in qf2.parameters():
            p.data += 1e-3 * torch.randn_like(p)
        for p in qf2_target.parameters():
            p.data += 1e-3 * torch.randn_like(p)

        
    elif args.critic_type == "impalalarge":

        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        qf1 = ImpalaLargeCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)
        qf1_target = ImpalaLargeCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)
        _ = torch.randn(1000)
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        _ = torch.randn(1000)
        qf2 = ImpalaLargeCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)
        qf2_target = ImpalaLargeCritic(
            envs,
            device,
            channel_scale=args.actor_channel_scale,
            fc_scale=args.actor_fc_scale,
            action_scale=args.action_scale).to(device)

    else:
        raise ValueError("Invalid critic type specified.")
    

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(SimpleModel())  # Wrap the model with DataParallel
    # else:
    #     model = SimpleModel()
    print("Actor and Critic parameter counts:")
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print(sum(p.numel() for p in qf1.parameters() if p.requires_grad))

    summary(actor, )
    summary(qf1,)

    # Write a test to fail if qf1 and qf2 have any identical parameters.
    any_different = False
    for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
        if not(p1.data.equal(p2.data)):
            any_different = True
            break

    # print([torch.allclose(p1, p2) for p1, p2 in zip(qf1.parameters(), qf2.parameters())])
    # print("qf1 and qf2 have different parameters.")
    

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    qf1_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_learning_rate)
    qf2_optimizer = optim.Adam(list(qf2.parameters()), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    envs.single_observation_space.dtype = np.float32

    if bptt:

        rb = ReplayBufferLSTM2(args.buffer_size)
        
    elif prior_state_models:

        rb = ReplayBufferWithHiddenStates(args.buffer_size)


    else:

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
            n_envs=args.num_envs
        )

    num_eval_rollouts = 4

    eval_dict = dict()
    for eval_rollout in range(num_eval_rollouts):

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


    episode_state = list()
    episode_action = list()
    episode_last_action = list()
    episode_reward = list()
    episode_last_reward = list()
    episode_next_state = list()
    episode_done = list()

    global_step = 0
    best_train_episode_return = -np.inf
    train_episodic_return_list = list()
    best_eval_episode_return = -np.inf
    initial_actor_hidden_in = None
    initial_actor_hidden_out = None
    initial_qf1_hidden_in = None
    initial_qf1_hidden_out = None
    initial_qf2_hidden_in = None
    initial_qf2_hidden_out = None
    actor_hidden = actor.get_zero_hidden()
    new_actor_hidden = actor.get_zero_hidden()
    qf1_hidden = qf1.get_zero_hidden()
    qf2_hidden = qf2.get_zero_hidden()    
    first_step_reward = None

    actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])

    noise_generator = OUNoiseTorch(
                 action_dim=envs.single_action_space.shape[0], 
                 mu=0.0, 
                 theta=0.15, 
                 sigma_initial=args.exploration_noise, 
                 min_sigma=args.exploration_noise * 0.01, 
                 decay_rate=args.decay_rate,
                 auto_decay=False,
                 device='cpu')

    print(actions)
    _, rewards, _, _, _ = envs.step(actions)

    for iteration in range(args.total_timesteps):
        print("Iteration: ", iteration)
        print("global_step: ", global_step)

     
        if args.save_model:

            # Check that we've complete at least one post-learning episode.
            if global_step > (args.learning_starts + args.max_episode_steps):

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
                    mean_zero_return_advantage = np.mean(episodic_returns_array / zero_policy_returns_array)
                    mean_random_return_advantage = np.mean(episodic_returns_array / random_policy_returns_array)

                    writer.add_scalar("eval/best_policy_mean_returns", mean_eval_episode_return, iteration)
                    writer.add_scalar("eval/best_policy_zero_return_advantage", mean_zero_return_advantage, iteration)
                    writer.add_scalar("eval/best_policy_random_return_advantage", mean_random_return_advantage, iteration)


                    print(f"Mean eval episodic return: {mean_eval_episode_return}")


            if iteration % args.model_save_interval == 0:

                use_torchsctipt = True
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
                    print("Torchscript model loaded.")

                else:
                    
                    model = torch.load(model_path, weights_only=False)
                    model.eval()
                    print("Pytorch model loaded.")

                print("Evaluating model.")

                eval_save_path = f"runs/{run_name}/eval_{args.exp_name}_{str(iteration)}"
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
                

        step_time = time.time()

        prior_actions = actions.copy()

        # ALGO LOGIC: put action logic here
        # If during prelearning, sample actions using the specified method.
        if global_step < (args.learning_starts + args.actor_training_delay):

            if args.prelearning_sample == "scales":

                if (iteration % args.max_episode_steps) == 0:
                    
                    print("Resetting scales.")

                    scales = [0.000001, 0.00001,0.0001, 0.001, 0.01, 0.1, 1.0]
                    action_scale = np.random.choice(scales)

                # actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
            
                actions = np.array([(actor.action_scale.cpu() * action_scale * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                
            elif args.prelearning_sample == "normal":
                
                actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
            
            else:
                
                actions = np.array([(actor.action_scale.cpu() * envs.single_action_space.sample()) for _ in range(envs.num_envs)])

        # Once prelearning is complete, sample actions using the actor.
        else:

            with torch.no_grad():


                # If the obs are 8bit images, convert to float32 and rescale.
                if torch.tensor(obs).dtype != torch.float32:

                        obs = np.array((obs / 255.0).astype(np.float32))

                # If the actor takes prior actions and rewards as input, pass them in.
                if args.actor_type == "impala" or args.actor_type == "impalalarge":
    
                    actions, new_actor_hidden = actor(
                        torch.tensor(obs).to(device),
                        torch.tensor(prior_actions).to(device),
                        torch.tensor(prior_rewards).unsqueeze(0).to(torch.float32).to(device),
                        actor_hidden
                    )
                
                # Otherwise, just pass in the obs.
                else:

                    actions = actor(torch.Tensor(obs).to(device))

                # Sample and add noise to the actions, then clip to action space bounds.
                noise = noise_generator.sample().to(device)
                actions += noise
                actions = actions.cpu().numpy().clip(
                    actor.action_scale.cpu().numpy() * envs.single_action_space.low,
                    actor.action_scale.cpu().numpy() * envs.single_action_space.high)

        # If using BPTT, we need to store the initial hidden states.
        # if initial_actor_hidden_in is None:
        #     initial_actor_hidden_in = actor.get_zero_hidden()
        #     initial_actor_hidden_out = initial_actor_hidden_in
        
        # Store the current rewards before generating a new transition.
        # TODO: should this be a copy? 
        prior_rewards = rewards

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Rescale the rewards.
        rewards = args.reward_scale * rewards

        # If this is the first step of the episode, store the rewards.
        if first_step_reward is None:
            first_step_reward = rewards

        if iteration % args.writer_interval == 0 and (global_step > args.learning_starts + args.actor_training_delay):
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

        if iteration % args.writer_interval == 0 and (global_step < args.learning_starts + args.actor_training_delay):
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
        if bptt:
            episode_state.append(obs)
            episode_action.append(actions)
            episode_last_action.append(prior_actions)
            episode_reward.append(rewards)
            episode_last_reward.append(prior_rewards)
            episode_next_state.append(next_obs)
            episode_done.append(terminations)  

        # If not using BPTT, we push the data to the replay buffer immediately.
        elif prior_state_models:



            # Push the transition, including hidden states, to the replay buffer.
            # rb.push(ini_hidden_in,
            #         ini_hidden_out,
            #         obs,
            #         actions,
            #         prior_actions,
            #         rewards,
            #         prior_rewards,
            #         real_next_obs,
            #         terminations)

            if actor_hidden[0].ndim == 3:
            
                raise ValueError("actor hidden 3 dimensions, expected 2.")

            rb.push(actor_hidden,
                    obs,
                    actions,
                    prior_actions,
                    rewards,
                    prior_rewards,
                    real_next_obs,
                    terminations)

        # If not using BPTT or prior state models, we push the data to the replay buffer immediately.
        else: 

            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)


         # TRY NOT TO MODIFY: record rrewards for plotting purposes
        if infos:

            for info in infos["final_info"]:


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

                if bptt:

                    rb.push(ini_hidden_in,
                            ini_hidden_out,
                            episode_state,
                            episode_action,
                            episode_last_action,
                            episode_reward,
                            episode_last_reward,
                            episode_next_state,
                            episode_done)
                    
                    ini_hidden_in = None
                    ini_hidden_out = None
                    episode_state = list()
                    episode_action = list()
                    episode_last_action = list()
                    episode_reward = list()
                    episode_last_reward = list()
                    episode_next_state = list()
                    episode_done = list()

                noise_generator.reset()
                if global_step > args.learning_starts:
                    noise_generator.decay()
                if actor.use_lstm:
                    print("Resetting Actor Hidden State")
                    actor_hidden = actor.get_zero_hidden()

                break


        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        actor_hidden = new_actor_hidden


        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            # Unpack data
            if bptt:

                (hidden_ins_batch,
                hidden_outs_batch,
                observations_batch,
                actions_batch_batch,
                prior_actions_batch,
                rewards_batch,
                prior_rewards_batch,
                next_observations_batch,
                dones_batch) = rb.sample(args.batch_size)

            elif prior_state_models:

                # (hidden_ins_batch,
                #  hidden_outs_batch,
                #  observations_batch,
                #  actions_batch_batch,
                #  prior_actions_batch,
                #  rewards_batch,
                #  prior_rewards_batch,
                #  next_observations_batch,
                #  dones_batch) = rb.sample(args.batch_size)
                
                # hidden_ins_batch = hidden_ins_batch
                # hidden_outs_batch = hidden_outs_batch
                # observations_batch = torch.tensor(np.array(observations_batch), dtype=torch.float32).squeeze(1)
                # actions_batch_batch = torch.tensor(np.array(actions_batch_batch), dtype=torch.float32).squeeze(1)
                # prior_actions_batch = torch.tensor(np.array(prior_actions_batch), dtype=torch.float32).squeeze(1)
                # rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float32).to(device)
                # prior_rewards_batch = torch.tensor(np.array(prior_rewards_batch), dtype=torch.float32)
                # next_observations_batch = torch.tensor(np.array(next_observations_batch), dtype=torch.float32).squeeze(1)
                # dones_batch = torch.tensor(np.array(dones_batch), dtype=torch.float32).to(device)


                (actor_hidden_batch,
                 observations_batch, 
                 actions_batch_batch,
                 prior_actions_batch,
                 rewards_batch,
                 prior_rewards_batch,
                 next_observations_batch,
                 dones_batch) = rb.sample(args.batch_size)
                
                actor_hidden_batch = actor_hidden_batch
                observations_batch = torch.tensor(np.array(observations_batch), dtype=torch.float32).squeeze(1)
                actions_batch_batch = torch.tensor(np.array(actions_batch_batch), dtype=torch.float32).squeeze(1)
                prior_actions_batch = torch.tensor(np.array(prior_actions_batch), dtype=torch.float32).squeeze(1)
                rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float32).to(device)
                prior_rewards_batch = torch.tensor(np.array(prior_rewards_batch), dtype=torch.float32)
                next_observations_batch = torch.tensor(np.array(next_observations_batch), dtype=torch.float32).squeeze(1)
                dones_batch = torch.tensor(np.array(dones_batch), dtype=torch.float32).to(device)
                
            else:
                
                data = rb.sample(args.batch_size)

                observations_batch = data.observations
                actions_batch_batch = data.actions
                rewards_batch = data.rewards
                next_observations_batch = data.next_observations
                dones_batch = data.dones

            with torch.no_grad():

                if prior_state_models:

                    # TODO: I belive target smoothing goes here, added to next_state_actions_batch.
                    # TODO: Make sure the noise doesn't need to be random per-element.

                    next_state_actions_batch, _ = target_actor(
                            next_observations_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device),
                            actor_hidden_batch,
                        )
                    policy_noise = 0.01

                    noise = (torch.randn_like(next_state_actions_batch) * policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    # TODO: WARNING: This will break asymmetric action spaces.
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
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device),
                            actor_hidden_batch
                        )
                    
                    qf2_next_target_batch, _ = qf2_target(
                            next_observations_batch.to(device),
                            next_state_actions_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device),
                            actor_hidden_batch
                        )
                    qf1_next_target_batch = torch.min(
                            qf1_next_target_batch,
                            qf2_next_target_batch
                        )

                else:

                    # TODO: Verify if this is the proper use of hidden states.
                    next_state_actions_batch, _ = target_actor(
                            next_observations_batch,
                            actor_hidden_batch
                        )
                    qf1_next_target_batch, _ = qf1_target(
                            next_observations_batch,
                            next_state_actions_batch,
                            actor_hidden_batch
                        )
                    qf2_next_target_batch = qf2_target(
                            next_observations_batch,
                            next_state_actions_batch,
                            actor_hidden_batch
                        )
                    qf1_next_target_batch = torch.min(
                            qf1_next_target_batch,
                            qf2_next_target_batch
                        )
                
                next_q_value_batch = rewards_batch.flatten() + (1 - dones_batch.flatten()) * args.gamma * (qf1_next_target_batch).view(-1)

            if prior_state_models:

                qf1_a_values_batch = qf1(
                    observations_batch.to(device),
                    actions_batch_batch.to(device),
                    prior_actions_batch.to(device),
                    prior_rewards_batch.to(device),
                    actor_hidden_batch
                )[0].view(-1)
                qf2_a_values_batch = qf2(
                    observations_batch.to(device),
                    actions_batch_batch.to(device),
                    prior_actions_batch.to(device),
                    prior_rewards_batch.to(device),
                    actor_hidden_batch
                )[0].view(-1)

            else:
                qf1_a_values_batch = qf1(
                    observations_batch.to(device),
                    actions_batch_batch.to(device)
                ).view(-1)


            clip_gradients = True
            
            qf1_loss = F.mse_loss(qf1_a_values_batch, next_q_value_batch)
            if iteration % args.writer_interval == 0:
                qf1_grad = get_grad_norm(qf1)
            qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=bptt)
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=args.max_grad_norm)
            if iteration % args.writer_interval == 0:
                qf1_grad_clipped = get_grad_norm(qf1)
                writer.add_scalar("grads/qf1_grad", qf1_grad, global_step)
                writer.add_scalar("grads/qf1_grad_clipped", qf1_grad_clipped, global_step)
            # Step the critic optimizers.
            qf1_optimizer.step()

            qf2_loss = F.mse_loss(qf2_a_values_batch, next_q_value_batch)
            if iteration % args.writer_interval == 0:
                qf2_grad = get_grad_norm(qf2)
            qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=bptt)
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=args.max_grad_norm)
            if iteration % args.writer_interval == 0:
                qf2_grad_clipped = get_grad_norm(qf2)
                writer.add_scalar("grads/qf2_grad", qf2_grad, global_step)
                writer.add_scalar("grads/qf2_grad_clipped", qf2_grad_clipped, global_step)
            # Step the critic optimizers.
            qf2_optimizer.step()

            # If it a policy update step and we're past the actor training delay, update the actor.
            if (global_step > args.actor_training_delay + (args.learning_starts)) and (global_step % args.policy_frequency == 0):

                if prior_state_models:
                    # actor_loss, _ = -qf1(
                    #     observations_batch.to(device),
                    #     actor(
                    #         observations_batch.to(device),
                    #         prior_actions_batch.to(device),
                    #         prior_rewards_batch.to(device),
                    #         actor_hidden_batch
                    #     )[0],
                    #     prior_actions_batch.to(device),
                    #     prior_rewards_batch.to(device),
                    #     actor_hidden_batch
                    # )[0].mean() + args.l2_reg * torch.linalg.vector_norm(
                    #         actor(observations_batch.to(device),
                    #               prior_actions_batch.to(device),
                    #               prior_rewards_batch.to(device),
                    #               actor_hidden_batch
                    #               )[0],
                    #         2
                    #     )
                    loss_actions, _ = actor(
                            observations_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device),
                            actor_hidden_batch
                        )
                    loss_qvalues, _ = qf1(
                        observations_batch.to(device),
                        loss_actions,
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device),
                        actor_hidden_batch
                    )
                    actor_loss = -loss_qvalues.mean()
                
                else:
                    actor_loss = -qf1(
                        observations_batch.to(device),
                        actor(observations_batch.to(device))
                    ).mean() + args.l2_reg * torch.linalg.vector_norm(actor(observations_batch.to(device)), 2)

                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=bptt)

                if iteration % args.writer_interval == 0:
                    actor_grad = get_grad_norm(actor)

                if clip_gradients:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)
                
                if iteration % args.writer_interval == 0:
                    actor_grad_clipped = get_grad_norm(actor)
                    writer.add_scalar("grads/actor_grad", actor_grad, global_step)
                    writer.add_scalar("grads/actor_grad_clipped", actor_grad_clipped, global_step)

                actor_optimizer.step()

                if iteration % args.writer_interval == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Soft update the target networks.
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


            if iteration % args.writer_interval == 0:
                writer.add_scalar("losses/qf1_a_values", qf1_a_values_batch.mean().item(), global_step)
                writer.add_scalar("losses/qf2_a_values", qf2_a_values_batch.mean().item(), global_step)

                # Write the l2 distance between the two qf1 and qf2 outputs.
                writer.add_scalar("losses/qf1_qf2_l2", torch.linalg.vector_norm(qf1_a_values_batch - qf2_a_values_batch).item(), global_step)

                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)


        if iteration % args.writer_interval == 0:
            print("Step time:", (time.time() - step_time) / args.num_envs)
            writer.add_scalar("charts/step_length", (time.time() - step_time), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/step_SPS", (args.num_envs / (time.time() - step_time)), global_step)

        global_step += args.num_envs

    envs.close()
    writer.close()
    print("Done.")
