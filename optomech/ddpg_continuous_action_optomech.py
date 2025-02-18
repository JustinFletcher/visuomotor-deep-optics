# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
from dataclasses import dataclass

import math

import gymnasium as gym
# from gymnasium.envs import box2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from anytree import Node, RenderTree

import matplotlib.pyplot as plt

from pathlib import Path


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 88
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
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

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
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

    # Actor model parameters
    actor_channel_scale: int = 16
    """The scale of the actor model channels."""
    actor_fc_scale: int = 64
    """The scale of the actor model fully connected layers."""
    low_dim_actor: bool = False
    """Whether the actor model is visual."""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # QNetwork model parameters
    qnetwork_channel_scale: int = 16
    """The scale of the QNetwork model channels."""
    qnetwork_fc_scale: int = 64
    """The scale of the QNetwork model fully connected layers."""
    low_dim_qnetwork: bool = False
    """Whether the qnetwork model is visual."""


    # Custom Algorthim Arguments
    """Which prelearning sample strategy to use (e.g., 'scales', 'normal')"""
    prelearning_sample: str = ""

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




class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = self.logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

    def logsumexp_2d(tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


class VanillaCritic(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True):
        super().__init__()

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
                nn.MaxPool2d(4),
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
            x_o = F.tanh(self.o_conv(o))
            x = torch.cat([x_o, a], 1)

        else: 
            x = a 
        x = F.relu(self.merge_fc1(x))
        x = F.relu(self.merge_fc2(x))
        x = F.relu(self.merge_fc3(x))
        q_vals = self.fc_q(x)

        return q_vals


class VanillaActor(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True):
        super().__init__()
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
                nn.MaxPool2d(4),
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
        self.fc2 = uniform_init(nn.Linear(fc_scale // 2, fc_scale // 4),
                                lower_bound=-1/np.sqrt(fc_scale // 2),
                                upper_bound=1/np.sqrt(fc_scale // 2))
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
            upper_bound=1/np.sqrt(fc_scale // 4))
                                
        # action rescaling
        self.register_buffer(
            # "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
            "action_scale", torch.tensor(1.0, dtype=torch.float32)
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
            x = F.tanh(self.conv(x))
        else:
            x = self.ones_output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # x = F.tanh(self.fc1(x))

        a = (x * self.action_scale + self.action_bias)
        return a


class CustomActor(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True):
        
        super().__init__()
        # Initialize the shape parameters

        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape
        print(obs_shape)
        die

        vector_action_size = envs.single_action_space.shape[0]
        # Seperate out the prior action and the image
        
        a_prior_shape
        obs_image_shape

        # Check if this is a channels-last environment
        if obs_image_shape[-1] < obs_image_shape[0]:
            self.channels_last = True
            input_channels = obs_image_shape[-1]
        else:
            self.channels_last = False
            input_channels = obs_image_shape[0]

        # self.visual = not(low_dim)

        # with torch.inference_mode():

        #     # Handle channels-last environments.
        #     x = torch.zeros(1, *obs_shape)
        #     if self.channels_last:
        #         x = x.permute(0, 3, 1, 2)
        #     output_dim = self.conv(x).shape[1]

        # self.ones_output = torch.ones(1, output_dim)

        self.visual_encoder = nn.Sequential(
            conv_init(
                nn.Conv2d(input_channels, 
                            64,
                            kernel_size=7,
                            stride=2)),
            nn.ReLU(),
            conv_init(nn.Conv2d(64,
                                32,
                                kernel_size=5,
                                stride=1)),
            nn.ReLU(),
            conv_init(nn.Conv2d(32,
                                16,
                                kernel_size=5,
                                stride=1)),
            nn.Flatten(),
        )


                                
        # action rescaling
        self.register_buffer(
            # "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
            "action_scale", torch.tensor(1.0, dtype=torch.float32)
        )
        self.register_buffer(
            # "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
            "action_bias", torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, x):

        o = x[0]
        a_prior = x[1]

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        if self.visual:
            x_o = F.tanh(self.conv(o))
        else:
            x = self.ones_output

        # Extract visual feature maps

        # Extract action features maps by deconvolution

        # Concatinate vision and action features

        # Merge the vision and action feature maps

        # Apply convolulational LSTM

        # Apply attention

        # Apply action prediciton head


        a = (x * self.action_scale + self.action_bias)
        return a


    
def uniform_init(layer, lower_bound=-1e-4, upper_bound=1e-4):

    # init this layer to have weights and biases drawn uniformly from bounds.
    nn.init.uniform_(layer.weight, a=lower_bound, b=upper_bound)
    nn.init.uniform_(layer.bias, a=lower_bound, b=upper_bound)
    return layer

def conv_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def display_observation_action(observation, action, num_frames=4):
    """
    Displays the observation frames side by side and prints the action values.
    
    Parameters:
        observation (np.array or torch.Tensor): The current observation, assumed to be shaped as (channels, height, width).
        action (np.array or torch.Tensor): The action taken by the agent.
        num_frames (int): The number of stacked frames in the observation.
    """
    # Move tensors to CPU if needed and convert to numpy
    if isinstance(observation, torch.Tensor):
        observation = observation.cpu().numpy()
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    # Ensure observation is in the correct format for visualization (channels-last)
    # if observation.shape[0] == 3:  # Assuming (C, H, W) format
    #     observation = observation.transpose(1, 2, 0)

    # Set up the figure with subplots for each frame in the observation
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2, 2))
    
    # Ensure axes is always iterable, even if there's only one frame
    if num_frames == 1:
        axes = [axes]
    
    # Show each frame
    for i, ax in enumerate(axes):
        ax.imshow(observation[:, :, i], cmap='gray')
        ax.axis("off")
    
    # Set a title with action values for the first frame
    axes[0].set_title(f"Action: {action}")
    plt.show()

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


if __name__ == "__main__":

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
        max_episode_steps=200,
    )

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    

    args = tyro.cli(Args)


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Check if MPS is available
    if torch.cuda.is_available():
        print("Running with CUDA")
        device = torch.device("cuda")
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
    

    actor_type = "vanilla"

    if actor_type == "vanilla":
    
        # actor = Actor(envs).to(device)
        actor = VanillaActor(envs,
                    channel_scale=args.actor_channel_scale,
                    fc_scale=args.actor_fc_scale,
                    low_dim=args.low_dim_actor).to(device)
        
        # target_actor = Actor(envs).to(device)
        target_actor = VanillaActor(envs,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                            low_dim=args.low_dim_actor).to(device)
        
    elif actor_type == "custom":
        actor = CustomActor(envs).to(device)
        target_actor = CustomActor(envs).to(device)
    else:

        raise ValueError("Invalid actor type specified.")
    


    critic_type = "vanilla"
    if critic_type == "vanilla":
        # qf1 = QNetwork(envs).to(device)
        qf1 = VanillaCritic(envs,
                    channel_scale=args.qnetwork_channel_scale,
                    fc_scale=args.qnetwork_fc_scale,
                    low_dim=args.low_dim_qnetwork).to(device)
        
        # qf1_target = QNetwork(envs).to(device)
        qf1_target = VanillaCritic(envs,
                            channel_scale=args.qnetwork_channel_scale,
                            fc_scale=args.qnetwork_fc_scale,
                            low_dim=args.low_dim_qnetwork).to(device)
    elif critic_type == "custom":

        qf1 = CustomCritic(envs).to(device)
        qf1_target = CustomCritic(envs).to(device)

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

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs
    )


    # TODO: Add a dud check here.

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    print("Resetting Environments.")
    obs, _ = envs.reset(seed=args.seed)
    print("Environments Reset.")
    global_step = 0
    for iteration in range(args.total_timesteps):

        print("Iteration: ", iteration)
        print("global_step: ", global_step)

        if args.save_model:

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

                from rollout import rollout_optomech_policy
                eval_save_path = f"runs/{run_name}/eval_{args.exp_name}_{str(iteration)}"
                episodic_returns = rollout_optomech_policy(
                    model_path,
                    env_vars_path=args_store_path,
                    rollout_episodes=1,
                    exploration_noise=args.exploration_noise,
                    eval_save_path=eval_save_path,
                )

                # for idx, episodic_return in enumerate(episodic_returns):
                    # print(f"Episodic return: {episodic_return}")

                writer.add_scalar("eval/episodic_return", np.mean(episodic_returns), iteration)
                print("Model evaluated.")
                
                # if args.upload_model:
                #     from cleanrl_utils.huggingface import push_to_hub
                #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                #     push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")
        
        step_time = time.time()
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:

            if args.prelearning_sample == "scales":

                pre_learn_iters = int(args.learning_starts / args.num_envs)
                num_scale_samples = pre_learn_iters // 20
                scale_reset_interval =  int(pre_learn_iters / num_scale_samples)
                # scale_reset_interval =  int(args.max_episode_steps / args.num_envs)
                if (iteration % 125) == 0:
                # if global_step % args.max_episode_steps == 0:
                    print("Resetting scales.")

                    scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                    action_std = np.random.choice(scales)

                actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
            
            elif args.prelearning_sample == "normal":
                
                actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
            
            else:
                
                actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])

        else:
            with torch.no_grad():

                decay_noise = True
                if decay_noise:
                    decay = 1.0 / (1.0 + (args.decay_rate * (global_step - args.learning_starts)))
                else:
                    decay = 1.0

                
                actions = actor(torch.Tensor(obs).to(device))
                # actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                noise = torch.normal(
                    0.0,
                    actor.action_scale.cpu() * args.exploration_noise * decay,
                    ).to(device)
                actions += noise
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

                # actions = actor(torch.Tensor(obs).to(device))
                # actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                # actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
         # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            print(infos)
            for info in infos["final_info"]:

                print(info)
                print(f"\n\nglobal_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)


            # if global_step % 1 == 0:
            #     display_observation_action(data.observations[0], data.actions[0], num_frames=1)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if iteration % args.policy_frequency == 0:

                action_reg = 0.001
                actor_loss = -qf1(data.observations, actor(data.observations)
                    ).mean() + (action_reg * (actor(data.observations)**2)
                        ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if iteration % 1 == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("action/mean", actions.mean().item(), global_step)
                    writer.add_scalar("action/std", actions.std().item(), global_step)
                    # writer.add_scalar("actions/l2",actions.mean().item(), global_step)
                    writer.add_scalar("reward/", rewards.mean().item(), global_step)
                    writer.add_scalar("decay/", decay, global_step)
                    writer.add_scalar("reward_std/", rewards.std().item(), global_step)

            gradient_log_interval = 256
            if iteration % gradient_log_interval == 0:

                log_gradients_in_model(actor, writer, global_step)
                log_gradients_in_model(qf1, writer, global_step)
                log_weights_in_model(actor, writer, global_step)
                log_weights_in_model(qf1, writer, global_step)


        print("Step time:", (time.time() - step_time) / args.num_envs)
        writer.add_scalar("charts/step_length", (time.time() - step_time), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/step_SPS", (args.num_envs / (time.time() - step_time)), global_step)

        global_step += args.num_envs

    envs.close()
    writer.close()
