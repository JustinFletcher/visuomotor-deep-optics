# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
import shutil
from dataclasses import dataclass

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
    total_timesteps: int = 10_000_000
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



class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

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
    
class CBAMCritic(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True):
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

class CBAMActor(nn.Module):

    def __init__(self, envs, channel_scale=16, fc_scale=8, low_dim=True):
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

        a = (x * self.action_scale + self.action_bias)
        return a

class RepeatVector(nn.Module):
    """
    A simple custom layer that takes an input of shape [batch_size, C]
    (or [batch_size, C, 1, 1]) and repeats it in H and W dimensions
    to produce [batch_size, C, H, W].
    """
    def __init__(self, out_h=117, out_w=117):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, x):
        # Assuming x has shape [batch_size, C]
        # 1) Unsqueeze twice -> shape [batch_size, C, 1, 1]
        x = x.unsqueeze(-1).unsqueeze(-1)
        
        # 2) Repeat along the last two dims -> shape [batch_size, C, H, W]
        x = x.repeat(1, 1, self.out_h, self.out_w)
        return x
    
class ZeroPad1dToLength(nn.Module):
    """
    A custom layer that takes a 2D tensor [batch_size, length]
    and pads it (on the right) to a specified 'target_length' with zeros.
    """
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        # x has shape: [batch_size, length]

        current_length = x.shape[-1]
        
        if current_length > self.target_length:
            raise ValueError(
                f"Input length ({current_length}) is greater than "
                f"the target length ({self.target_length})."
            )
        
        pad_size = self.target_length - current_length
        
        # F.pad takes a tuple (pad_left, pad_right) for 1D padding
        # We pad on the right (second value in tuple).
        x_padded = F.pad(x, (0, pad_size), mode='constant', value=0.0)
        # Now x_padded has shape: [batch_size, target_length]
        
        return x_padded

class CustomActor(nn.Module):

    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.device = device
        self.use_lstm = True
        self.lstm_hidden_dim = lstm_hidden_dim
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
                          16,
                          kernel_size=8,
                          stride=4)),
            nn.ReLU(),
            conv_init(
                nn.Conv2d(16,
                          32,
                          kernel_size=4,
                          stride=2)),
            nn.ReLU(),
            nn.Flatten()
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
            uniform_init(
                nn.Linear(
                    int(np.prod(visual_output_shape[1:])),
                    mlp_output_size),
                lower_bound=-1/np.sqrt(np.prod(visual_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(visual_output_shape[1:]))
            ),
            nn.ReLU(),
        )

        # Define the LSTM
        self.lstm = nn.LSTM(
            input_size=mlp_output_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        # Get the output shape of the LSTM
        with torch.inference_mode():
            x = torch.zeros(1, mlp_output_size)
            pre_head_output_shape = self.lstm(x)[0].shape

        # Build the action head following the convolutional LSTM
        small_initial_action = True
        if small_initial_action: 
            self.action_head = nn.Sequential(
                uniform_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        int(np.prod(envs.single_action_space.shape))
                            ),
                    lower_bound=-1e-4,
                    upper_bound=1e-4
                ),
                nn.Tanh()
            )
        else:
            self.action_head = nn.Sequential(
                uniform_init(
                    nn.Linear(
                        int(np.prod(pre_head_output_shape[1:])),
                        int(np.prod(envs.single_action_space.shape))
                            ),
                    lower_bound=-1/np.sqrt(np.prod(pre_head_output_shape[1:])),
                    upper_bound=1/np.sqrt(np.prod(pre_head_output_shape[1:]))
                ),
                nn.Tanh()
            )
                                
        # action rescaling
        self.register_buffer(
            # "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
            # "action_scale", torch.tensor(0.01, dtype=torch.float32)
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

    def forward(self, o):

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        x_o = self.visual_encoder(o)
        x = self.mlp(x_o)
        # TODO: add a_prior, r_prior here
        # x, new_hidden = self.lstm(torch.cat([x, a], 1), self.hidden)
        x, new_hidden = self.lstm(x, self.hidden)

        # new_hidden is a tuple (h, c) after processing x
        # detached_hidden = new_hidden[0].detach().to(self.device), new_hidden[1].detach().to(self.device)
        detached_hidden = new_hidden[0], new_hidden[1]
        self.hidden = detached_hidden

        # Apply action prediciton head and activation function
        a = self.action_head(x)

        a = (a * self.action_scale + self.action_bias)
        return a

class CustomCritic(nn.Module):
    
    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True,
                                   bptt=False):
        
        super().__init__()
        # Initialize the shape parameters

        self.use_lstm = True
        self.device = device

        self.lstm_hidden_dim = lstm_hidden_dim
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
                          16,
                          kernel_size=8,
                          stride=4)
                          ),
            nn.ReLU(),
            conv_init(
                nn.Conv2d(16,
                          32,
                          kernel_size=4,
                          stride=2)
                          ),
            nn.ReLU(),
            nn.Flatten()
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
            uniform_init(
                nn.Linear(
                    int(np.prod(visual_output_shape[1:])),
                    mlp_output_size),
                lower_bound=-1/np.sqrt(np.prod(visual_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(visual_output_shape[1:]))
            ),
            nn.ReLU(),
        )

        # Define the LSTM
        self.lstm = nn.LSTM(
            input_size=mlp_output_size + vector_action_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        # Get the output shape of the LSTM
        with torch.inference_mode():
            x = torch.zeros(1, mlp_output_size + vector_action_size)
            pre_head_output_shape = self.lstm(x)[0].shape

        # Build the q head following the convolutional LSTM
        self.q_head = nn.Sequential(
            uniform_init(
                nn.Linear(int(np.prod(pre_head_output_shape[1:])), 1),
                lower_bound=-1/np.sqrt(np.prod(pre_head_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(pre_head_output_shape[1:]))
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


    def forward(self, o, a):
    

        # Handle channels-last environments.
        if self.channels_last:
            o = o.permute(0, 3, 1, 2)

        # Extract visual feature maps
        x_o = self.visual_encoder(o)
        x = self.mlp(x_o)
        # TODO: add a_prior, r_prior here
        x, new_hidden = self.lstm(torch.cat([x, 100 * a], 1), self.hidden)

        # new_hidden is a tuple (h, c) after processing x
        detached_hidden = new_hidden[0].detach().to(self.device), new_hidden[1].detach().to(self.device)
        self.hidden = detached_hidden

        # Apply action prediciton head and activation function
        q = self.q_head(x)
        return q
    


class ImpalaActor(nn.Module):

    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.device = device
        self.use_lstm = True
        self.bptt = bptt
        self.lstm_hidden_dim = lstm_hidden_dim
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
                          16,
                          kernel_size=8,
                          stride=4)),
            nn.ReLU(),
            conv_init(
                nn.Conv2d(16,
                          32,
                          kernel_size=4,
                          stride=2)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Get the output shape of the visual encoder
        with torch.inference_mode():

            # Handle channels-last environments.
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            visual_output_shape = self.visual_encoder(x).shape
    
        mlp_output_size = 256
        self.mlp = nn.Sequential(
            nn.Flatten(),
            uniform_init(
                nn.Linear(
                    int(np.prod(visual_output_shape[1:])),
                    mlp_output_size),
                lower_bound=-1/np.sqrt(np.prod(visual_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(visual_output_shape[1:]))
            ),
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

        # Build the action head following the convolutional LSTM
        self.action_head = nn.Sequential(
            uniform_init(
                nn.Linear(
                    int(np.prod(pre_head_output_shape[1:])),
                    int(np.prod(envs.single_action_space.shape))
                        ),
                lower_bound=-1/np.sqrt(np.prod(pre_head_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(pre_head_output_shape[1:]))
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

class ImpalaCritic(nn.Module):
    
    def __init__(self, envs, device, lstm_hidden_dim=256, lstm_num_layers=1, channel_scale=16, fc_scale=8, low_dim=True, bptt=False, action_scale=1.0):
        
        super().__init__()
        # Initialize the shape parameters

        self.use_lstm = True
        self.device = device
        self.bptt = bptt

        self.lstm_hidden_dim = lstm_hidden_dim
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
                          16,
                          kernel_size=8,
                          stride=4)
                          ),
            nn.ReLU(),
            conv_init(
                nn.Conv2d(16,
                          32,
                          kernel_size=4,
                          stride=2)
                          ),
            nn.ReLU(),
            nn.Flatten()
        )

        # Get the output shape of the visual encoder
        with torch.inference_mode():

            # Handle channels-last environments.
            x = torch.zeros(1, *obs_shape)
            if self.channels_last:
                x = x.permute(0, 3, 1, 2)
            visual_output_shape = self.visual_encoder(x).shape


        mlp_output_size = 256
        self.mlp = nn.Sequential(
            nn.Flatten(),
            uniform_init(
                nn.Linear(
                    int(np.prod(visual_output_shape[1:])),
                    mlp_output_size),
                lower_bound=-1/np.sqrt(np.prod(visual_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(visual_output_shape[1:]))
            ),
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
            uniform_init(
                nn.Linear(int(np.prod(pre_head_output_shape[1:])), 1),
                lower_bound=-1/np.sqrt(np.prod(pre_head_output_shape[1:])),
                upper_bound=1/np.sqrt(np.prod(pre_head_output_shape[1:]))
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

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

    if args.actor_type == "impala":
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
        
    elif args.actor_type == "custom":
        actor = CustomActor(envs,
                            device,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                            bptt=bptt,
                           action_scale=args.action_scale).to(device)
        target_actor = CustomActor(envs,
                                   device,
                                   channel_scale=args.actor_channel_scale,
                                   fc_scale=args.actor_fc_scale,
                                   bptt=bptt,
                           action_scale=args.action_scale).to(device)

    elif args.actor_type == "impala":

        actor = ImpalaActor(envs,
                            device,
                            channel_scale=args.actor_channel_scale,
                            fc_scale=args.actor_fc_scale,
                           action_scale=args.action_scale,).to(device)
        target_actor = ImpalaActor(envs,
                                   device,
                                   channel_scale=args.actor_channel_scale,
                                   fc_scale=args.actor_fc_scale,
                           action_scale=args.action_scale).to(device)


    else:

        raise ValueError("Invalid actor type specified.")
    

    if args.critic_type == "vanilla":
        # qf1 = QNetwork(envs).to(device)
        qf1 = VanillaCritic(envs,
                    channel_scale=args.qnetwork_channel_scale,
                    fc_scale=args.qnetwork_fc_scale,
                    low_dim=args.low_dim_qnetwork,
                           action_scale=args.action_scale).to(device)
        
        # qf1_target = QNetwork(envs).to(device)
        qf1_target = VanillaCritic(envs,
                            channel_scale=args.qnetwork_channel_scale,
                            fc_scale=args.qnetwork_fc_scale,
                            low_dim=args.low_dim_qnetwork,
                           action_scale=args.action_scale).to(device)

    elif args.critic_type == "custom":

        qf1 = CustomCritic(envs,
                           device,
                           channel_scale=args.actor_channel_scale,
                           fc_scale=args.actor_fc_scale,
                                   bptt=bptt).to(device)
        qf1_target = CustomCritic(envs,
                                  device,
                                  channel_scale=args.actor_channel_scale,
                                  fc_scale=args.actor_fc_scale,
                                   bptt=bptt,
                           action_scale=args.action_scale).to(device)

    elif args.critic_type == "impala":
        qf1 = ImpalaCritic(envs,
                           device,
                           channel_scale=args.actor_channel_scale,
                           fc_scale=args.actor_fc_scale,
                           action_scale=args.action_scale).to(device)
        qf1_target = ImpalaCritic(envs,
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

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    envs.single_observation_space.dtype = np.float32

    if bptt or prior_state_models:

        rb = ReplayBufferLSTM2(args.buffer_size)

    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
            n_envs=args.num_envs
        )

    # replay_buffer = ReplayBuffer()
    
    # TensorDictReplayBuffer(
    #     storage=LazyTensorStorage(args.buffer_size),
    #     batch_size=args.batch_size
    #     )


    # TODO: Add a dud check here.

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    print("Resetting Environments.")
    obs, info = envs.reset(seed=args.seed)

    print("Environments Reset.")

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
    ini_hidden_in = None
    ini_hidden_out = None

    actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])

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
                    num_rollout_episodes = 10
                    episodic_returns_list = list()
                    for i in range(num_rollout_episodes):
                        print(f"Rollout episode: {i}")
                        seed = np.random.randint(0, 999999)
                        eval_save_path = f"{rollouts_path}/"
                        env_kwargs = {"write_env_state_info": True,
                                      "record_env_state_info": True,
                                      "seed": seed}
                        
                        episodic_returns = rollout_optomech_policy(
                            model_path,
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            eval_save_path=eval_save_path,
                            env_kwargs=env_kwargs,
                        )

                        episodic_returns_list.append(episodic_returns)
                    mean_eval_episode_return = np.mean(episodic_returns_list)
                    print(f"Mean eval episodic return: {mean_eval_episode_return}")

                    writer.add_scalar("eval/best_policy_mean_returns", mean_eval_episode_return, iteration)


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
                    exploration_noise=0.0,
                    eval_save_path=eval_save_path,
                )

                # if episodic_returns > best_eval_episode_return:
                #     best_eval_episode_return = episodic_returns
                #     print(f"New best eval episodic return: {best_eval_episode_return}")

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

        prior_actions = actions.copy()

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:

            if args.prelearning_sample == "scales":

                pre_learn_iters = int(args.learning_starts / args.num_envs)
                num_scale_samples = pre_learn_iters // 20
                scale_reset_interval =  int(pre_learn_iters / num_scale_samples)
                # scale_reset_interval =  int(args.max_episode_steps / args.num_envs)
                if (iteration % 1) == 0:
                # if global_step % args.max_episode_steps == 0:
                    print("Resetting scales.")

                    scales = [0.0001, 0.001, 0.01, 0.1, 1.0]
                    action_scale = np.random.choice(scales)

                # actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
            
                actions = np.array([(action_scale * envs.single_action_space.sample()) for _ in range(envs.num_envs)])
                
            elif args.prelearning_sample == "normal":
                
                actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
            
            else:
                
                actions = np.array([(actor.action_scale.cpu() * envs.single_action_space.sample()) for _ in range(envs.num_envs)])

        else:
            with torch.no_grad():

                decay_noise = True
                if decay_noise:
                    decay = (1.0 / (1.0 + (args.decay_rate * (global_step - args.learning_starts))))
                else:
                    decay = 1.0

                if args.actor_type == "impala":

                    # print(torch.tensor(obs))
                    # print(torch.tensor(prior_actions))
                    # print(torch.tensor(prior_rewards).unsqueeze(0))
                    if torch.tensor(obs).dtype != torch.float32:

                        obs = np.array((obs / 255.0).astype(np.float32))
                    
                    actions = actor(torch.tensor(obs).to(device),
                                    torch.tensor(prior_actions).to(device), 
                                    torch.tensor(prior_rewards).unsqueeze(0).to(torch.float32).to(device))
                else:
                    actions = actor(torch.Tensor(obs).to(device))

                noise = torch.normal(
                    0.0,
                    actor.action_scale.cpu() * args.exploration_noise * decay,
                    ).to(device)
                actions += noise
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        if ini_hidden_in is None:
            ini_hidden_in = actor.hidden
            ini_hidden_out = ini_hidden_in
                
        # TRY NOT TO MODIFY: execute the game and log data.
        # prior_rewards = rewards.copy()
        prior_rewards = rewards
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = args.reward_scale * rewards

        if iteration % args.writer_interval == 0:
            writer.add_scalar("online/action_mean", actions.mean().item(), global_step)
            writer.add_scalar("online/action_std", actions.std().item(), global_step)
            # writer.add_scalar("actions/l2",actions.mean().item(), global_step)
            writer.add_scalar("online/reward_mean/", np.mean(rewards), global_step)
            writer.add_scalar("online/reward_std/", np.std(rewards), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # print(real_next_obs)
        # print(infos)
        # for idx, trunc in enumerate(truncations):
        #     print(idx)
        #     if trunc:
        #         real_next_obs = infos["final_observation"]


        if bptt:
            episode_state.append(obs)
            episode_action.append(actions)
            episode_last_action.append(prior_actions)
            episode_reward.append(rewards)
            episode_last_reward.append(prior_rewards)
            episode_next_state.append(next_obs)
            episode_done.append(terminations)  

        if prior_state_models:

            rb.push(ini_hidden_in,
                    ini_hidden_out,
                    obs,
                    actions,
                    prior_actions,
                    rewards,
                    prior_rewards,
                    real_next_obs,
                    terminations)

        else: 

            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        
        # rewards = rewards * 20
        
         # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:

        #         print(info)
        #         print(f"\n\nglobal_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

         # TRY NOT TO MODIFY: record rewards for plotting purposes
        if infos:

            for info in infos["final_info"]:

                # print(infos)
                print(f"\n\nglobal_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                print("Episode %d has ended with %d steps." % (global_step, info["episode"]["l"]))
                print("Episode %d has ended with %d reward." % (global_step, info["episode"]["r"]))

                # Add this episodes return to the list...
                train_episodic_return_list.append(info["episode"]["r"])

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


                if actor.use_lstm:
                    print("Resetting Actor Hidden State")
                    actor.reset_hidden()
                    print("Resetting Critic Hidden State")
                    qf1.reset_hidden()
                    print("Resetting Target Actor Hidden State")
                    target_actor.reset_hidden()
                    print("Resetting Target Critic Hidden State")
                    qf1_target.reset_hidden()
                    print("Agent Reset.")
                break



        # print(infos)
        # if isinstance(infos, list):
        #     for i, (info,
        #             prior_action,
        #             prior_reward) in enumerate(zip(infos,
        #                                            prior_actions,
        #                                            prior_rewards)):

        #         info["prior_action"] = prior_action
        #         info["prior_reward"] = prior_reward
        #         infos[i] = infos
        # else:

        #     infos["prior_action"] = prior_actions
        #     infos["prior_reward"] = prior_rewards
 




        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            # data = rb.sample(args.batch_size)

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

                (hidden_ins_batch,
                 hidden_outs_batch,
                 observations_batch,
                 actions_batch_batch,
                 prior_actions_batch,
                 rewards_batch,
                 prior_rewards_batch,
                 next_observations_batch,
                 dones_batch) = rb.sample(args.batch_size)
                

                hidden_ins_batch = hidden_ins_batch
                hidden_outs_batch = hidden_outs_batch
                observations_batch = torch.tensor(np.array(observations_batch), dtype=torch.float32).squeeze(1)
                actions_batch_batch = torch.tensor(np.array(actions_batch_batch), dtype=torch.float32).squeeze(1)
                prior_actions_batch = torch.tensor(np.array(prior_actions_batch), dtype=torch.float32).squeeze(1)
                rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float32).to(device)
                prior_rewards_batch = torch.tensor(np.array(prior_rewards_batch), dtype=torch.float32)
                next_observations_batch = torch.tensor(np.array(next_observations_batch), dtype=torch.float32).squeeze(1)
                dones_batch = torch.tensor(np.array(dones_batch), dtype=torch.float32).to(device)

                # print shapes
                # print("Shapes:")
                # print("observations.shape")
                # print(observations.shape)
                # print("actions_batch.shape")
                # print(actions_batch.shape)
                # print("prior_actions.shape")
                # print(prior_actions.shape)
                # print("rewards.shape")
                # print(rewards.shape)
                # print("prior_rewards.shape")
                # print(prior_rewards.shape)
                # print("next_observations.shape")
                # print(next_observations.shape)
                # print("dones")
                # print(dones)
                # die
                
            else:
                
                data = rb.sample(args.batch_size)

                observations_batch = data.observations
                actions_batch_batch = data.actions
                rewards_batch = data.rewards
                next_observations_batch = data.next_observations
                dones_batch = data.dones


            with torch.no_grad():

                if prior_state_models:

                    next_state_actions_batch = target_actor(
                        next_observations_batch.to(device),
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device)
                    )
                    qf1_next_target_batch = qf1_target(
                        next_observations_batch.to(device),
                        next_state_actions_batch.to(device),
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device)
                    )

                else:

                    next_state_actions_batch = target_actor(next_observations_batch)
                    qf1_next_target_batch = qf1_target(next_observations_batch, next_state_actions_batch)
                
                next_q_value_batch = rewards_batch.flatten() + (1 - dones_batch.flatten()) * args.gamma * (qf1_next_target_batch).view(-1)

            if prior_state_models:
                qf1_a_values_batch = qf1(
                    observations_batch.to(device),
                    actions_batch_batch.to(device),
                    prior_actions_batch.to(device),
                    prior_rewards_batch.to(device)
                ).view(-1)
            else:
                qf1_a_values_batch = qf1(
                    observations_batch.to(device),
                    actions_batch_batch.to(device)
                ).view(-1)

            qf1_loss = F.mse_loss(qf1_a_values_batch, next_q_value_batch)

            clip_gradients = True
            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=bptt)
            # qf1_loss.backward(retain_graph=True)
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:

                # l1_reg = 0.00000
                # l1_reg = 0.00000

                if prior_state_models:
                    actor_loss = -qf1(
                        observations_batch.to(device),
                        actor(
                            observations_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device)
                        ),
                        prior_actions_batch.to(device),
                        prior_rewards_batch.to(device)
                    ).mean() + args.l2_reg * torch.linalg.vector_norm(actor(observations_batch.to(device),
                            prior_actions_batch.to(device),
                            prior_rewards_batch.to(device)), 2)
                
                else:
                    actor_loss = -qf1(
                        observations_batch.to(device),
                        actor(observations_batch.to(device))
                    ).mean() + args.l2_reg * torch.linalg.vector_norm(actor(observations_batch.to(device)), 2)

                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=bptt)
                # actor_loss.backward(retain_graph=True)
                if clip_gradients:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)
                actor_optimizer.step()

                if iteration % args.writer_interval == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # TODO: Once stable, update this outward.
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


            if iteration % args.writer_interval == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values_batch.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                for i, action in enumerate(actions):
                    for j, action_element in enumerate(action):
                        action_label = f"action_{i}_{j}"
                        writer.add_scalar(action_label, action_element, global_step)
                writer.add_scalar("decay/", decay, global_step)

                



        if iteration % args.writer_interval == 0:
            print("Step time:", (time.time() - step_time) / args.num_envs)
            writer.add_scalar("charts/step_length", (time.time() - step_time), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/step_SPS", (args.num_envs / (time.time() - step_time)), global_step)

        global_step += args.num_envs

    envs.close()
    writer.close()
    print("Done.")
