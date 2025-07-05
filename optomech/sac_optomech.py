# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from anytree import Node, RenderTree

from pprint import pprint
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""


def make_env(env_id, seed, idx, capture_video, run_name, flags):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **vars(flags))
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **vars(flags))
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk



# Register our custom DASIE environment.
gym.envs.registration.register(
    id='DASIE-v1',
    entry_point='deep-optics-gym.dasie:DasieEnv',
    # max_episode_steps=4,
    # reward_threshold=flags.reward_threshold,
)


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals



def build_tree_from_action_space(action_space):

    action_node = Node("action",
                       content="")

    linear_address = 0


    # Iterate over each predictive command.
    for step_num, step in enumerate(action_space):

        step_node = Node(f"step_{step_num}",
                         parent=action_node,
                         content="")

        # Iterate over stages (e.g., secondaries, primary, DM) in the step.
        for stage_num, stage in enumerate(step):

            stage_node = Node(f"stage_{stage_num}",
                              parent=step_node,
                              content="")

            # Iterate over components (e.g., secondary, tensioner) in the stage.
            for component_num, component in enumerate(stage):

                component_node = Node(f"component_{component_num}",
                                      parent=stage_node,
                                      content="")


                if hasattr(component, '__iter__'):

                    # Iterate over commands (e.g., force, displacement) for the component.
                    for command_num, command in enumerate(component):


                        linear_address += 1

                        action_space_address= f"{step_num}_{stage_num}_{component_num}_{command_num}"
                        

                        command_node = Node(f"command_{command_num}",
                                            parent=component_node,
                                            content=command,
                                            action_space_address=action_space_address,
                                            linear_address=linear_address)

                else:

                    linear_address += 1

                    action_space_address= f"{step_num}_{stage_num}_{component_num}_{0}"

                    command_node = Node(f"command_0",
                                        parent=component_node,
                                        content=component,
                                        action_space_address=action_space_address,
                                        linear_address=linear_address)

    action_node.num_leaf_nodes = linear_address
    return action_node

def encode_action_space_from_vector(action_space, action_vector):


    # Define the 4-level nested list
    my_list = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]

    # Define the string of indices
    indices_str = "0_1_0_1"

    # Convert the string to a list of integers
    indices = list(map(int, indices_str.split('_')))

    # Use the indices to assign a new value to the corresponding element in the nested list
    sublist = my_list
    for index in indices[:-1]:
        sublist = sublist[index]
    sublist[indices[-1]] = 'new_value'

    print(my_list)  

    # for linear_address, action_element in enumerate(action_vector):

    #     # Get the nested address, given the linear address.



    #     # Define the 4-level nested list
    #     my_list = [[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]

    #     # Define the string of indices
    #     action_space_address = "0_1_0_1"

    #     # Convert the string to a list of integers
    #     indices = list(map(int, action_space_address.split('_')))

    #     # Use the indices to assign a new value to the corresponding element in the nested list
    #     sublist = my_list
    #     for index in indices[:-1]:
    #         sublist = sublist[index]
    #     sublist[indices[-1]] = 'new_value'

    # print(my_list)  # Output: [[['a', 'b'], ['c', 'new_value']], [['e', 'f'], ['g', 'h']]]



class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape

        # Get the action space from the environment.
        action_space = envs.single_action_space
        action_space_tree = build_tree_from_action_space(action_space)
        vector_action_size = action_space_tree.num_leaf_nodes

        for pre, fill, node in RenderTree(action_space_tree):

            try:
                print("%s%s, %s, %s, %s" % (pre,
                                            node.name,
                                            node.content,
                                            node.action_space_address,
                                            node.linear_address))
            except:
                print("%s%s" % (pre, node.name))


        # Create a vector of random numbers between 0 and 1 the size of the number of leaf nodes in the tree.
        vectorized_action = np.random.rand(vector_action_size)

        action = encode_action_space_from_vector(
            action_space,
            vectorized_action
        )

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        # TODO: Build heterogenous, continuous action space.

        # Discrete action space
        self.fc_logits = layer_init(nn.Linear(512, vector_action_size))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        # TODO: Make observation space an int image.
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


def cli_main(flags): 
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Build a gym environment; pass the CLI flags to the constructor as kwargs.
    envs = gym.vector.SyncVectorEnv(
        [make_env('DASIE-v1', args.seed, 0, args.capture_video, run_name, flags)]
        )

    # print(envs.single_action_space.shape)
    # print(envs.single_action_space)


    # action_space_size = len(action_space)

    # for n, action in enumerate(action_space):
    #     print(n)      

    #     print(action)

    # print(len(action_space))

    # tensioner_commands = action_space[0][1]
    # secondaries_commands = action_space[0][0]
    # print([command.shape for command in secondaries_commands])
    # print([command.shape for command in tensioner_commands])
    # print(secondaries_commands.shape)


    # env = gym.make('DASIE-v1', **vars(flags))
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)

    die

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
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
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list',
                        type=str,
                        default="0",
                        help='GPUs to use with this model.')
    
    parser.add_argument('--render',
                        action='store_true',
                        default=False,
                        help='Render the environment.')

    parser.add_argument('--report_time',
                        action='store_true',
                        default=False,
                        help='If provided, report time to run each step.')    
    
    parser.add_argument('--action_type',
                        type=str,
                        default="none",
                        help='Type of action to take ("random" or "none")')
    
    ### Gym simulation setup ###
    parser.add_argument('--object_type',
                        type=str,
                        default="binary",
                        help='Type of object to simulate.')
    
    parser.add_argument('--aperture_type',
                        type=str,
                        default="circular",
                        help='Type of aperture to simulate.')
    
    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=10000,
                        help='Steps per episode limit.')

    parser.add_argument('--num_episodes',
                        type=int,
                        default=1,
                        help='Number of episodes to run.')
    
    parser.add_argument('--num_atmosphere_layers',
                        type=int,
                        default=0,
                        help='Number of atmosphere layers to simulate.')

    parser.add_argument('--reward_threshold', 
                        type=float,
                        default=25.0,
                        help='Max reward per episode.')
    
    parser.add_argument('--num_steps',
                        type=int,
                        default=500,
                        help='Number of steps to run.')

    parser.add_argument('--silence',
                        action='store_true',
                        default=False,
                        help='If provided, be quiet.')

    parser.add_argument('--dasie_version', 
                        type=str,
                        default="test",
                        help='Which version of the DASIE sim do we use?')

    
    parser.add_argument('--reward_function', 
                        type=str,
                        default="ao_rms_slope",
                        help='The reward function name.')


    parser.add_argument('--render_frequency',
                        type=int,
                        default=1,
                        help='Render gif this frequently, in steps.')

    parser.add_argument('--ao_interval_ms',
                        type=float,
                        default=1.0,
                        help='Reciprocal of AO frequency in milliseconds.')
    
    parser.add_argument('--control_interval_ms',
                        type=float,
                        default=4.0,
                        help='Action control interval in milliseconds.')
    
    # Flags for natural differntial motion.

    # Flag for simulate_differential_motion
    parser.add_argument('--simulate_differential_motion',
                        action='store_true',
                        default=False,
                        help='If provided, simulate differential motion.')
    


    parser.add_argument('--frame_interval_ms',
                        type=float,
                        default=12.0,
                        help='Frame integration interval in milliseconds.')
    
    parser.add_argument('--decision_interval_ms',
                        type=float,
                        default=24.0,
                        help='Decision (inference) interval in milliseconds.')
    
    parser.add_argument('--focal_plane_image_size_pixels',
                        type=int,
                        default=256,
                        help='Size of the focal plane image in pixels.')
    
    parser.add_argument('--render_dpi',
                        type=float,
                        default=500.0,
                        help='DPI of the rendered image.')

    parser.add_argument('--record_env_state_info',
                        action='store_true',
                        default=False,
                        help='If provided, record the environment state info.')
    
    parser.add_argument('--write_env_state_info',
                        action='store_true',
                        default=False,
                        help='If provided, write the env state info to disk.')
    
    parser.add_argument('--state_info_save_dir',
                        type=str,
                        default="./tmp/",
                        help='The directory in which to write state data.')
    
    parser.add_argument('--randomize_dm',
                        action='store_true',
                        default=False,
                        help='If True, randomize the DM on reset.')
    

    


    ############################ DASIE FLAGS ##################################
    parser.add_argument('--extended_object_image_file', type=str,
                        default=".\\resources\\sample_image.png",
                        help='Filename of image to convolve PSF with (if none, PSF returned)')

    parser.add_argument('--extended_object_distance', type=str,
                        default=None,
                        help='Distance in meters to the extended object.')

    parser.add_argument('--extended_object_extent', type=str,
                        default=None,
                        help='Extent in meters of the extended object image.')

    parser.add_argument('--observation_window_size',
                        type=int,
                        default=2**1,
                        help='Number of frames input to the model.')
    
    parser.add_argument('--num_tensioners',
                        type=int,
                        default=16,
                        help='Number of tensioners to model.')
    
    

    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


