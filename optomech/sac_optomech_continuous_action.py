# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
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
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    # learning_starts: int = 64
    learning_starts: int = 64
    """timestep to start learning"""
    policy_lr: float = 1e-4
    # policy_lr: float = 1e-4 # known best working
    """the learning rate of the policy network optimizer"""
    # q_lr: float = 1e-3
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    # alpha: float = .8
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

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
    max_episode_steps: int = 10000
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
    dasie_version: str = "test"
    """The version of DASIE to use."""
    reward_function: str = "ao_rms_slope"
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


def make_env(env_id, seed, idx, capture_video, run_name, flags):

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **vars(flags))
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **vars(flags))
        env = gym.wrappers.RecordEpisodeStatistics(env)
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

# Visual Soft Q Network.
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape
        vector_action_size = envs.single_action_space.shape[0]

        model_scale = 1
        self.o_conv = nn.Sequential(
                    layer_init(nn.Conv2d(obs_shape[0], model_scale, kernel_size=4, stride=2)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale, model_scale * 2, kernel_size=4, stride=4)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale * 2, model_scale * 4, kernel_size=4, stride=4)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale * 4, model_scale * 8, kernel_size=4, stride=2)),
                    nn.Flatten(),
                )

        with torch.inference_mode():
            output_dim = self.o_conv(torch.zeros(1, *obs_shape)).shape[1]

        self.o_fc1 = layer_init(nn.Linear(output_dim, model_scale * 32))

        self.a_fc1 = layer_init(nn.Linear(vector_action_size, model_scale * 32))

        self.merge_fc1 = layer_init(nn.Linear(model_scale * 64, model_scale * 32))

        self.fc_q = layer_init(nn.Linear(model_scale * 32, 1))

    def forward(self, o, a):

        # TODO: fix observation encoding to be int image.

        # x_o = F.relu(self.o_conv(o / 255.0))
        x_o = F.relu(self.o_conv(o))
        x_o = F.relu(self.o_fc1(x_o))

        
        x_a = F.relu(self.a_fc1(a))

        # Concatenate the two inputs.
        x = torch.cat([x_o, x_a], 1)

        x = F.relu(self.merge_fc1(x))

        q_vals = self.fc_q(x)
        return q_vals



LOG_STD_MAX = 3
LOG_STD_MIN = -50

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Get the observation space shape from the environment.
        obs_shape = env.single_observation_space.shape

        # Get the action space from the environment.
        # action_space = env.single_action_space
        # action_space_tree = build_tree_from_action_space(action_space)
        # vector_action_size = action_space_tree.num_leaf_nodes

        # vector_action_size = get_vector_action_size(env.single_action_space)
        vector_action_size = env.single_action_space.shape[0]


        # self.conv = nn.Sequential(
        #             layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
        #             nn.ReLU(),
        #             layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        #             nn.ReLU(),
        #             layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        #             nn.Flatten(),
        #         )

        # Idea - use fourier nn to extract spatial frequency features

        model_scale = 1
        self.conv = nn.Sequential(
                    layer_init(nn.Conv2d(obs_shape[0], model_scale, kernel_size=2, stride=2)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale, model_scale * 2, kernel_size=4, stride=4)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale * 2, model_scale * 4, kernel_size=4, stride=4)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(model_scale * 4, model_scale * 8, kernel_size=4, stride=2)),
                    nn.Flatten(),
                )

        
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, model_scale * 32))

        # Discrete action space
        # self.fc_logits = layer_init(nn.Linear(512, vector_action_size))
        self.fc_mean = layer_init(nn.Linear(model_scale * 32, vector_action_size))
        self.fc_logstd = layer_init(nn.Linear(model_scale * 32, vector_action_size))

        self.fc_logstd = layer_init(nn.Linear(model_scale * 32, vector_action_size))

        # TODO: Review later if this is needed.
        # action rescaling
        # action_space_high = env.action_space.high 
        # action_space_low = env.action_space.low
        action_space_high = 1.0
        action_space_low = -1.0
        # self.register_buffer(
        #     "action_scale", torch.tensor((action_space_high - action_space_low) / 2.0, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "action_bias", torch.tensor((action_space_high - action_space_low) / 2.0, dtype=torch.float32)
        # )
        # Emprically, an action scale of 0.001 seems to result in closure.
        self.register_buffer(
            "action_scale", torch.tensor(0.000001, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, x):
        
        conv_x = self.conv(x)
        x = F.relu(conv_x)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):

        # Compute the raw action vector.
        mean, log_std = self(x)
    
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)
        
        action = y_t * self.action_scale + self.action_bias

        torch.tensor(0.0, dtype=torch.float32)

        fixed_action = False
        if fixed_action:

            action = (action * 0.0)

        # Compute the log probability of the action for the loss.
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)



        return action, log_prob, mean


def cli_main(flags):

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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env('DASIE-v1', args.seed, 0, args.capture_video, run_name, args)]
        )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # TODO: Make the Box action space symetric and centered around 0.
    # max_action = float(envs.single_action_space.high[0])
    max_action = 1.0

    actor = Actor(envs).to(device)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape[0]).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    for flag, value in args.__dict__.items():
        envs.metadata[flag] = value

    # Create an episode UUID.
    episode_uuid = uuid.uuid4()

    if args.record_env_state_info:

        episode_save_path = os.path.join(args.state_info_save_dir,
                                            str(episode_uuid))
        
        # Create the save directory if it doesn't already exist.
        Path(episode_save_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(episode_save_path, 'episode_metadata.json'), 'w') as f:
            
            json.dump(envs.metadata, f)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        step_time = time.time()
        # ALGO LOGIC: put action logic here

        print("global_step", global_step)
        if global_step < args.learning_starts:
            actions = np.array([(envs.single_action_space.sample() * 0.0) for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        environment_save_interval = 16

        if global_step % environment_save_interval == 0:

            if args.write_env_state_info:

                if not args.record_env_state_info:
                    raise ValueError("You're trying to write, but haven't recorded, the " +
                                    "step state information. Add --record_env_state_info.")

    
                info = infos

                info["step_index"] = global_step
                info["reward"] = rewards[0]
                info["terminated"] = terminations[0]
                info["truncated"] = truncations[0]
                info["action"] = actions[0]
                info["observation"] = next_obs[0]

                # Save the info dictionary.
                with open(os.path.join(episode_save_path,
                                    'step_' + str(global_step) + '.pkl'), 'wb') as f:
                    pickle.dump(info, f)

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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

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

            if global_step % 4 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/mean_reward", np.mean(rewards), global_step)
                writer.add_scalar("losses/std_action", np.std(actions), global_step)
                writer.add_scalar("losses/l2_action", np.sqrt(np.sum((actions)**2))), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("Step time:", time.time() - step_time)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/SPS_float", (time.time() - step_time), global_step)
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
    
    # parser.add_argument('--render',
    #                     action='store_true',
    #                     default=False,
    #                     help='Render the environment.')

    # parser.add_argument('--report_time',
    #                     action='store_true',
    #                     default=False,
    #                     help='If provided, report time to run each step.')    
    
    # parser.add_argument('--action_type',
    #                     type=str,
    #                     default="none",
    #                     help='Type of action to take ("random" or "none")')
    
    # ### Gym simulation setup ###
    # parser.add_argument('--object_type',
    #                     type=str,
    #                     default="binary",
    #                     help='Type of object to simulate.')
    
    # parser.add_argument('--aperture_type',
    #                     type=str,
    #                     default="elf",
    #                     help='Type of aperture to simulate.')
    
    # parser.add_argument('--max_episode_steps',
    #                     type=int,
    #                     default=10000,
    #                     help='Steps per episode limit.')

    # parser.add_argument('--num_episodes',
    #                     type=int,
    #                     default=1,
    #                     help='Number of episodes to run.')
    
    # parser.add_argument('--num_atmosphere_layers',
    #                     type=int,
    #                     default=0,
    #                     help='Number of atmosphere layers to simulate.')

    # parser.add_argument('--reward_threshold', 
    #                     type=float,
    #                     default=25.0,
    #                     help='Max reward per episode.')
    
    # parser.add_argument('--num_steps',
    #                     type=int,
    #                     default=16,
    #                     help='Number of steps to run.')

    # parser.add_argument('--silence',
    #                     action='store_true',
    #                     default=False,
    #                     help='If provided, be quiet.')

    # parser.add_argument('--dasie_version', 
    #                     type=str,
    #                     default="test",
    #                     help='Which version of the DASIE sim do we use?')

    
    # parser.add_argument('--reward_function', 
    #                     type=str,
    #                     default="ao_rms_slope",
    #                     help='The reward function name.')


    # parser.add_argument('--render_frequency',
    #                     type=int,
    #                     default=1,
    #                     help='Render gif this frequently, in steps.')

    # parser.add_argument('--ao_interval_ms',
    #                     type=float,
    #                     default=1.0,
    #                     help='Reciprocal of AO frequency in milliseconds.')
    
    # parser.add_argument('--control_interval_ms',
    #                     type=float,
    #                     default=2.0,
    #                     help='Action control interval in milliseconds.')
    
    # # Flags for natural differntial motion.

    # # Flag for simulate_differential_motion
    # parser.add_argument('--simulate_differential_motion',
    #                     action='store_true',
    #                     default=False,
    #                     help='If provided, simulate differential motion.')

    # parser.add_argument('--frame_interval_ms',
    #                     type=float,
    #                     default=4.0,
    #                     help='Frame integration interval in milliseconds.')
    
    # parser.add_argument('--decision_interval_ms',
    #                     type=float,
    #                     default=8.0,
    #                     help='Decision (inference) interval in milliseconds.')
    
    # parser.add_argument('--focal_plane_image_size_pixels',
    #                     type=int,
    #                     default=256,
    #                     help='Size of the focal plane image in pixels.')
    
    # parser.add_argument('--render_dpi',
    #                     type=float,
    #                     default=500.0,
    #                     help='DPI of the rendered image.')

    # parser.add_argument('--record_env_state_info',
    #                     action='store_true',
    #                     default=False,
    #                     help='If provided, record the environment state info.')
    
    # parser.add_argument('--write_env_state_info',
    #                     action='store_true',
    #                     default=False,
    #                     help='If provided, write the env state info to disk.')
    
    # parser.add_argument('--state_info_save_dir',
    #                     type=str,
    #                     default="./tmp/",
    #                     help='The directory in which to write state data.')
    
    # parser.add_argument('--randomize_dm',
    #                     action='store_true',
    #                     default=False,
    #                     help='If True, randomize the DM on reset.')
    

    


    # ############################ DASIE FLAGS ##################################
    # parser.add_argument('--extended_object_image_file', type=str,
    #                     default=".\\resources\\sample_image.png",
    #                     help='Filename of image to convolve PSF with (if none, PSF returned)')

    # parser.add_argument('--extended_object_distance', type=str,
    #                     default=None,
    #                     help='Distance in meters to the extended object.')

    # parser.add_argument('--extended_object_extent', type=str,
    #                     default=None,
    #                     help='Extent in meters of the extended object image.')

    # parser.add_argument('--observation_window_size',
    #                     type=int,
    #                     default=2**1,
    #                     help='Number of frames input to the model.')
    
    # parser.add_argument('--num_tensioners',
    #                     type=int,
    #                     default=16,
    #                     help='Number of tensioners to model.')
    
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)
