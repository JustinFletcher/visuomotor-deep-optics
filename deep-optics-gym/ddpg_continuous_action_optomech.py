# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
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

from pathlib import Path
from anytree import Node, RenderTree

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 64
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Get the observation space shape from the environment.
        obs_shape = envs.single_observation_space.shape
        vector_action_size = envs.single_action_space.shape[0]

        model_scale = 8
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

        model_scale = 8
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

        self.fc1 = layer_init(nn.Linear(output_dim, 256))
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            # "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
            "action_scale", torch.tensor(0.01, dtype=torch.float32)
        )
        self.register_buffer(
            # "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
            "action_bias", torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
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
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

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


    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):


        step_time = time.time()
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([(envs.single_action_space.sample() * 0.0) for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break


        # Added for optomech.

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
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 4 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/l2_action", np.mean((actions)**2), global_step)
                writer.add_scalar("losses/mean_reward", np.mean(rewards), global_step)
                writer.add_scalar("losses/std_action", np.std(actions), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("Step time:", time.time() - step_time)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/SPS_float", (time.time() - step_time), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
