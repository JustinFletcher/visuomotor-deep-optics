from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn

import numpy as np
import random
import json
from argparse import Namespace
import pickle
import os

# Import CLI utilities and related packages.
from dataclasses import dataclass
import tyro
import uuid
import json


from pathlib import Path

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


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.0,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    qf = Model[1](envs).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()
    # note: qf is not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

def make_env(env_id, flags):

    if env_id == "optomech-v1":

        def thunk():

            env = gym.make(env_id, **vars(flags))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.action_space.seed(seed)
            return env

        return thunk

    else:

        def thunk():

            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.action_space.seed(seed)
            return env

        return thunk



# TRY NOT TO MODIFY: seeding
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = args.torch_deterministic

def rollout_optomech_policy(model_path=None,
                            env_vars_path=None,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            env_kwargs=None,
                            eval_save_path=None,
                            prelearning_sample=None,
                            dataset=False,
                            dataset_name="dataset",
                            scale_reset_interval=125):
    

    # TODO: Test support for multiple environment rollouts.

    # Either load the environment vars from a path or use the provided kwargs.
    if env_vars_path is None:
        
        if env_kwargs is None:
            raise ValueError("Provide either env_vars_path or env_kwargs.")
        else:
            print("Using args provided through the CLI")
            args = env_kwargs

    else:
        
        with open(env_vars_path, "r") as f:
            print("Loading args from file")
            args = Namespace(**json.load(f))

        # Make only the non-default kwargs overwrite args loaded from disk.
        if env_kwargs:


            for key, value in env_kwargs.items():


                if value is not None:
                    setattr(args, key, value)

    # If no eval_save_path is provided, use the directory of the model.
    if eval_save_path is None:
        if model_path is not None:
            eval_save_path = os.path.dirname(model_path)    
        else:
            eval_save_path = "./tmp"

    # Register our custom optomech environment.
    gym.envs.registration.register(
        id='optomech-v1',
        entry_point='optomech.optomech:OptomechEnv',
        max_episode_steps=args.max_episode_steps,
        # reward_threshold=flags.reward_threshold,
    )

    gym.envs.registration.register(
        id='VisualPendulum-v1',
        entry_point='visual_pendulum:VisualPendulumEnv',
        max_episode_steps=args.max_episode_steps,
    )


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

    # Setup our Evironments.
    if args.subproc_env:
        print("Initializing SubprocVectorEnv")
        envs = gym.vector.SubprocVectorEnv(
            [make_env(args.env_id, args) for i in range(args.num_envs)],
        )
    if args.async_env:
        print("Initializing AsyncVectorEnv")
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, args) for i in range(args.num_envs)],
        )
    else:
        print("Initializing SyncVectorEnv")
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args) for i in range(args.num_envs)],
        )

    # Reset the environments, and create a list to hold the returns.
    obs, _ = envs.reset()
    episodic_returns = list()

    if model_path is not None:
        # Load our actor model.
        actor = torch.load(model_path, weights_only=False, map_location=device)

    # Reset the global step counter.
    global_step = 0

    # Generate UUIDs for each environment episode.
    # env_uuids = [env.uuid for env in envs]
    env_uuid_attrs = envs.get_attr("uuid")
    env_uuids = [str(env_uuid_attr) for env_uuid_attr in env_uuid_attrs]
    episode_data = list()
    

    prior_actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])
    _, rewards, _, _, _ = envs.step(prior_actions)
    prior_rewards = rewards

    # Evaluate the policy for the specified number of episodes.
    while len(episodic_returns) < rollout_episodes:
        print("Rollout episode: ", len(episodic_returns))
        print("Global step: ", global_step)

        # Create directories for each environment.
        for env_uuid in env_uuids:

            # Create save directory if it doesn't already exist.
            episode_save_path = os.path.join(
                eval_save_path,
                env_uuid,
            )
            Path(episode_save_path).mkdir(parents=True, exist_ok=True)

            # Write the metadata for this save path. (redundant, minor impact)
            with open(os.path.join(episode_save_path,
                                'episode_metadata.json'),
                                'w') as f:

                for flag, value in args.__dict__.items():
                    envs.metadata[flag] = value

                json.dump(envs.metadata, f)

        if model_path is not None:

            # Get the actions from the actor model, adding noise if requested.
            with torch.no_grad():

                # image = torch.Tensor(obs['image']).to(device)
                # # print("rollout image shape")
                # # print(image.shape)
                # prior_action = torch.Tensor(obs['prior_action']).to(device)
                # print("rollout prior action shape")
                # print(prior_action.shape)
                if args.actor_type == "impala":
            
                    actions = actor(torch.Tensor(obs).to(device),
                                    torch.Tensor(prior_actions).to(device),
                                    torch.Tensor(prior_rewards).unsqueeze(0).to(device))
                else:
                    actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0,
                                        actor.action_scale * exploration_noise)
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low,
                    envs.single_action_space.high)
        else:

            if prelearning_sample == "scales":

                pre_learn_iters = int(args.learning_starts / args.num_envs)
                num_scale_samples = pre_learn_iters // 20
                # scale_reset_interval =  int(pre_learn_iters / num_scale_samples)
                # scale_reset_interval =  int(args.max_episode_steps / args.num_envs)
                if (global_step % scale_reset_interval) == 0:
                # if global_step % args.max_episode_steps == 0:
                    print("Resetting scales.")

                    scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                    action_std = np.random.choice(scales)

                actions = np.array([(sample_normal_action(envs.single_action_space, std_dev=action_std)) for _ in range(envs.num_envs)])
            
            elif prelearning_sample == "normal":
                
                actions = np.array([(sample_normal_action(envs.single_action_space)) for _ in range(envs.num_envs)])
            
            elif prelearning_sample == "zeros":
                
                actions = np.array([(envs.single_action_space.sample() * 0.0) for _ in range(envs.num_envs)])

            else:
                
                actions = np.array([(envs.single_action_space.sample()) for _ in range(envs.num_envs)])

        # Step the environment forward.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        prior_actions = actions
        prior_rewards = rewards

        # If saving dataset, add the actions and rewards to the dataset file.
        if dataset:

            for i, (action, reward) in enumerate(zip(actions, rewards)):
                episode_data += [[action.tolist(), reward]]

        # Handle special saving for optomech environments.
        if (args.env_id == "optomech-v1") and args.write_env_state_info:

            # zip over each environment and save the state information.
            for i, (action,
                    next_ob,
                    reward,
                    termination,
                    truncation,
                    info) in enumerate(zip(actions,
                                            next_obs,
                                            rewards,
                                            terminations,
                                            truncations,
                                            [infos])):

                info["step_index"] = global_step
                info["reward"] = reward
                info["terminated"] = termination
                info["truncated"] = truncation
                info["action"] = action
                info["observation"] = next_ob

                episode_save_path = os.path.join(
                            eval_save_path,
                            env_uuids[i],
                        )

                # Save the info dictionary.
                path = os.path.join(
                    episode_save_path,
                    'step_' + str(global_step) + '.pkl'
                )
                with open(path, 'wb') as f:
                    pickle.dump(info, f)

        # Record the episodic returns.
    

        if "final_info" in infos:
            # if infos:
            # print(infos)

            for info in infos["final_info"]:
                # if "final_info" not in info:
                #     continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")

                # Create or open the dataset json.
                dataset_path = os.path.join(
                    episode_save_path,
                    dataset_name + "_" + str(len(episodic_returns)) + ".json")
                Path(eval_save_path).mkdir(parents=True, exist_ok=True)

                # Save the dataset back to the file.
                with open(dataset_path, "w") as f:
                    json.dump(episode_data, f)
                # print("/n/n/n/n")
                # print(info)
                episodic_returns += [info["episode"]["r"]]

                episode_data = list()
                # Generate new UUIDs for the next environments.
                env_uuids = [str(uuid.uuid4()) for _ in range(args.num_envs)]


        # Update the observations for the next step.
        obs = next_obs

        # Increment the global step.
        global_step += 1

    return episodic_returns
    


@dataclass
class Args:
    """
    The arguments for the rollout script.
    """

    # Rollout arguements.
    model_path: str = None
    """The path to the model."""
    env_vars_path: str = None
    """The path to the environment variables JSON."""
    num_episodes: int = 1
    """The number of episodes to run."""
    exploration_noise: float = 0.0
    """The amount of exploration noise to add."""
    dataset: bool = False
    """Toggle to enable dataset saving."""
    prelearning_sample: str = None
    """The type of prelearning sample to use."""
    eval_save_path: str = None  
    """The path to save the evaluation data."""
    dataset_name: str = "dataset"
    """The name of the dataset file."""
    scale_reset_interval: int = 125
    """The interval to reset the action scale."""
    
    # TODO: Replace with an import and manually set each to None.
    # Environment arguments.
    seed: int = -- 88
    """the name of this experiment"""
    report_time: bool = False
    """Whether to report time statistics."""
    action_type: str = None
    """The type of action to use."""
    object_type: str = None
    """The type of object to use."""
    aperture_type: str = None
    """The type of aperture to use."""
    max_episode_steps: int = None
    """The type of aperture to use."""
    ao_loop_active: bool = None
    """The maximum number of steps per episode."""
    num_atmosphere_layers: int = None
    """The number of atmosphere layers."""
    reward_threshold: float = None
    """The reward threshold to reach."""
    optomech: str = None
    """The version of optomech to use."""
    reward_function: str = None
    """The reward function to use."""
    render_frequency: int = None
    """The frequency of rendering."""
    ao_interval_ms: float = None
    """The interval between AO updates."""
    control_interval_ms: float = None
    """The interval between control updates."""
    init_differential_motion: bool = None
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = None
    """Whether to simulate differential motion."""
    frame_interval_ms: float = None
    """The interval between frames."""
    decision_interval_ms: float = None
    """The interval between decisions."""
    focal_plane_image_size_pixels: int = None
    """The size of the focal plane image in pixels."""
    record_env_state_info: bool = None
    """Whether to record environment state information."""
    write_env_state_info: bool = None
    """Whether to write environment state information."""
    state_info_save_dir: str = None
    """The directory to save state information."""
    randomize_dm: bool = None
    """Whether to randomize the DM."""
    extended_object_image_file: str = None
    """The file for the extended object image."""
    extended_object_distance: str = None
    """The distance to the extended object."""
    extended_object_extent: str = None
    """The extent of the extended object."""
    num_tensioners: int = None
    """The number of tensioners."""
    model_wind_diff_motion: bool = None
    """Whether to model wind differential motion."""
    model_gravity_diff_motion: bool = None
    """Whether to model gravity differential motion."""
    model_temp_diff_motion: bool = None
    """Whether to model temperature differential motion."""
    command_tensioners: bool = None
    """Toggle to enable agent control of tensioners."""
    command_secondaries: bool = None
    """Toggle to enable agent control of tensioners."""
    command_tip_tilt: bool = False
    """Toggle to enable agent control of tip/tilt for large mirrors."""
    incremental_control: bool = False
    """Toggle to enable incremental control."""
    command_dm: bool = None
    """ The type of observation to model 'image_only' or 'image_action'."""
    observation_mode: bool = "image_only"
    """Toggle to enable agent control of tensioners."""
    async_env: bool = None
    """Whether to use an AsynchronousVectorEnv"""
    subproc_env: bool = None
    """Whether to use a SubprocVectorEnv"""
    num_envs: int = 1
    """The number of environments to create."""
def cli_main(args): 

    rollout_optomech_policy(model_path=args.model_path,
                            env_vars_path=args.env_vars_path,
                            rollout_episodes=args.num_episodes,
                            exploration_noise=args.exploration_noise,
                            env_kwargs=vars(args),
                            prelearning_sample=args.prelearning_sample,
                            eval_save_path=args.eval_save_path,
                            dataset=args.dataset,
                            dataset_name=args.dataset_name,
                            scale_reset_interval=args.scale_reset_interval)

    return


if __name__ == "__main__":


    args = tyro.cli(Args)
    cli_main(args)

