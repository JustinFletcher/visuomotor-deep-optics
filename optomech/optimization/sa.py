# Standard library imports
import os
import sys
import gc
import copy
import json
import uuid
import pickle
import random
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path for local imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Third-party imports
import numpy as np
import torch
import gymnasium as gym
import tyro
from matplotlib import pyplot as plt

# Local imports
from replay_buffers import *
try:
    from chunked_dataset_manager import ChunkedDatasetManager
    CHUNKED_AVAILABLE = True
except ImportError:
    from dataset_manager import DatasetManager
    CHUNKED_AVAILABLE = False


def sample_normal_action(action_space, std_dev=0.1):
    """
    Sample an action from a normal distribution, clipped to action space bounds.

    Args:
        action_space (gymnasium.Space): The action space of the environment.
        std_dev (float): Standard deviation for the normal distribution.

    Returns:
        np.ndarray: Sampled action within action space bounds.
    """
    mean = 0
    shape = action_space.shape
    epsilon = 1e-7
    action = np.clip(
        np.random.normal(loc=mean, scale=std_dev, size=shape),
        action_space.low + epsilon,
        action_space.high - epsilon,
    )
    return action.astype(action_space.dtype)


def sample_cauchy_action(action_space):
    """
    Sample an action from a standard Cauchy distribution, clipped to action space bounds.

    Args:
        action_space (gymnasium.Space): The action space of the environment.

    Returns:
        np.ndarray: Sampled action within action space bounds.
    """
    shape = action_space.shape
    epsilon = 1e-7
    action = np.clip(
        np.random.standard_cauchy(size=shape),
        action_space.low + epsilon,
        action_space.high - epsilon,
    )
    return action.astype(action_space.dtype)


def sample_q_gaussian_action(action_space, T, q, scale=1.0):
    """
    Sample a heavy-tailed perturbation based on an approximate q-Gaussian (Student's t) distribution.

    Args:
        action_space (gymnasium.Space): The action space of the environment.
        T (float): Temperature parameter.
        q (float): Tsallis q parameter (controls tail thickness).
        scale (float): Scaling factor for perturbation.

    Returns:
        np.ndarray: Sampled perturbation.
    """
    shape = action_space.shape
    # Student's t distribution with df related to q (q = (df+3)/(df+1))
    df = 2 / (q - 1) - 1 if 1 < q < 3 else 10
    delta = np.random.standard_t(df, size=shape) * scale * np.sqrt(T)
    return delta


def make_env(env_id, flags):
    """
    Create a thunk that instantiates the specified environment.

    Args:
        env_id (str): Gymnasium environment ID.
        flags (Namespace): Arguments for environment configuration.

    Returns:
        Callable: Thunk that creates and returns the environment.
    """
    if env_id == "optomech-v1":
        def thunk():
            env = gym.make(env_id, **vars(flags))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk
    else:
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk


def calculate_perfect_action(optical_system):
    """
    Calculate the perfect action to correct optical system aberrations.
    
    Args:
        optical_system: The optical system object from the environment
        
    Returns:
        np.ndarray: Perfect action array to correct all aberrations
    """
    perfect_action_list = []
    segment_baseline_dict = copy.deepcopy(optical_system.segment_baseline_dict)
    
    for segment_id, baseline_dict in segment_baseline_dict.items():
        for key, value in baseline_dict.items():
            if key == "piston":
                piston_action_value = -1 * value / optical_system.max_piston_correction
                if piston_action_value != 0.0:
                    perfect_action_list.append(piston_action_value)
            elif key == "tip":
                tip_action_value = -1 * value / optical_system.max_tip_correction
                if tip_action_value != 0.0:
                    perfect_action_list.append(tip_action_value)
            elif key == "tilt":
                tilt_action_value = -1 * value / optical_system.max_tilt_correction
                if tilt_action_value != 0.0:
                    perfect_action_list.append(tilt_action_value)
    
    return np.array(perfect_action_list, dtype=np.float32)


@dataclass
class Args:
    """
    Command-line arguments for the simulated annealing rollout script.
    """
    # 1. Rollout and Dataset Settings
    num_processes: int = 1
    num_episodes: int = 1
    dataset: bool = False
    chunked_dataset: bool = False  # Use chunked approach for large episodes
    chunk_size: int = 10000  # Transitions per chunk for chunked mode
    eval_save_path: str = "./tmp/"
    dataset_save_path: str = None  # If None, uses eval_save_path
    dataset_name: str = "dataset"

    # 2. Environment Configuration
    env_id: str = "Hopper-v4"
    total_timesteps: int = 100_000_000
    action_type: str = "none"
    object_type: str = "single"
    aperture_type: str = "elf"
    max_episode_steps: int = 100
    num_envs: int = 1
    observation_mode: str = "image_only"
    focal_plane_image_size_pixels: int = 256
    observation_window_size: int = 2**1
    num_tensioners: int = 0
    num_atmosphere_layers: int = 0
    optomech_version: str = "test"
    reward_function: str = "strehl"
    silence: bool = False

    # 3. Control Settings
    incremental_control: bool = False
    command_tensioners: bool = False
    command_secondaries: bool = False
    command_tip_tilt: bool = False
    command_dm: bool = False
    discrete_control: bool = False
    discrete_control_steps: int = 128

    # 4. Rendering and Logging
    render: bool = False
    render_frequency: int = 1
    render_dpi: float = 500.0
    record_env_state_info: bool = False
    write_env_state_info: bool = False
    write_state_interval: int = 1
    state_info_save_dir: str = "./tmp/"
    report_time: bool = False

    # 5. Simulation Timing Parameters
    ao_loop_active: bool = False
    ao_interval_ms: float = 1.0
    control_interval_ms: float = 2.0
    frame_interval_ms: float = 4.0
    decision_interval_ms: float = 8.0

    # 6. Optomech-Specific Modeling Toggles
    init_differential_motion: bool = False
    simulate_differential_motion: bool = False
    randomize_dm: bool = False
    model_wind_diff_motion: bool = False
    model_gravity_diff_motion: bool = False
    model_temp_diff_motion: bool = False

    # 7. Hardware/Infrastructure Settings
    gpu_list: str = "0"

    # 8. Extended Object Parameters
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    extended_object_distance: str = None
    extended_object_extent: str = None

    # 9. Simulated Annealing Parameters
    init_temperature: float = 100.0
    std_dev_patience: int = 10
    sparsity_patience: int = 100000000000000
    temperature_patience: int = 1000000000000000
    init_std_dev: float = 1.0

def cli_main(args):
    """
    Command-line interface entry point for the simulated annealing rollout.
    Parses arguments and calls the main SA function.
    """
    # Print the parsed arguments
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Call the main SA function with the parsed arguments
    sa(args)

def sa(args):
    """
    Main entry point for the simulated annealing rollout.
    Registers environments, configures hardware, and runs the optimization loop.
    """
    # Register custom environments
    gym.envs.registration.register(
        id='optomech-v1',
        entry_point='optomech.optomech:OptomechEnv',
        max_episode_steps=args.max_episode_steps,
    )
    gym.envs.registration.register(
        id='VisualPendulum-v1',
        entry_point='visual_pendulum:VisualPendulumEnv',
        max_episode_steps=args.max_episode_steps,
    )

    # Select device
    if torch.cuda.is_available():
        print("Running with CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running with MSP")
        device = torch.device("mps")
    else:
        print("Running with CPU")
        device = torch.device("cpu")

    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args) for _ in range(args.num_envs)]
    )

    # envs = gym.make(args.env_id, **vars(args))
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)
    # # envs = make_env(args.env_id, args)

    # Reset environments and initialize variables
    obs, _ = envs.reset()
    episodic_returns = []
    global_step = 0
    rollout_step = 0
    env_uuid_attrs = envs.get_attr("uuid")
    env_uuids = [str(env_uuid_attr) for env_uuid_attr in env_uuid_attrs]
    episode_data = []
    prior_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    action_size = envs.single_action_space.shape[0]
    
    # Initialize dataset manager for seamless data saving
    if args.dataset:
        # Use dataset_save_path if provided, otherwise fall back to eval_save_path
        dataset_path = args.dataset_save_path if args.dataset_save_path is not None else args.eval_save_path
        
        # Choose dataset manager based on episode length and chunked setting
        if args.chunked_dataset and CHUNKED_AVAILABLE and args.max_episode_steps > 10000:
            print(f"Using chunked dataset manager (chunk_size={args.chunk_size})")
            dataset_manager = ChunkedDatasetManager(dataset_path, chunk_size=args.chunk_size)
            use_chunked = True
        else:
            print("Using standard dataset manager")
            from dataset_manager import DatasetManager
            dataset_manager = DatasetManager(dataset_path)
            use_chunked = False
            
        # Initialize episode data collectors (for non-chunked mode)
        if not use_chunked:
            episode_observations = []
            episode_next_observations = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_perfect_actions = []
            episode_best_actions = []

    obs, rewards, _, _, infos = envs.step(prior_actions)
    prior_rewards = rewards
    prior_obs = obs
    current_cost = -rewards
    actions = prior_actions
    best_reward = rewards[0]
    best_action = actions[0]
    current_command = actions
    current_infos = infos
    current_state = obs

    steps_since_acceptance = 0
    last_action_accepted = False
    temperature_step = 0
    rb = ReplayBufferWithHiddenStates(args.max_episode_steps)

    # Create output directories and write metadata for each environment
    for env_uuid in env_uuids:
        episode_save_path = os.path.join(args.eval_save_path, env_uuid)
        Path(episode_save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(episode_save_path, 'episode_metadata.json'), 'w') as f:
            for flag, value in args.__dict__.items():
                envs.metadata[flag] = value
            json.dump(envs.metadata, f)

    import time
    start_time = time.time()
    while len(episodic_returns) < args.num_episodes:

        # Simulated annealing temperature scheduling
        if steps_since_acceptance > args.temperature_patience:
            temperature_step = 1
        else:
            temperature_step += 1
        temperature = args.init_temperature / temperature_step

        # Adaptive std_dev scheduling
        if steps_since_acceptance > args.std_dev_patience:
            std_dev = args.init_std_dev * np.array([
                np.random.choice([
                    np.random.uniform(1e+2, 1e+3),
                    np.random.uniform(1e+1, 1e+2),
                    np.random.uniform(1e+1, 1e-0),
                    np.random.uniform(1e-0, 1e-1),
                    np.random.uniform(1e-1, 1e-2),
                    np.random.uniform(1e-2, 1e-3),
                    np.random.uniform(1e-3, 1e-4),
                    np.random.uniform(1e-4, 1e-5),
                    np.random.uniform(1e-5, 1e-6),
                    np.random.uniform(1e-6, 1e-7),
                    np.random.uniform(1e-7, 1e-8),
                    0.0
                ], size=1, replace=True)
                for _ in range(action_size)
            ], dtype=np.float32).flatten()
        else:
            std_dev = args.init_std_dev

        # Action perturbation strategy selection
        fsa = False  # Use Cauchy for fast simulated annealing
        gsa = False # Set True to use q-Gaussian for generalized simulated annealing

        if gsa:
            q = 1.5
            T = 1.0
            actions = np.array([
                actions[i] + sample_q_gaussian_action(envs.single_action_space, T, q, scale=std_dev)
                for i in range(envs.num_envs)
            ])
        elif fsa:
            actions = np.array([
                actions[i] + std_dev * sample_cauchy_action(envs.single_action_space)
                for i in range(envs.num_envs)
            ])
        else:
            actions = np.array([
                actions[i] + sample_normal_action(envs.single_action_space, std_dev=std_dev)
                for i in range(envs.num_envs)
            ])

        # Action sparsity scheduling
        if steps_since_acceptance > args.sparsity_patience:
            sparsity = np.random.uniform(0.0, 1.0)
        else:
            sparsity = 0.0
        sparsity_size = np.min([int(actions.shape[1] * sparsity), actions.shape[1]-1])
        zero_out = np.random.choice(actions.shape[1], size=sparsity_size, replace=False)
        actions[:, zero_out] = 0.0

        # Clip actions to action space bounds
        actions = np.clip(actions, envs.single_action_space.low, envs.single_action_space.high)

        # Step the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Update best reward and action if improved
        if rewards[0] > best_reward:
            best_reward = rewards[0]
            best_action = actions[0]

        # Compute cost and cost delta
        candidate_cost = -rewards
        cost_delta = candidate_cost - current_cost

        # ETA calculation
        elapsed = time.time() - start_time
        total_iters = args.max_episode_steps
        iter_num = rollout_step + 1
        if iter_num > 0 and total_iters > 0:
            eta = (elapsed / iter_num) * (total_iters - iter_num)
        else:
            eta = 0.0
        # Compute steps per second
        steps_per_second = iter_num / elapsed if elapsed > 0 else 0.0



        # Print progress if not silenced
        if not args.silence:
            print(f"Steps per second: {steps_per_second:.2f}")

        # Print progress with requested formatting
        # reward: 3 decimal, cost_delta: 3 decimal, temp: 3 decimal, elapsed: 1 decimal, ETA: hh:mm:ss
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        print(
            f"[{iter_num}/{total_iters}] R={best_reward:.3f} Δ={cost_delta[0]:3.3f} T={temperature:.3f} "
            f"\u23F1{elapsed:.1f} ETA={eta_str}"
        )

        # Acceptance criteria: accept if cost is reduced or with probability ~exp(-delta/T)
        accepted = False
        if (cost_delta <= 0.0) or (np.exp(-cost_delta / temperature) > random.uniform(0.0, 1.0)):
            current_cost = candidate_cost
            steps_since_acceptance = 0
            accepted = True
        else:
            actions = prior_actions
            steps_since_acceptance += 1

        # Store transition in replay buffer
        real_next_obs = next_obs
        lstm_num_layers = 1
        lstm_hidden_dim = 128
        actor_hidden = (
            torch.zeros(lstm_num_layers, lstm_hidden_dim,),
            torch.zeros(lstm_num_layers, lstm_hidden_dim,)
        )


        # If accepted, update state/action/infos for next step
        if accepted:
            current_state = next_obs
            current_command = actions
            current_infos = infos

            rb.push(
                actor_hidden,
                prior_obs,
                actions,
                prior_actions,
                rewards,
                prior_rewards,
                current_state,
                terminations,
                advantageous=True
            )
            last_action_accepted = True
            
            # Save dataset if requested - collect every accepted transition
            if args.dataset:
                # Only collect from the first environment (index 0) to avoid shape mismatches
                episode_observations.append(prior_obs[0].tolist())
                episode_next_observations.append(current_state[0].tolist())
                episode_actions.append(actions[0].tolist())
                episode_rewards.append(rewards[0])
                episode_dones.append(False)  # Will be updated to True at episode end

            # Save environment state info if required
            if (args.env_id == "optomech-v1") and args.write_env_state_info:
                if global_step % args.write_state_interval == 0:
                    for i, (action, next_ob, reward, termination, truncation, info) in enumerate(
                        zip(current_command, current_state, current_cost, terminations, truncations, [current_infos])
                    ):
                        info["step_index"] = global_step
                        info["reward"] = reward
                        info["terminated"] = termination
                        info["truncated"] = truncation
                        info["action"] = action
                        info["observation"] = next_ob
                        episode_save_path = os.path.join(args.eval_save_path, env_uuids[i])
                        path = os.path.join(episode_save_path, f'step_{global_step}.pkl')
                        with open(path, 'wb') as f:
                            pickle.dump(info, f)
        # If rejected, do not update state/action/infos
        else:
            if last_action_accepted:
                rb.push(
                    actor_hidden,
                    prior_obs,
                    actions,
                    prior_actions,
                    rewards,
                    prior_rewards,
                    current_state,
                    terminations,
                    advantageous=False
                )
                last_action_accepted = False

        # Update prior actions and rewards
            


        prior_actions = actions
        prior_rewards = rewards
        prior_obs = current_state

        # Check for episode termination and save results
        if "final_info" in infos:
            for info in infos["final_info"]:

                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                
                dataset_path = os.path.join(
                    episode_save_path,
                    f"{args.dataset_name}_{len(episodic_returns)}.json"
                )
                optical_system = envs.get_attr("optical_system")[0]
                # print(optical_system.segment_baseline_dict)
                # print(optical_system.max_piston_correction)
                # print(optical_system.max_tip_correction)
                # print(optical_system.max_tilt_correction)
                

                # iterate over each item in
                perfect_action_list = list()
                segment_baseline_dict = copy.deepcopy(optical_system.segment_baseline_dict)
                for segment_id, baseline_dict in segment_baseline_dict.items():
                    print(f"Segment {segment_id}")
                    for key, value in baseline_dict.items():
                        print(f"  {key}: {value}")
                        if key == "piston":
                            piston_action_value = -1 * value / optical_system.max_piston_correction
                            segment_baseline_dict[segment_id][key] = piston_action_value
                            if piston_action_value != 0.0:
                                perfect_action_list.append(piston_action_value)
                        elif key == "tip":
                            tip_action_value = -1 * value / optical_system.max_tip_correction
                            segment_baseline_dict[segment_id][key] = tip_action_value
                            if tip_action_value != 0.0:
                                perfect_action_list.append(tip_action_value)
                        elif key == "tilt":
                            tilt_action_value = -1 * value / optical_system.max_tilt_correction
                            segment_baseline_dict[segment_id][key] = tilt_action_value
                            if tilt_action_value != 0.0:
                                perfect_action_list.append(tilt_action_value)

                # print(best_action.tolist())
                # print(perfect_action_list)
                # print(envs.single_action_space)
                perfect_action_list = np.array(perfect_action_list, dtype=np.float32)
                debug = False
                if debug:
                    print(perfect_action_list)
                    _, rewards, _, _, _ = envs.step([perfect_action_list])
                    print(f"Perfect reward: {rewards[0]}")
                    plt.imshow(np.log(next_obs[0][0, :, :]))
                    plt.show()
                # Save episode using the new dataset manager
                if args.dataset and len(episode_observations) > 0:
                    # Mark the last step as done
                    if len(episode_dones) > 0:
                        episode_dones[-1] = True
                    
                    # Add perfect and best actions to each transition
                    episode_perfect_actions = []
                    episode_best_actions = []
                    
                    # Add the perfect action to each step
                    for i in range(len(episode_observations)):
                        episode_perfect_actions.append(perfect_action_list.tolist())
                        episode_best_actions.append(best_action.tolist())
                    
                    # Prepare episode data for dataset manager
                    episode_data_dict = {
                        'observations': np.array(episode_observations).tolist(),
                        'next_observations': np.array(episode_next_observations).tolist(),
                        'actions': np.array(episode_actions).tolist(),
                        'rewards': np.array(episode_rewards).tolist(),
                        'dones': np.array(episode_dones).tolist(),
                        'perfect_actions': episode_perfect_actions,
                        'best_actions': episode_best_actions
                    }
                    
                    # Prepare metadata (removed perfect_action and best_action from metadata)
                    metadata = {
                        'reward_function': args.reward_function,
                        'env_id': args.env_id,
                        'object_type': args.object_type,
                        'aperture_type': args.aperture_type,
                        'episode_number': len(episodic_returns),
                        'total_reward': float(info['episode']['r']),
                        'episode_length': len(episode_observations)
                    }
                    
                    # Save using dataset manager
                    episode_id = dataset_manager.save_episode(episode_data_dict, metadata)

                episodic_returns.append(info["episode"]["r"])
                
                # Reset episode data collectors for next episode
                if args.dataset:
                    episode_observations = []
                    episode_next_observations = []
                    episode_actions = []
                    episode_rewards = []
                    episode_dones = []
                
                # Generate new UUIDs for the next environments
                env_uuids = [str(uuid.uuid4()) for _ in range(args.num_envs)]
                rollout_step = 0

        obs = next_obs
        global_step += 1
        rollout_step += 1
        if global_step % 1000 == 0:

            # Write final reward to a readable text file
            with open(os.path.join(episode_save_path, "best_reward.txt"), "w") as f:
                f.write(f"Global step: {global_step}\n")
                f.write(f"Best reward: {best_reward}\n")  
                f.write(f"SPS: {steps_per_second:.2f}\n")
                f.write(f"Done: 0\n")

    # Save the replay buffer to disk
    print("Saving replay buffer to disk...")
    Path(episode_save_path).mkdir(parents=True, exist_ok=True)
    rb.save(os.path.join(episode_save_path, "replay_buffer.pt"))
    # Write final reward to a readable text file
    with open(os.path.join(episode_save_path, "best_reward.txt"), "w") as f:
        f.write(f"Global step: {global_step}\n")
        f.write(f"Best reward: {best_reward}\n")  
        f.write(f"SPS: {steps_per_second:.2f}\n")
        f.write(f"Done: 1\n")

    gc.collect()  # Clean up memory
    torch.cuda.empty_cache()
    del envs  # Clean up the environment
    return


if __name__ == "__main__":
    args = tyro.cli(Args)
    cli_main(args)
    sys.exit(0)

