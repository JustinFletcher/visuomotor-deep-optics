"""
Recurrent PPO for continuous-action POMDPs.

On-policy PPO with a shared-encoder LSTM actor-critic, designed for
partially observable environments (e.g., optomech telescope control).

Key features:
- Shared CNN encoder + LSTM between policy and value function
- Sequence-chunked TBPTT with hidden-state resets at episode boundaries
- GAE-lambda advantage estimation
- Clipped surrogate objective with entropy bonus
- Compatible with existing rollout/evaluation tools via PPOActorWrapper

Usage:
    python -m optomech.rl.ppo_recurrent --config optomech/rl/ppo_optomech_config.json
    python -m optomech.rl.ppo_recurrent --env-id Hopper-v4 --num-envs 4
"""

import os
import copy
import json
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Add parent directories to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ppo_models import RecurrentActorCritic, PPOActorWrapper
from optomech.rollout import rollout_optomech_policy
from optomech.rl.td3_replay_buffer import load_pretrained_encoder


# ============================================================================
# Args
# ============================================================================


@dataclass
class Args:
    config: Optional[str] = None
    """Path to JSON config file (overrides defaults, but is overridden by command-line args)"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if using cuda, set the gpu"""
    gpu_list: int = 0
    """GPU device index"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "optomech-ppo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    num_envs: int = 4
    """number of parallel environments"""
    async_env: bool = False
    """Whether to use an AsynchronousVectorEnv"""
    subproc_env: bool = False
    """Whether to use a SubprocVectorEnv"""
    save_model: bool = True
    """whether to save model checkpoints"""
    model_save_interval: int = 50
    """interval (in updates) between periodic model saves"""
    writer_interval: int = 1
    """interval (in updates) between tensorboard writes"""
    num_eval_rollouts: int = 1
    """number of rollouts for evaluation"""

    # === PPO Algorithm ===
    env_id: str = "optomech-v1"
    """the gymnasium environment id"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 3e-4
    """learning rate for the shared optimizer"""
    num_steps: int = 128
    """number of steps per environment per rollout"""
    num_minibatches: int = 4
    """number of minibatches per update epoch"""
    update_epochs: int = 4
    """number of epochs per rollout batch"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_coef: float = 0.2
    """PPO clipping coefficient (epsilon)"""
    clip_vloss: bool = True
    """whether to clip the value function loss"""
    ent_coef: float = 0.01
    """entropy bonus coefficient"""
    vf_coef: float = 0.5
    """value function loss coefficient"""
    max_grad_norm: float = 0.5
    """maximum gradient norm for clipping"""
    target_kl: Optional[float] = None
    """target KL divergence for early stopping (None = disabled)"""
    norm_adv: bool = True
    """whether to normalize advantages per minibatch"""
    anneal_lr: bool = True
    """whether to linearly anneal the learning rate"""
    reward_scale: float = 1.0
    """multiplicative scale applied to rewards"""
    action_scale: float = 1.0
    """scale factor for actions"""

    # === Recurrence ===
    seq_len: int = 16
    """TBPTT sequence chunk length for LSTM training"""

    # === Model Architecture ===
    channel_scale: int = 16
    """CNN channel multiplier"""
    fc_scale: int = 64
    """MLP hidden dimension"""
    lstm_hidden_dim: int = 128
    """LSTM hidden dimension"""
    init_log_std: float = -0.5
    """initial value for policy log standard deviation"""

    # === Pretrained Encoder ===
    pretrained_encoder_path: Optional[str] = None
    """path to pretrained autoencoder checkpoint"""
    freeze_encoder: bool = True
    """whether to freeze the pretrained encoder"""

    # === Environment-specific arguments ===
    gpu_list: str = "0"
    """GPU device list"""
    render: bool = False
    """whether to render the environment"""
    report_time: bool = False
    """whether to report time statistics"""
    action_type: str = "none"
    """the type of action to use"""
    object_type: str = "binary"
    """the type of object to use"""
    aperture_type: str = "elf"
    """the type of aperture to use"""
    max_episode_steps: int = 300
    """maximum steps per episode"""
    discrete_control: bool = False
    """toggle to enable discrete control"""
    discrete_control_steps: int = 128
    """the number of discrete control steps"""
    incremental_control: bool = False
    """toggle to enable incremental control"""
    command_tensioners: bool = False
    """toggle to enable agent control of tensioners"""
    command_secondaries: bool = False
    """toggle to enable agent control of secondaries"""
    command_tip_tilt: bool = False
    """toggle to enable agent control of tip/tilt for large mirrors"""
    command_dm: bool = False
    """toggle to enable agent control of dm"""
    observation_mode: str = "image_only"
    """the type of observation to model 'image_only' or 'image_action'"""
    ao_loop_active: bool = False
    """whether the AO loop is active"""
    num_episodes: int = 1
    """the number of episodes to run"""
    num_atmosphere_layers: int = 0
    """the number of atmosphere layers"""
    reward_threshold: float = 25.0
    """the reward threshold to reach"""
    silence: bool = False
    """whether to silence the output"""
    optomech_version: str = "test"
    """the version of optomech to use"""
    reward_function: str = "strehl"
    """the reward function to use"""
    render_frequency: int = 1
    """the frequency of rendering"""
    ao_interval_ms: float = 1.0
    """the interval between AO updates"""
    control_interval_ms: float = 2.0
    """the interval between control updates"""
    init_differential_motion: bool = False
    """whether to initialize differential motion"""
    simulate_differential_motion: bool = False
    """whether to simulate differential motion"""
    frame_interval_ms: float = 4.0
    """the interval between frames"""
    decision_interval_ms: float = 8.0
    """the interval between decisions"""
    focal_plane_image_size_pixels: int = 256
    """the size of the focal plane image in pixels"""
    render_dpi: float = 500.0
    """the DPI for rendering"""
    record_env_state_info: bool = False
    """whether to record environment state information"""
    write_env_state_info: bool = False
    """whether to write environment state information"""
    state_info_save_dir: str = "./tmp/"
    """the directory to save state information"""
    randomize_dm: bool = False
    """whether to randomize the DM"""
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    """the file for the extended object image"""
    extended_object_distance: str = None
    """the distance to the extended object"""
    extended_object_extent: str = None
    """the extent of the extended object"""
    observation_window_size: int = 2
    """the size of the observation window"""
    num_tensioners: int = 16
    """the number of tensioners"""
    model_wind_diff_motion: bool = False
    """whether to model wind differential motion"""
    model_gravity_diff_motion: bool = False
    """whether to model gravity differential motion"""
    model_temp_diff_motion: bool = False
    """whether to model temperature differential motion"""


# ============================================================================
# Environment factory
# ============================================================================


def make_env(env_id, idx, capture_video, run_name, flags):
    if env_id == "optomech-v1":

        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", **vars(flags))
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **vars(flags))
            env = gym.wrappers.RecordEpisodeStatistics(env)
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
            return env

        return thunk


# ============================================================================
# Observation normalization
# ============================================================================


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Normalize observations based on dtype, converting to float32 in [0, 1]."""
    if obs.dtype == np.uint8:
        return (obs / 255.0).astype(np.float32)
    elif obs.dtype == np.uint16:
        return (obs / 65535.0).astype(np.float32)
    elif obs.dtype in (np.float32, np.float64):
        if obs.max() > 256:
            return (obs / 65535.0).astype(np.float32)
        elif obs.max() > 1.0:
            return (obs / 255.0).astype(np.float32)
    return obs.astype(np.float32)


# ============================================================================
# GAE computation
# ============================================================================


def compute_gae(
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """
    Compute Generalized Advantage Estimation.

    Args:
        next_value: V(s_{T+1}) bootstrap, shape [num_envs]
        rewards: [num_steps, num_envs]
        dones: [num_steps, num_envs]
        values: [num_steps, num_envs]
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [num_steps, num_envs]
        returns: [num_steps, num_envs]
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )

    returns = advantages + values
    return advantages, returns


# ============================================================================
# Recurrent sequence generator
# ============================================================================


def recurrent_generator(
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    prior_actions: torch.Tensor,
    prior_rewards: torch.Tensor,
    hidden_h: torch.Tensor,
    hidden_c: torch.Tensor,
    episode_starts: torch.Tensor,
    dones: torch.Tensor,
    num_minibatches: int,
    seq_len: int,
):
    """
    Yield minibatches of sequence chunks for recurrent PPO training.

    Reshapes [T, N, ...] rollout data into non-overlapping chunks of
    length seq_len, then randomly shuffles and splits into minibatches.

    Args:
        obs, actions, etc.: rollout data, all [T, N, ...]
        hidden_h, hidden_c: stored LSTM hidden states [T, N, num_layers, hidden_dim]
        episode_starts: [T, N] binary mask, 1.0 at first step of new episode
        num_minibatches: how many minibatches to split into
        seq_len: TBPTT chunk length

    Yields:
        dict with batched tensors for one minibatch, each with batch dim first
    """
    T, N = obs.shape[:2]
    num_chunks_per_env = T // seq_len
    total_chunks = num_chunks_per_env * N

    # Reshape into chunks: [num_chunks_per_env, N, seq_len, ...]
    def chunk(x):
        rest = x.shape[2:]
        return x[: num_chunks_per_env * seq_len].reshape(
            num_chunks_per_env, seq_len, N, *rest
        ).transpose(1, 2).reshape(total_chunks, seq_len, *rest)
        # Final shape: [total_chunks, seq_len, ...]

    obs_chunks = chunk(obs)
    action_chunks = chunk(actions)
    log_prob_chunks = chunk(log_probs)
    value_chunks = chunk(values.unsqueeze(-1)).squeeze(-1)
    adv_chunks = chunk(advantages.unsqueeze(-1)).squeeze(-1)
    ret_chunks = chunk(returns.unsqueeze(-1)).squeeze(-1)
    prior_action_chunks = chunk(prior_actions)
    prior_reward_chunks = chunk(prior_rewards.unsqueeze(-1)).squeeze(-1)
    episode_start_chunks = chunk(episode_starts.unsqueeze(-1)).squeeze(-1)
    done_chunks = chunk(dones.unsqueeze(-1)).squeeze(-1)

    # Initial hidden states: take the hidden state at the start of each chunk
    # hidden_h shape: [T, N, num_layers, hidden_dim]
    # We want the hidden state at timesteps 0, seq_len, 2*seq_len, ...
    chunk_start_indices = torch.arange(0, num_chunks_per_env * seq_len, seq_len)
    # [num_chunks_per_env, N, num_layers, hidden_dim]
    h_init = hidden_h[chunk_start_indices]
    c_init = hidden_c[chunk_start_indices]
    # Reshape to [total_chunks, num_layers, hidden_dim]
    h_init = h_init.reshape(total_chunks, *h_init.shape[2:])
    c_init = c_init.reshape(total_chunks, *c_init.shape[2:])

    # Shuffle and yield minibatches
    chunk_size = total_chunks // num_minibatches
    indices = torch.randperm(total_chunks)

    for start in range(0, total_chunks, chunk_size):
        end = start + chunk_size
        mb_idx = indices[start:end]

        yield {
            "obs": obs_chunks[mb_idx],
            "actions": action_chunks[mb_idx],
            "log_probs": log_prob_chunks[mb_idx],
            "values": value_chunks[mb_idx],
            "advantages": adv_chunks[mb_idx],
            "returns": ret_chunks[mb_idx],
            "prior_actions": prior_action_chunks[mb_idx],
            "prior_rewards": prior_reward_chunks[mb_idx],
            "episode_starts": episode_start_chunks[mb_idx],
            "dones": done_chunks[mb_idx],
            "hidden_h": h_init[mb_idx],
            "hidden_c": c_init[mb_idx],
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Load config file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

        cli_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                cli_args.add(arg.split("=")[0].replace("--", "").replace("-", "_"))

        num_applied = 0
        for key, value in config.items():
            if hasattr(args, key) and key not in cli_args and key != "config":
                setattr(args, key, value)
                num_applied += 1
        print(f"Loaded {num_applied} config values from {args.config}")

    # Validate that num_steps is divisible by seq_len
    assert args.num_steps % args.seq_len == 0, (
        f"num_steps ({args.num_steps}) must be divisible by seq_len ({args.seq_len})"
    )

    # Register environments
    gym.envs.registration.register(
        id="optomech-v1",
        entry_point="optomech.optomech:OptomechEnv",
    )
    gym.envs.registration.register(
        id="VisualPendulum-v1",
        entry_point="visual_pendulum:VisualPendulumEnv",
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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Save args to disk
    args_store_path = f"./runs/{run_name}/args.json"
    with open(args_store_path, "w") as f:
        json.dump(vars(args), f)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Device
    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(int(args.gpu_list))
        print(f"Device: CUDA (GPU {args.gpu_list})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Environment setup
    if args.subproc_env:
        envs = gym.vector.SubprocVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name, args)
                for i in range(args.num_envs)
            ],
        )
        print(f"Environment: SubprocVectorEnv ({args.num_envs} envs)")
    elif args.async_env:
        envs = gym.vector.AsyncVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name, args)
                for i in range(args.num_envs)
            ],
        )
        print(f"Environment: AsyncVectorEnv ({args.num_envs} envs)")
    else:
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name, args)
                for i in range(args.num_envs)
            ],
        )
        print(f"Environment: SyncVectorEnv ({args.num_envs} envs)")

    # Load pretrained encoder if specified
    base_encoder = None
    if args.pretrained_encoder_path:
        base_encoder = load_pretrained_encoder(
            args.pretrained_encoder_path,
            None,
            device=device,
            freeze=args.freeze_encoder,
        )
        print(
            f"Loaded pretrained encoder ({'frozen' if args.freeze_encoder else 'trainable'})"
        )

    # Create actor-critic
    agent = RecurrentActorCritic(
        envs=envs,
        device=device,
        encoder=copy.deepcopy(base_encoder) if base_encoder is not None else None,
        lstm_hidden_dim=args.lstm_hidden_dim,
        channel_scale=args.channel_scale,
        fc_scale=args.fc_scale,
        action_scale=args.action_scale,
        init_log_std=args.init_log_std,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Actor-Critic: {total_params:,} trainable parameters")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Derived constants
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    batch_size = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // batch_size
    minibatch_size = batch_size // args.num_minibatches

    print(f"\nTraining configuration:")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Batch size: {batch_size} ({args.num_envs} envs x {args.num_steps} steps)")
    print(f"  Num updates: {num_updates:,}")
    print(f"  Minibatch size: {minibatch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Update epochs: {args.update_epochs}")

    # ------------------------------------------------------------------
    # Pre-allocate rollout storage
    # ------------------------------------------------------------------
    obs_buf = torch.zeros(args.num_steps, args.num_envs, *obs_shape)
    action_buf = torch.zeros(args.num_steps, args.num_envs, *action_shape)
    logprob_buf = torch.zeros(args.num_steps, args.num_envs)
    reward_buf = torch.zeros(args.num_steps, args.num_envs)
    done_buf = torch.zeros(args.num_steps, args.num_envs)
    value_buf = torch.zeros(args.num_steps, args.num_envs)
    prior_action_buf = torch.zeros(args.num_steps, args.num_envs, *action_shape)
    prior_reward_buf = torch.zeros(args.num_steps, args.num_envs)
    hidden_h_buf = torch.zeros(
        args.num_steps, args.num_envs, agent.lstm_num_layers, args.lstm_hidden_dim
    )
    hidden_c_buf = torch.zeros(
        args.num_steps, args.num_envs, agent.lstm_num_layers, args.lstm_hidden_dim
    )
    episode_start_buf = torch.zeros(args.num_steps, args.num_envs)

    # ------------------------------------------------------------------
    # Baseline evaluation (same pattern as DDPG)
    # ------------------------------------------------------------------
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
            prelearning_sample="zeros",
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

    # ------------------------------------------------------------------
    # Initialize environment state
    # ------------------------------------------------------------------
    obs_np, _ = envs.reset(seed=args.seed)
    obs_np = normalize_obs(obs_np)

    # Print observation diagnostics
    print(f"\nObservation shape: {obs_np.shape}, dtype: {obs_np.dtype}")
    print(f"Observation range: [{obs_np.min():.4f}, {obs_np.max():.4f}]")

    # Running state for each environment
    next_obs = torch.from_numpy(obs_np).float()
    next_done = torch.zeros(args.num_envs)

    # LSTM hidden state: [num_layers, hidden_dim] per environment
    # Stored as list of (h, c) tuples, one per env
    lstm_h = torch.zeros(agent.lstm_num_layers, args.num_envs, args.lstm_hidden_dim, device=device)
    lstm_c = torch.zeros(agent.lstm_num_layers, args.num_envs, args.lstm_hidden_dim, device=device)

    # Prior action and reward per environment
    running_prior_action = torch.zeros(args.num_envs, *action_shape)
    running_prior_reward = torch.zeros(args.num_envs)

    # Track whether each env just started a new episode
    running_episode_start = torch.ones(args.num_envs)  # True at very start

    # Track first-step reward for gain computation
    first_step_reward = None

    # Episode tracking
    best_train_return = -np.inf
    recent_returns = []

    global_step = 0
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Starting PPO training: {args.total_timesteps:,} timesteps, {num_updates:,} updates")
    print(f"{'='*60}\n")

    # ==================================================================
    # Training loop
    # ==================================================================
    for update in range(1, num_updates + 1):

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        # ==============================================================
        # ROLLOUT PHASE: collect num_steps transitions from num_envs
        # ==============================================================
        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs

            # Store current state in buffer
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            prior_action_buf[step] = running_prior_action
            prior_reward_buf[step] = running_prior_reward
            episode_start_buf[step] = running_episode_start

            # Store hidden state BEFORE this step's forward pass
            hidden_h_buf[step] = lstm_h.permute(1, 0, 2).cpu()  # [N, layers, hidden]
            hidden_c_buf[step] = lstm_c.permute(1, 0, 2).cpu()

            with torch.no_grad():
                obs_device = next_obs.to(device)
                prior_act_device = running_prior_action.to(device)
                prior_rew_device = running_prior_reward.to(device)

                action, log_prob, _, value, (lstm_h, lstm_c) = (
                    agent.get_action_and_value(
                        obs_device,
                        prior_act_device,
                        prior_rew_device,
                        (lstm_h, lstm_c),
                    )
                )

            # Store action outputs
            action_buf[step] = action.cpu()
            logprob_buf[step] = log_prob.cpu()
            value_buf[step] = value.cpu()

            # Clamp actions for environment interaction
            action_np = (
                action.cpu()
                .numpy()
                .clip(
                    args.action_scale * envs.single_action_space.low,
                    args.action_scale * envs.single_action_space.high,
                )
            )

            # Step environment
            next_obs_np, rewards_np, terminations, truncations, infos = envs.step(
                action_np
            )
            next_obs_np = normalize_obs(next_obs_np)
            rewards_np = args.reward_scale * rewards_np

            if first_step_reward is None:
                first_step_reward = rewards_np.copy()

            # Convert to tensors
            next_obs = torch.from_numpy(next_obs_np).float()
            reward_buf[step] = torch.from_numpy(rewards_np).float()
            dones_step = np.logical_or(terminations, truncations)
            next_done = torch.from_numpy(dones_step.astype(np.float32))

            # Update running prior action/reward
            running_prior_action = action.cpu().clone()
            running_prior_reward = torch.from_numpy(rewards_np).float()

            # Track episode starts for next step
            running_episode_start = next_done.clone()

            # Handle episode resets
            for env_idx in range(args.num_envs):
                if dones_step[env_idx]:
                    # Reset hidden state for this environment
                    lstm_h[:, env_idx, :] = 0.0
                    lstm_c[:, env_idx, :] = 0.0
                    # Reset prior action/reward
                    running_prior_action[env_idx] = 0.0
                    running_prior_reward[env_idx] = 0.0

            # Log episode completions
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None:
                        continue
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    ep_gain = ep_return - (
                        args.max_episode_steps
                        * np.mean(first_step_reward)
                        / args.reward_scale
                    )

                    writer.add_scalar(
                        "episode/episodic_return", ep_return, global_step
                    )
                    writer.add_scalar(
                        "episode/episodic_return_gain", ep_gain, global_step
                    )
                    writer.add_scalar(
                        "episode/episodic_length", ep_length, global_step
                    )

                    recent_returns.append(ep_gain)
                    if len(recent_returns) > 100:
                        recent_returns.pop(0)

                    print(
                        f"Episode: steps={ep_length}, return={ep_return:.2f}, "
                        f"gain={ep_gain:.2f} (step {global_step})"
                    )

                    first_step_reward = None

        # ==============================================================
        # COMPUTE GAE
        # ==============================================================
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs.to(device),
                running_prior_action.to(device),
                running_prior_reward.to(device),
                (lstm_h, lstm_c),
            ).cpu()

        advantages, returns = compute_gae(
            next_value,
            reward_buf,
            done_buf,
            value_buf,
            args.gamma,
            args.gae_lambda,
        )

        # ==============================================================
        # OPTIMIZATION PHASE
        # ==============================================================
        agent.train()

        # Track metrics across epochs
        all_pg_losses = []
        all_v_losses = []
        all_entropy = []
        all_approx_kl = []
        all_clipfrac = []

        for epoch in range(args.update_epochs):
            for batch in recurrent_generator(
                obs_buf,
                action_buf,
                logprob_buf,
                value_buf,
                advantages,
                returns,
                prior_action_buf,
                prior_reward_buf,
                hidden_h_buf,
                hidden_c_buf,
                episode_start_buf,
                done_buf,
                args.num_minibatches,
                args.seq_len,
            ):
                # Move batch to device
                b_obs = batch["obs"].to(device)
                b_actions = batch["actions"].to(device)
                b_logprobs = batch["log_probs"].to(device)
                b_values = batch["values"].to(device)
                b_advantages = batch["advantages"].to(device)
                b_returns = batch["returns"].to(device)
                b_prior_actions = batch["prior_actions"].to(device)
                b_prior_rewards = batch["prior_rewards"].to(device)
                b_episode_starts = batch["episode_starts"].to(device)
                b_hidden_h = batch["hidden_h"].to(device)
                b_hidden_c = batch["hidden_c"].to(device)

                # Prepare hidden state: [batch, num_layers, hidden] -> [num_layers, batch, hidden]
                h0 = b_hidden_h.permute(1, 0, 2).contiguous()
                c0 = b_hidden_c.permute(1, 0, 2).contiguous()

                # Forward pass through sequences with episode boundary handling
                _, new_log_prob, entropy, new_value, _ = (
                    agent.get_action_and_value(
                        b_obs,
                        b_prior_actions,
                        b_prior_rewards,
                        (h0, c0),
                        action=b_actions,
                        episode_starts=b_episode_starts,
                    )
                )
                # new_log_prob, entropy, new_value are [batch, seq_len]

                # Flatten sequence dims for loss computation
                new_log_prob = new_log_prob.reshape(-1)
                entropy = entropy.reshape(-1)
                new_value = new_value.reshape(-1)
                old_log_prob = b_logprobs.reshape(-1)
                mb_advantages = b_advantages.reshape(-1)
                mb_returns = b_returns.reshape(-1)
                mb_values = b_values.reshape(-1)

                # Normalize advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # PPO clipped surrogate objective
                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = (
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Combined loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                all_pg_losses.append(pg_loss.item())
                all_v_losses.append(v_loss.item())
                all_entropy.append(entropy_loss.item())
                all_approx_kl.append(approx_kl.item())
                all_clipfrac.append(clipfrac.item())

            # KL early stopping
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ==============================================================
        # LOGGING
        # ==============================================================
        if update % args.writer_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0

            writer.add_scalar("losses/policy_loss", np.mean(all_pg_losses), global_step)
            writer.add_scalar("losses/value_loss", np.mean(all_v_losses), global_step)
            writer.add_scalar("losses/entropy", np.mean(all_entropy), global_step)
            writer.add_scalar("losses/approx_kl", np.mean(all_approx_kl), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(all_clipfrac), global_step)
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "charts/explained_variance",
                1.0
                - (returns - value_buf).var().item()
                / (returns.var().item() + 1e-8),
                global_step,
            )

        # Progress reporting
        if update % 10 == 0:
            elapsed = time.time() - start_time
            progress = 100 * global_step / args.total_timesteps
            sps = global_step / elapsed if elapsed > 0 else 0
            eta_h = (args.total_timesteps - global_step) / sps / 3600 if sps > 0 else 0
            print(
                f"Update {update}/{num_updates} | {progress:.1f}% | "
                f"Step {global_step:,} | {sps:.0f} SPS | ETA: {eta_h:.1f}h | "
                f"PG: {np.mean(all_pg_losses):.4f} | V: {np.mean(all_v_losses):.4f} | "
                f"Ent: {np.mean(all_entropy):.4f} | KL: {np.mean(all_approx_kl):.4f}"
            )

        # ==============================================================
        # MODEL SAVING
        # ==============================================================
        if args.save_model and len(recent_returns) > 0:
            mean_recent = np.mean(recent_returns)

            # Save best model
            if mean_recent > best_train_return:
                best_train_return = mean_recent

                best_models_path = f"./runs/{run_name}/best_models"
                Path(best_models_path).mkdir(parents=True, exist_ok=True)
                model_path = f"{best_models_path}/best_train_policy.pt"

                wrapper = PPOActorWrapper(agent)
                torch.save(wrapper, model_path)
                print(
                    f"Saved best model (mean return: {best_train_return:.2f}) "
                    f"[step {global_step}]"
                )

                # Evaluate best model
                if args.num_eval_rollouts > 0:
                    episodic_returns_list = []
                    zero_returns_list = []
                    random_returns_list = []

                    for i, eval_rollout_dict in eval_dict.items():
                        env_kwargs = {"seed": eval_rollout_dict["seed"]}
                        episodic_returns = rollout_optomech_policy(
                            model_path,
                            env_vars_path=args_store_path,
                            rollout_episodes=1,
                            exploration_noise=0.0,
                            env_kwargs=env_kwargs,
                        )
                        eval_rollout_dict["on_policy_returns"][update] = episodic_returns
                        episodic_returns_list.append(episodic_returns)
                        zero_returns_list.append(
                            eval_rollout_dict["zero_policy_returns"]
                        )
                        random_returns_list.append(
                            eval_rollout_dict["random_policy_returns"]
                        )

                    ep_arr = np.array(episodic_returns_list).flatten()
                    zero_arr = np.array(zero_returns_list).flatten()
                    random_arr = np.array(random_returns_list).flatten()

                    mean_eval = np.mean(ep_arr)
                    writer.add_scalar(
                        "eval/best_policy_mean_returns", mean_eval, global_step
                    )
                    writer.add_scalar(
                        "eval/best_policy_zero_return_advantage",
                        np.mean(ep_arr / zero_arr),
                        global_step,
                    )
                    writer.add_scalar(
                        "eval/best_policy_random_return_advantage",
                        np.mean(ep_arr / random_arr),
                        global_step,
                    )
                    print(f"Eval mean return: {mean_eval:.2f}")

            # Periodic checkpoint
            elif update % args.model_save_interval == 0:
                ckpt_path = f"./runs/{run_name}/checkpoints"
                Path(ckpt_path).mkdir(parents=True, exist_ok=True)

                # Save deployable actor
                wrapper = PPOActorWrapper(agent)
                torch.save(
                    wrapper,
                    f"{ckpt_path}/{args.exp_name}_{update}_policy.pt",
                )

                # Save full checkpoint for resume
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "update": update,
                    },
                    f"{ckpt_path}/{args.exp_name}_{update}_checkpoint.pt",
                )
                print(f"Saved checkpoint at update {update}")

    envs.close()
    writer.close()
    print("Done.")
