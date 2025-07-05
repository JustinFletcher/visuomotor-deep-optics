import torch
import tqdm
import multiprocessing
from tensordict.nn import (
    TensorDictModule as TensorDictModule,
    TensorDictSequential,
)
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyModule, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate


import tyro
from dataclasses import dataclass

import os

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 88
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
    buffer_size: int = int(1e5)
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

    init_differential_motion: bool = False
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = False
    """Whether to simulate differential motion."""


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
    control_interval_ms: float = 1.0
    """The interval between control updates."""
    frame_interval_ms: float = 1.0
    """The interval between frames."""
    decision_interval_ms: float = 1.0
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




def main(args):

    is_fork = multiprocessing.get_start_method() == "fork"

    # Check if MPS is available
    if torch.cuda.is_available() and not is_fork:
        print("Running with CUDA")
        device = torch.device("cuda")
    else:
        print("Running with CPU")
        device = torch.device("cpu")


    env = TransformedEnv(
        GymEnv("CartPole-v1", from_pixels=True, device=device),
        Compose(
            ToTensorImage(),
            GrayScale(),
            Resize(84, 84),
            StepCounter(),
            InitTracker(),
            RewardScaling(loc=0.0, scale=0.1),
            ObservationNorm(standard_normal=True, in_keys=["pixels"]),
        ),
    )


    env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])
    td = env.reset()

    feature = TensorDictModule(
        ConvNet(
            num_cells=[32, 32, 64],
            squeeze_output=True,
            aggregator_class=nn.AdaptiveAvgPool2d,
            aggregator_kwargs={"output_size": (1, 1)},
            device=device,
        ),
        in_keys=["pixels"],
        out_keys=["embed"],
    )

    n_cells = feature(env.reset())["embed"].shape[-1]

    lstm = LSTMModule(
        input_size=n_cells,
        hidden_size=128,
        device=device,
        in_key="embed",
        out_key="embed",
    )

    print("in_keys", lstm.in_keys)
    print("out_keys", lstm.out_keys)

    env.append_transform(lstm.make_tensordict_primer())

    print(env)

    mlp = MLP(
        out_features=2,
        num_cells=[
            64,
        ],
        device=device,
    )

    mlp[-1].bias.data.fill_(0.0)
    mlp = TensorDictModule(mlp, in_keys=["embed"], out_keys=["action_value"])

    qval = QValueModule(action_space=None, spec=env.action_spec)

    stoch_policy = TensorDictSequential(feature, lstm, mlp, qval)

    exploration_module = EGreedyModule(
        annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
    )
    stoch_policy = TensorDictSequential(
        stoch_policy,
        exploration_module,
    )

    policy = TensorDictSequential(feature, lstm.set_recurrent_mode(True), mlp, qval,)

    policy(env.reset())

    loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

    updater = SoftUpdate(loss_fn, eps=0.95)

    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    collector = SyncDataCollector(env, stoch_policy, frames_per_batch=50, total_frames=200)
    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(20_000), batch_size=4, prefetch=10
    )

    utd = 16
    pbar = tqdm.tqdm(total=collector.total_frames)
    longest = 0

    traj_lens = []
    for i, data in enumerate(collector):
        if i == 0:
            print(
                "Let us print the first batch of data.\nPay attention to the key names "
                "which will reflect what can be found in this data structure, in particular: "
                "the output of the QValueModule (action_values, action and chosen_action_value),"
                "the 'is_init' key that will tell us if a step is initial or not, and the "
                "recurrent_state keys.\n",
                data,
            )
        pbar.update(data.numel())
        # it is important to pass data that is not flattened
        rb.extend(data.unsqueeze(0).to_tensordict().cpu())
        for _ in range(utd):
            s = rb.sample().to(device, non_blocking=True)
            loss_vals = loss_fn(s)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
        longest = max(longest, data["step_count"].max().item())
        pbar.set_description(
            f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
        )
        exploration_module.step(data.numel())
        updater.step()

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = env.rollout(10000, stoch_policy)
            traj_lens.append(rollout.get(("next", "step_count")).max().item())

        if traj_lens:
            from matplotlib import pyplot as plt

            plt.plot(traj_lens)
            plt.xlabel("Test collection")
            plt.title("Test trajectory lengths")
            plt.show()

    print("Done")

if __name__ == "__main__":
    args = Args()
    main(args)