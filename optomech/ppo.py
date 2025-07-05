from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
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
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, ConvNet, EGreedyModule, LSTMModule, MLP, QValueModule

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


num_cells = 256  # number of cells in each layer i.e. output dim.


# Training parameters
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
total_frames = 1_000_000

# PPO parameters
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# env = TransformedEnv(
#     GymEnv("Pendulum-v1", device=device),
#     Compose(
#         # normalize observations
#         ObservationNorm(in_keys=["observation"]),
#         DoubleToFloat(),
#         StepCounter(),
#     ),
# )

env = TransformedEnv(
    GymEnv("Pendulum-v1", from_pixels=True, device=device),
    Compose(
        InitTracker(),
        DoubleToFloat(),
        StepCounter(),
        ToTensorImage(),
        GrayScale(),
        Resize(84, 84),
        ObservationNorm(standard_normal=True, in_keys=["pixels"]),
        # InitTracker(),
    ),
)

# env.transform[-1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
# Initialize the statistics of the normalization layer.
env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])
td = env.reset()

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)
check_env_specs(env)
rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)


# Build a visual feature extractor.
visual_feature_extractor = TensorDictModule(
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

# Build an LSTM module to map the visual features to a hidden state.
n_cells = visual_feature_extractor(env.reset())["embed"].shape[-1]
lstm_module = LSTMModule(
    input_size=n_cells,
    hidden_size=64,
    in_keys=["embed", "rs_h", "rs_c"],
    out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")]
)
env.append_transform(lstm_module.make_tensordict_primer())

# Build an MLP module w/ 2 outputs (mean & std) for each action dim.
mlp = MLP(
    out_features=2 * env.action_spec.shape[-1],
    num_cells=[
        64,
    ],
    device=device,
)
mlp[-1].bias.data.fill_(0.0)
mlp = TensorDictModule(mlp, in_keys=["intermediate"], out_keys=["action"])

# Build a module that formats MLP output as the mean and std of a distribution.
normal_module = TensorDictModule(
    NormalParamExtractor(),
    in_keys=["action"],
    out_keys=["loc", "scale"],
)

# Combine the modules into a policy network.
policy_network = TensorDictSequential(
    visual_feature_extractor,
    lstm_module.set_recurrent_mode(True),
    mlp,
    normal_module,
)

# Wrap the policy network in a probabilistic actor module to sample actions.
policy_module = ProbabilisticActor(
    module=policy_network,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

# Test the policy network.
print("Running policy:", policy_network(env.reset()))
print(env.rollout(100, policy=policy_module, break_when_any_done=False))
print("Running stoch policy:", policy_module(env.reset()))

# Build a visual feature extractor.
value_visual_feature_extractor = TensorDictModule(
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

# Build an LSTM module to map the visual features to a hidden state.
n_cells = value_visual_feature_extractor(env.reset())["embed"].shape[-1]
value_lstm_module = LSTMModule(
    input_size=n_cells,
    hidden_size=64,
    in_keys=["embed", "rs_h", "rs_c"],
    out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")]
)
env.append_transform(value_lstm_module.make_tensordict_primer())

# Build an MLP module w/ 1 output for the value.
value_mlp = MLP(
    out_features=1,
    num_cells=[
        64,
    ],
    device=device,
)
value_mlp[-1].bias.data.fill_(0.0)
value_mlp = TensorDictModule(value_mlp,
                             in_keys=["intermediate"],
                             out_keys=["value"])

# Combine the modules into a policy network.
value_network = TensorDictSequential(
    value_visual_feature_extractor,
    value_lstm_module.set_recurrent_mode(True),
    value_mlp,
)

print("Running value:", value_network(env.reset()))

value_module = ValueOperator(
    module=value_network,
    in_keys=["pixels"],
)

print("\n\n\n === \n\n\n env.reset():", env.reset()["is_init"])
# Test the value network.
print("Running value:", value_module(env.reset()))
nowexplode

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()