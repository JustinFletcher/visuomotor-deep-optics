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

policy = TensorDictSequential(feature, lstm.set_recurrent_mode(True), mlp, qval, )

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