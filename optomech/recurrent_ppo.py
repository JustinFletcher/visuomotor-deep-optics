import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, \
  LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE



import gymnasium as gym

gym.envs.registration.register(
    id='optomech-v1',
    entry_point='optomech.optomech:OptomechEnv',
    # max_episode_steps=4,
    # reward_threshold=flags.reward_threshold,
)

env = GymEnv("optomech-v1") 
model = TensorDictModule(
  nn.Sequential(
      nn.Linear(3, 128), nn.Tanh(),
      nn.Linear(128, 128), nn.Tanh(),
      nn.Linear(128, 128), nn.Tanh(),
      nn.Linear(128, 2),
      NormalParamExtractor()
  ),
  in_keys=["observation"],
  out_keys=["loc", "scale"]
)
critic = ValueOperator(
  nn.Sequential(
      nn.Linear(3, 128), nn.Tanh(),
      nn.Linear(128, 128), nn.Tanh(),
      nn.Linear(128, 128), nn.Tanh(),
      nn.Linear(128, 1),
  ),
  in_keys=["observation"],
)
actor = ProbabilisticActor(
  model,
  in_keys=["loc", "scale"],
  distribution_class=TanhNormal,
  distribution_kwargs={"low": -1.0, "high": 1.0},
  return_log_prob=True
  )
buffer = TensorDictReplayBuffer(
  storage=LazyTensorStorage(1000),
  sampler=SamplerWithoutReplacement(),
  batch_size=50,
  )
collector = SyncDataCollector(
  env,
  actor,
  frames_per_batch=1000,
  total_frames=1_000_000,
)
loss_fn = ClipPPOLoss(actor, critic)
adv_fn = GAE(value_network=critic, average_gae=True, gamma=0.99, lmbda=0.95)
optim = torch.optim.Adam(loss_fn.parameters(), lr=2e-4)

for data in collector:  # collect data
  for epoch in range(10):
      adv_fn(data)  # compute advantage
      buffer.extend(data)
      for sample in buffer:  # consume data
          loss_vals = loss_fn(sample)
          loss_val = sum(
              value for key, value in loss_vals.items() if
              key.startswith("loss")
              )
          loss_val.backward()
          optim.step()
          optim.zero_grad()
  print(f"avg reward: {data['next', 'reward'].mean().item(): 4.4f}")