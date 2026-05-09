"""Direct diagnostic: load a checkpoint, inspect policy mean L1, and
compare deterministic-mean vs zero-action Strehl from the same env state.

Usage:
    poetry run python train/ppo/diagnose_strehl.py <run_dir>

where <run_dir> contains ppo_optomech_*/checkpoints/latest.pt
"""
import sys, os, contextlib, io
from glob import glob
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from train.ppo.train_ppo_elf_dm_strehl_only import ENV_KWARGS, LOCAL_CONFIG
from optomech.optomech.optomech_v5 import BatchedOptomechEnv
from train.ppo.bilateral_dm import BilateralDMVectorEnv
from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
ckpt_path = sorted(glob(os.path.join(
    run_dir, "ppo_optomech_*", "checkpoints", "latest.pt")),
    key=os.path.getmtime)
if not ckpt_path:
    ckpt_path = sorted(glob(os.path.join(
        run_dir, "ppo_optomech_*", "checkpoints", "best.pt")))
ckpt_path = ckpt_path[-1]
print(f"checkpoint: {ckpt_path}")

# Build env (matches training env)
kw = dict(ENV_KWARGS); kw['silence']=True; kw['max_episode_steps']=8
kw['observation_window_size']=1
with contextlib.redirect_stdout(io.StringIO()):
    base = BatchedOptomechEnv(num_envs=4, device='cpu', **kw)
    env = BilateralDMVectorEnv(base, freeze_segments=True, mode='fixed_vertical')

# Load checkpoint
class _Shim:
    single_observation_space = env.single_observation_space
    single_action_space = env.single_action_space
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
config = ckpt["config"]
agent = RecurrentActorCritic(
    _Shim(), torch.device('cpu'),
    lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
    channel_scale=config.get("channel_scale", 32),
    fc_scale=config.get("fc_scale", 256),
    action_scale=config.get("action_scale", 1.0),
    init_log_std=config.get("init_log_std", -0.5),
    model_type=config.get("model_type", "small"),
    target_dim=int(config.get("target_dim", 0)),
    log_std_max=config.get("log_std_max", None),
    log_std_min=config.get("log_std_min", None),
)
agent.load_state_dict(ckpt["model_state_dict"])
agent.eval()
wrapper = PPOActorWrapper(agent)

# 1. Inspect policy log_std envelope
ls = agent.log_std.detach()
print(f"\nlog_std stats: mean={ls.mean():.3f}  min={ls.min():.3f}  "
      f"max={ls.max():.3f}  (config caps: min={config.get('log_std_min')}, "
      f"max={config.get('log_std_max')})")
print(f"sigma stats:   mean={ls.exp().mean():.4f}  min={ls.exp().min():.4f}  "
      f"max={ls.exp().max():.4f}")

# 2. Run one episode with deterministic mean, capture per-step mean L1 + Strehl.
obs, info = env.reset(seed=0)
h = torch.zeros(agent.lstm_num_layers, 4, agent.lstm_hidden_dim)
c = torch.zeros(agent.lstm_num_layers, 4, agent.lstm_hidden_dim)
prior_action = torch.zeros(4, agent.action_dim)
prior_reward = torch.zeros(4)

print(f"\n{'step':>4} {'|mean_act|_L1':>14} {'|mean_act|_max':>14} "
      f"{'env_strehl':>11} {'reward':>11}")
for step in range(8):
    obs_t = torch.from_numpy(obs).float()
    with torch.no_grad():
        action_t, (h, c) = wrapper(obs_t, prior_action, prior_reward, (h, c))
    a_np = action_t.cpu().numpy()
    obs, rew, term, trunc, info = env.step(a_np)
    print(f"{step:>4d} "
          f"{float(np.abs(a_np).mean()):>14.5f} "
          f"{float(np.abs(a_np).max()):>14.5f} "
          f"{float(info['strehl'].mean()):>11.4f} "
          f"{float(rew.mean()):>11.4f}")
    prior_action = action_t

# 3. As a control, do the same rollout but force action=0.
obs, info = env.reset(seed=0)
print(f"\n-- control rollout: action forced to 0 each step --")
print(f"{'step':>4} {'env_strehl':>11} {'reward':>11}")
A = env.single_action_space.shape[0]
for step in range(8):
    a_np = np.zeros((4, A), dtype=np.float32)
    obs, rew, term, trunc, info = env.step(a_np)
    print(f"{step:>4d} "
          f"{float(info['strehl'].mean()):>11.4f} "
          f"{float(rew.mean()):>11.4f}")
