# CLAUDE.md — Agent Briefing for visuomotor-deep-optics

## What this project is

Recurrent PPO agents that control a segmented space telescope in real time.
The agent observes a single focal-plane image each step and outputs
piston / tip / tilt corrections to co-phase mirror segments, maximising
Strehl ratio under atmospheric and mechanical disturbances.

The optical simulation (HCIPy + GPU-accelerated Matrix Fourier Transform)
runs as a Gymnasium environment.  Training uses a single-file recurrent PPO
implementation with TBPTT and curriculum annealing.

## Repository layout

```
train/ppo/                  # Training, evaluation, and agent code
  train_ppo_optomech.py     # Core PPO loop (shared by all experiments)
  train_ppo_nanoelf_*.py    # Per-experiment configs (env_kwargs + PPO hyperparams)
  ppo_models.py             # RecurrentActorCritic, PPOActorWrapper
  agents.py                 # BaseAgent / SingleModelAgent / CompositeAgent
  policy_spec.py            # YAML policy spec loader
  rollout.py                # Standalone evaluation with GIFs + figures
  sweep_tiptilt.py          # Tip/tilt error sweep evaluation
  analyze_sweep.py          # 15-figure deep analysis of sweep results
  specs/                    # YAML policy specifications

optomech/optomech/          # Gymnasium environments
  optomech_v2.py            # Configurable HCIPy env (CPU)
  optomech_v3.py            # CPU-optimised HCIPy
  optomech_v4.py            # GPU-accelerated (extracted MFT matrices) — default local
  optomech_v5.py            # Batched GPU VectorEnv — for HPC
  env_config.py             # OptomechEnvConfig dataclass (100+ params)

experiments/                # SLURM sbatch scripts
  basic/                    # Hyperparameter sweeps and ablations
  nanoelfplus/              # Larger aperture experiments
  multiscale/               # Multi-scale training
  tbptt/                    # TBPTT-specific experiments
  vispend/                  # Visual pendulum baseline

utils/                      # Utilities
  sync_remote_runs.py       # Rsync runs from HPC (Kerberos + retry)

tests/                      # Tests
  test_ppo_nanoelf.py       # Smoke test: trains piston agent, checks it beats baselines
  test_ppo_visual_pendulum.py
  test_v4_parity.py         # V4 output matches V3

figures/                    # Paper figure generation scripts (untracked)
runs/                       # Training outputs: checkpoints, TB logs (gitignored)
```

## Setup

```bash
# Requires Python 3.10.x exactly
poetry install
```

Key dependencies: PyTorch >=2.0, Gymnasium <1.0, HCIPy >=0.6, stable-baselines3 2.5.0, PyYAML.

## Environment versions

| Version | Description | Speed | Use case |
|---------|-------------|-------|----------|
| v3 | CPU HCIPy | ~200 SPS laptop | Legacy |
| v4 | GPU MFT extraction | ~650 SPS H100 | **Local training & eval** |
| v5 | Batched GPU VectorEnv | ~4000 SPS @ 64 envs | **HPC training** |

V4/V5 pre-extract MFT matrices from HCIPy at init, then run
`E_focal = norm * w_in * (M1 @ E_pupil @ M2)` as PyTorch ops on GPU.

On Apple Silicon, LSTM falls back to CPU (MPS has LSTM issues).

## Training

Every experiment script defines its own `ENV_KWARGS`, `LOCAL_CONFIG`, and
`HPC_CONFIG` dicts, then calls `run_main()` from `train_ppo_optomech.py`.
All hyperparameters are embedded in the script for reproducibility.

### Local training

```bash
poetry run python train/ppo/train_ppo_nanoelf_piston.py          # 2 DOF, fast
poetry run python train/ppo/train_ppo_nanoelf_ptt_curriculum.py   # 6 DOF, curriculum
```

### HPC (SLURM)

```bash
sbatch experiments/basic/optomech_nanoelf_piston.slurm
```

Or directly:
```bash
poetry run python train/ppo/train_ppo_nanoelf_ptt_curriculum.py --hpc --no-eval
```

`--hpc` switches to V5 batched env with 64 parallel envs.
`--no-eval` skips expensive eval episodes during training.

### Key training scripts

| Script | DOF | TT error | Notes |
|--------|-----|----------|-------|
| `train_ppo_nanoelf_piston.py` | 2 | N/A | Piston only; fastest to converge |
| `train_ppo_nanoelf_ptt_curriculum.py` | 6 | 0→2.0 arcsec | Progressive TT ramp over 200M steps |
| `train_ppo_nanoelf_ptt_nocurr.py` | 6 | 2.0 | Full TT from step 0 |
| `train_ppo_nanoelf_ptt_nocurr_hbanneal.py` | 6 | 0.0 | Zero TT; holding bonus anneals to 0 |
| `train_ppo_nanoelf_ptt_nocurr_medium.py` | 6 | 2.0 | ResNet18 encoder |

## Model architecture

```
obs (B, C, H, W)
  → CNN [small: 3-layer Impala | medium: ResNet18]
  → MLP (1 hidden, ReLU)
  → cat(features, prior_action, prior_reward)
  → LSTM (1-2 layers, hidden_dim 128-256)
  ├→ policy_head → μ(s), log_std → Normal(μ, σ) → clamp to action bounds
  └→ value_head → V(s)
```

Shared encoder between policy and value.  Recurrent because the agent only
sees one focal-plane image (partial observability).

## Evaluation

### Single rollout

```bash
poetry run python train/ppo/rollout.py \
    --checkpoint runs/<run>/checkpoints/update_XXXXX.pt \
    --env-version v4 --num-episodes 8
```

### Tip/tilt sweep

```bash
poetry run python train/ppo/sweep_tiptilt.py \
    --checkpoint runs/<run>/checkpoints/update_XXXXX.pt \
    --env-version v4 --tt-min 0.0 --tt-max 2.0 --tt-steps 8 \
    --num-episodes 4
```

### Composite policy (multi-phase)

Create a YAML spec (see `train/ppo/specs/` for examples):

```yaml
type: composite
phases:
  - name: coarse-align
    checkpoint: runs/.../model_a.pt
    until:
      step: 32                    # switch after 32 steps
  - name: fine-hold
    checkpoint: runs/.../model_b.pt
```

Then use `--policy-spec` instead of `--checkpoint`:

```bash
poetry run python train/ppo/rollout.py \
    --policy-spec train/ppo/specs/my_spec.yaml \
    --env-version v4 --num-episodes 4

poetry run python train/ppo/sweep_tiptilt.py \
    --policy-spec train/ppo/specs/my_spec.yaml \
    --env-version v4 --tt-min 0.0 --tt-max 2.0 --tt-steps 8
```

Supported phase triggers:
- `step: N` — switch after N steps in the phase
- `metric_above: {strehl: 0.8}` — switch when metric exceeds threshold
- `metric_below: {mse: 0.01}` — switch when metric drops below threshold
- `episode_fraction: 0.5` — switch at fraction of max_episode_steps

### Replot existing sweep

```bash
poetry run python train/ppo/sweep_tiptilt.py \
    --replot test_output/sweep_.../sweep_results.json
```

## Reward function

```
R = w_strehl × Strehl
  + w_centered_strehl × CenteredStrehl
  + w_action × (-||a||₁)
  + w_oob × (-oob_fraction)
  + w_holding × (1 if |a| < threshold else 0)
```

Weights are set per-experiment in ENV_KWARGS.  Holding bonus is typically
annealed from 1.0 to 0.0 over training via the curriculum system.

## Environment parameters (important ones)

```python
# Control DOFs
"command_secondaries": True       # enable piston control
"command_tip_tilt": True          # enable tip/tilt control

# Disturbances
"init_wind_piston_micron_std": 3.0
"init_wind_tip_arcsec_std_tt": 0.05  # swept in evaluation
"init_wind_tilt_arcsec_std_tt": 0.05

# Reward
"action_penalty_weight": 0.5
"reward_weight_strehl": 1.0
"holding_bonus_weight": 1.0
"holding_bonus_threshold": -0.7

# Episode
"max_episode_steps": 256
```

## Checkpoint structure

```python
{
    "model_state_dict": agent.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": int,
    "update": int,
    "best_eval_return": float,
    "config": dict,   # full hyperparams (env_kwargs + PPO config)
}
```

Healthy checkpoints are ~27MB.  A truncated file (e.g. 23MB) indicates
corruption from a killed process — use the previous checkpoint.

## Syncing runs from HPC

```bash
poetry run python utils/sync_remote_runs.py
```

Uses rsync with Kerberos auth and retry logic.  Configured for MHPCC.

## Git workflow

- `main`: stable, pushed to remote
- `dev`: integration branch, pushed to remote
- Feature work happens on `dev`; merge to `main` when stable

## Common pitfalls

- **MPS + LSTM**: Apple Silicon MPS backend has LSTM bugs; the code falls
  back to CPU automatically via `_auto_device()` in rollout.py.
- **Gymnasium version**: Must be <1.0.  The API changed in 1.0.
- **Python version**: Strict 3.10.x requirement (HCIPy + torch compatibility).
- **Corrupted checkpoints**: If `torch.load` raises
  `PytorchStreamReader failed reading zip archive`, the file was truncated.
  Use `ls -lh` to check size (should be ~27MB) and fall back to the
  previous numbered checkpoint.
- **V5 on laptop**: Don't use V5 locally — it's designed for multi-GPU HPC.
  Use V4 for local work.
- **Environment registration**: Each call to `run_rollouts` or the env
  constructors calls `register_optomech()`.  This is idempotent but
  required before `gym.make()`.
