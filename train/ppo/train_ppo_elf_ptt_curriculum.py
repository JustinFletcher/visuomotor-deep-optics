"""
PPO training: ELF PTT with progressive tip/tilt curriculum (45 DOF).

Trains a recurrent PPO agent to co-phase the ELF distributed-aperture
telescope (15 segments × 3 DOF = 45 total: piston + tip + tilt).

Reward is distance-weighted (centered) Strehl. Tip/tilt error ramps
from 0.0 to 2.0 arcsec over training via curriculum annealing.

Curriculum: hold at 0.0 TT for 100M steps, then ramp to 2.0 over 200M steps.

All hyperparameters are embedded in this file so that a training run
is fully reproducible from this single artifact.

Usage:
    python train/ppo/train_ppo_elf_ptt_curriculum.py                  # local run
    python train/ppo/train_ppo_elf_ptt_curriculum.py --hpc            # HPC run (v5, 64 envs)
    python train/ppo/train_ppo_elf_ptt_curriculum.py --hpc --no-eval  # HPC, skip eval
"""

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_piston import ELF_PISTON_ENV_KWARGS

# Start from piston config, enable tip/tilt, zero initial TT for curriculum.
ELF_PTT_CURRICULUM_ENV_KWARGS = {
    **ELF_PISTON_ENV_KWARGS,
    "command_tip_tilt": True,         # <<< ENABLE TIP/TILT (45 DOF total)
    "init_wind_tip_arcsec_std_tt": 0.0,   # curriculum starts aligned
    "init_wind_tilt_arcsec_std_tt": 0.0,  # curriculum starts aligned
    "holding_bonus_weight": 1.0,
    "holding_bonus_min_reward": -1.0,
    "holding_bonus_threshold": -0.7,
}

# ============================================================================
# Curriculum schedule
# ============================================================================
CURRICULUM = dict(
    tip_tilt_start=0.0,           # start aligned (easy)
    tip_tilt_end=2.0,             # ramp to 2.0 arcsec (hard)
    warmup_timesteps=100_000_000, # hold at 0.0 for first 100M steps
    anneal_timesteps=200_000_000, # then ramp over next 200M steps
)

# ============================================================================
# PPO hyperparameters
# ============================================================================

LOCAL_CONFIG = dict(
    # --- PPO algorithm ---
    total_timesteps=1_000_000_000,
    num_envs=8,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=False,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    # --- Model architecture ---
    lstm_hidden_dim=256,
    channel_scale=32,
    fc_scale=256,
    init_log_std=-2.0,
    action_scale=1.0,
    # --- Environment ---
    max_episode_steps=256,
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- Env kwargs ---
    env_kwargs=ELF_PTT_CURRICULUM_ENV_KWARGS,
    # --- Curriculum ---
    curriculum=CURRICULUM,
)

HPC_CONFIG = dict(
    # --- PPO algorithm (tuned for V5 batched GPU env on H100) ---
    total_timesteps=1_000_000_000,
    num_envs=64,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=1.0,
    anneal_lr=False,
    norm_adv=True,
    clip_vloss=True,
    reward_scale=1.0,
    # --- Model architecture ---
    lstm_hidden_dim=256,
    channel_scale=32,
    fc_scale=256,
    init_log_std=-2.0,
    action_scale=1.0,
    # --- Environment ---
    max_episode_steps=256,
    env_version="v5",
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- Env kwargs ---
    env_kwargs=ELF_PTT_CURRICULUM_ENV_KWARGS,
    # --- Curriculum ---
    curriculum=CURRICULUM,
)


if __name__ == "__main__":
    run_main(LOCAL_CONFIG, HPC_CONFIG)
