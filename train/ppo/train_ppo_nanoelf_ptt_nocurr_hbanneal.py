"""
PPO training: nanoelf PTT, zero TT error, holding bonus annealed down.

No tip/tilt error injected — the agent must learn to maximize Strehl from
a near-perfect starting point. Holding bonus starts at 1.0 for 50M steps
(learn to hold still), then anneals to 0.0 over 100M steps. As the holding
bonus fades, the agent's own exploration causes TT perturbations that it
must learn to correct — a self-generated curriculum for TT correction.

Usage:
    python train/ppo/train_ppo_nanoelf_ptt_nocurr_hbanneal.py                  # local
    python train/ppo/train_ppo_nanoelf_ptt_nocurr_hbanneal.py --hpc --no-eval  # HPC
"""

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_nanoelf_ptt import NANOELF_TT_ENV_KWARGS

# Zero TT error + holding bonus ON — agent learns from its own exploration
HBANNEAL_ENV_KWARGS = {
    **NANOELF_TT_ENV_KWARGS,
    "init_wind_tip_arcsec_std_tt": 0.0,
    "init_wind_tilt_arcsec_std_tt": 0.0,
    "holding_bonus_weight": 1.0,
    "holding_bonus_min_reward": -1.0,
    "holding_bonus_threshold": -0.7,
}

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
    env_kwargs=HBANNEAL_ENV_KWARGS,
    # --- Holding bonus annealing ---
    holding_bonus_anneal=dict(
        start_value=1.0,
        end_value=0.0,
        warmup_timesteps=50_000_000,
        anneal_timesteps=100_000_000,
    ),
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
    env_kwargs=HBANNEAL_ENV_KWARGS,
    # --- Holding bonus annealing ---
    holding_bonus_anneal=dict(
        start_value=1.0,
        end_value=0.0,
        warmup_timesteps=50_000_000,
        anneal_timesteps=100_000_000,
    ),
)


if __name__ == "__main__":
    run_main(LOCAL_CONFIG, HPC_CONFIG)
