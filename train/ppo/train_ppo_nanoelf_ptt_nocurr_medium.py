"""
PPO training: nanoelf PTT, no curriculum, MEDIUM model (ResNet18 encoder).

Same environment setup as train_ppo_nanoelf_ptt_nocurr.py (full TT=2.0 from
step 0, no curriculum) but uses the "medium" model architecture:
  - ResNet18 backbone as visual encoder (replacing 3-layer CNN)
  - 256-dim LSTM hidden state
  - 256-dim FC layers in policy/value heads

Usage:
    python train/ppo/train_ppo_nanoelf_ptt_nocurr_medium.py                  # local
    python train/ppo/train_ppo_nanoelf_ptt_nocurr_medium.py --hpc --no-eval  # HPC
"""

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_nanoelf_ptt_nocurr import NOCURR_ENV_KWARGS

# ============================================================================
# PPO hyperparameters — medium model, no curriculum
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
    # --- Model architecture (medium: ResNet18 + 256 LSTM/FC) ---
    model_type="medium",
    lstm_hidden_dim=256,
    channel_scale=32,     # unused by ResNet18, kept for config consistency
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
    env_kwargs=NOCURR_ENV_KWARGS,
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
    # --- Model architecture (medium: ResNet18 + 256 LSTM/FC) ---
    model_type="medium",
    lstm_hidden_dim=256,
    channel_scale=32,     # unused by ResNet18, kept for config consistency
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
    env_kwargs=NOCURR_ENV_KWARGS,
)


if __name__ == "__main__":
    run_main(LOCAL_CONFIG, HPC_CONFIG)
