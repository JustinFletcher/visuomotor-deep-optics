"""
PPO training: ELF incremental bootstrapping (45 DOF).

Trains a recurrent PPO agent to co-phase one additional segment of the
ELF distributed-aperture telescope, given that ``phased_count`` segments
are already aligned.  Non-target segments are tipped/tilted off the
focal plane so the agent only needs to learn corrections for the target.

The full 45-DOF action space (15 segments × 3 PTT) is always active.
The agent learns through the reward signal which DOFs are relevant.

Chain 15 trained models (phased_count 0..14) via a CompositeAgent YAML
spec for full end-to-end alignment.

Usage:
    # Single phase locally
    python train/ppo/train_ppo_elf_bootstrap.py --phased-count 0

    # HPC with SLURM array job (see run_elf_bootstrap.sbatch)
    python train/ppo/train_ppo_elf_bootstrap.py --hpc --no-eval --phased-count $SLURM_ARRAY_TASK_ID
"""

import sys
import argparse

from train.ppo.train_ppo_elf_piston import ELF_PISTON_ENV_KWARGS
from train.ppo.train_ppo_optomech import run_main

# ============================================================================
# Environment kwargs — inherits ELF piston base, enables tip/tilt + bootstrap
# ============================================================================

ELF_BOOTSTRAP_ENV_KWARGS = {
    **ELF_PISTON_ENV_KWARGS,
    "command_tip_tilt": True,            # full 45 DOF
    "bootstrap_phase": True,
    "bootstrap_phased_count": 0,         # overridden by --phased-count
    # Hard DOF mask: every action DOF except the target segment's PTT
    # is zeroed before the action reaches the optical system. This is
    # the strict equivalent of SMAES's free_segments=[target]
    # constraint — structurally inert non-target outputs, no reliance
    # on a penalty beating the entropy bonus.
    "bootstrap_mask_nontarget": True,
    # Retained for backward compatibility but ignored while the mask
    # is active. Kept at base (1x) so nothing surprising happens if
    # the mask is ever disabled.
    "bootstrap_nontarget_penalty_multiplier": 1.0,
    # Tight action bounds (~3x the wind disturbance magnitude) — same as
    # ELF_PTT_TIGHT in the SMAES bootstrap pipeline.  This prevents the
    # optimizer from wandering into off-axis pseudo-phased solutions.
    "max_piston_correction_micron": 1.0,
    "max_tip_correction_arcsec":    3.0,
    "max_tilt_correction_arcsec":   3.0,
    # 1 ms timing throughout: AO sub-step, control command, detector
    # frame, and agent decision all cadence at 1 kHz. Overrides the
    # 100 ms defaults inherited from ELF_PISTON_ENV_KWARGS.
    "ao_interval_ms": 1.0,
    "control_interval_ms": 1.0,
    "frame_interval_ms": 1.0,
    "decision_interval_ms": 1.0,
    # Rescale centered_strehl reward per phase so -1.0 corresponds to
    # the prior phase's solved state and 0.0 corresponds to this
    # phase's solved state. Without this, higher phases see a narrow
    # [~-0.13, 0] reward window because start strehl is already close
    # to 1.0 against the full-aligned reference — less dynamic range
    # for PPO to exploit.
    "bootstrap_rescale_reward": True,
}


# ============================================================================
# PPO hyperparameters
# ============================================================================

LOCAL_CONFIG = dict(
    # --- PPO algorithm ---
    total_timesteps=100_000_000,
    num_envs=8,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.98,
    gae_lambda=0.95,
    clip_coef=0.2,
    # Entropy is summed across 45 DOFs (dist.entropy().sum(dim=-1)), so
    # the bonus scales linearly with action_dim. The action penalty is
    # averaged across DOFs, so the entropy/penalty ratio grew by 45²/6²
    # ≈ 56× relative to NanoELF. Scale ent_coef by 6/45 to restore the
    # per-DOF balance that worked at the smaller aperture.
    ent_coef=0.001,
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
    max_episode_steps=64,
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- Env kwargs ---
    env_kwargs=ELF_BOOTSTRAP_ENV_KWARGS,
)

HPC_CONFIG = dict(
    # --- PPO algorithm (tuned for V5 batched GPU env on H100) ---
    total_timesteps=100_000_000,
    num_envs=64,
    num_steps=128,
    num_minibatches=4,
    update_epochs=4,
    seq_len=32,
    learning_rate=3e-4,
    gamma=0.98,
    gae_lambda=0.95,
    clip_coef=0.2,
    # Entropy is summed across 45 DOFs (dist.entropy().sum(dim=-1)), so
    # the bonus scales linearly with action_dim. The action penalty is
    # averaged across DOFs, so the entropy/penalty ratio grew by 45²/6²
    # ≈ 56× relative to NanoELF. Scale ent_coef by 6/45 to restore the
    # per-DOF balance that worked at the smaller aperture.
    ent_coef=0.001,
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
    max_episode_steps=64,
    env_version="v5",
    # --- Evaluation ---
    eval_interval=100,
    eval_episodes=8,
    eval_seeds=None,
    pass_threshold_ratio=1.1,
    seed=1,
    # --- Model saving ---
    model_save_interval=100,
    # --- TensorBoard ---
    # Downsample per-step scalars by 32x. With num_steps=128 we emit 4
    # step-logs per rollout instead of 128 — roughly 30x smaller TB
    # event files without losing per-update visibility.
    tb_step_log_interval=32,
    # Eval-time figure DPI. Default 100; dropping to 48 cuts each
    # embedded TB figure roughly 4x in bytes (area scales as dpi^2).
    # With 3 figures per eval round and ~25 eval rounds over a 100M-
    # step run at eval_interval=100, this keeps figure bloat under
    # ~40 MB per phase.
    eval_figure_dpi=48,
    # --- Env kwargs ---
    env_kwargs=ELF_BOOTSTRAP_ENV_KWARGS,
)


if __name__ == "__main__":
    # Parse --phased-count before run_main sees the args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--phased-count", type=int, default=0,
        help="Number of already co-phased segments [0..14]")
    pre_args, remaining = pre_parser.parse_known_args()

    # Inject phased_count into env_kwargs for both configs
    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        cfg["env_kwargs"]["bootstrap_phased_count"] = pre_args.phased_count

    # Remove --phased-count from sys.argv so run_main's parser doesn't choke
    sys.argv = [sys.argv[0]] + remaining

    run_main(LOCAL_CONFIG, HPC_CONFIG)
