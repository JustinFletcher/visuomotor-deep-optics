#!/bin/bash
#SBATCH --job-name=tbptt           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=wdata/scratch/fletch/tbptt.out
#SBATCH --error=wdata/scratch/fletch/tbptt.err
#SBATCH --ntasks-per-node=16          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(

    "--gpu_list=0 --max_episode_steps=256 --exp_name="tbptt-steps256""

)

cd visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_tbptt_lstm.py \
        --env-id optomech-v1 \
        --object_type=single \
        --ao_interval_ms=5.0 \
        --control_interval_ms=5.0 \
        --frame_interval_ms=5.0 \
        --decision_interval_ms=5.0 \
        --num_atmosphere_layers=0 \
        --aperture_type=elf \
        --focal_plane_image_size_pixels=256 \
        --observation_mode=image_only \
        --command_secondaries \
        --init_differential_motion \
        --model_wind_diff_motion \
        --num_envs=1 \
        --reward_function=align \
        --batch_size=32 \
        --tau=0.0003 \
        --policy_frequency=2 \
        --reward_scale=1.0 \
        --max_grad_norm=100_000 \
        --buffer_size=1_000_000 \
        --learning_starts=5_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=1_000 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-4 \
        --critic_learning_rate=3e-4 \
        --qnetwork_channel_scale=32 \
        --actor_channel_scale=32 \
        --actor_fc_scale=512 \
        --qnetwork_fc_scale=512 \
        --action_scale=1.0 \
        --use_q_bias \
        --exploration_noise=0.000001 \
        --decay_rate=0.998 \
        --writer_interval=100 \
        --save_model \
        --tbptt_seq_len=64 \
        --num_eval_rollouts=4 \
        --model_save_interval=10_000 \
        $VARIANT &  # Run in background
done

wait