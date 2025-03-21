#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=48:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(

    "--seed=111111 --actor_fc_scale=256 --qnetwork_fc_scale=256 --gpu_list=0 --exp_name="optomech-std-256""
    "--seed=222222 --actor_fc_scale=256 --qnetwork_fc_scale=256 --gpu_list=1 --exp_name="optomech-std-256""
    "--seed=333333 --actor_fc_scale=256 --qnetwork_fc_scale=256 --gpu_list=2 --exp_name="optomech-std-256""
    "--seed=444444 --actor_fc_scale=256 --qnetwork_fc_scale=256 --gpu_list=3 --exp_name="optomech-std-256""
    "--seed=111111 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --gpu_list=1 --exp_name="optomech-std-1024""
    "--seed=222222 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --gpu_list=2 --exp_name="optomech-std-1024""
    "--seed=333333 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --gpu_list=3 --exp_name="optomech-std-1024""
    "--seed=444444 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --gpu_list=4 --exp_name="optomech-std-1024""
)

cd visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_lstm.py \
        --env-id optomech-v1 \
        --object_type=single \
        --ao_interval_ms=1.0 \
        --control_interval_ms=1.0 \
        --frame_interval_ms=1.0 \
        --decision_interval_ms=1.0 \
        --num_atmosphere_layers=0 \
        --aperture_type=nanoelf \
        --focal_plane_image_size_pixels=64 \
        --observation_mode=image_only \
        --command_secondaries \
        --incremental_control \
        --model_wind_diff_motion \
        --init_differential_motion \
        --save_model \
        --model_save_interval=100000 \
        --num_envs=1 \
        --batch_size=32 \
        --tau=0.0001 \
        --exploration_noise=0.01 \
        --policy_frequency=2 \
        --decay_rate=0.995 \
        --l2_reg=0.0 \
        --reward_function=negastrehl \
        --reward_scale=10.0 \
        --max_grad_norm=1.0 \
        --buffer_size=1_000_000 \
        --learning_starts=100_000 \
        --max_episode_steps=500 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-4 \
        --critic_learning_rate=3e-4 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --action_scale=1.0 \
        $VARIANT &  # Run in background
done

wait