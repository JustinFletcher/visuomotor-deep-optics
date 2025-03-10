#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=96:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(
    "--seed=88887 --gpu_list=0 --action_scale=1e-1  --exp_name="optomech-action-scale-1e-1""
    "--seed=88887 --gpu_list=1 --action_scale=1e-2  --exp_name="optomech-action-scale-1e-2""
    "--seed=88887 --gpu_list=2 --action_scale=1e-3  --exp_name="optomech-action-scale-1e-3""
    "--seed=88887 --gpu_list=3 --action_scale=1e-4  --exp_name="optomech-action-scale-1e-4""
    "--seed=88887 --gpu_list=4 --action_scale=1e-5  --exp_name="optomech-action-scale-1e-5""
    "--seed=88887 --gpu_list=5 --action_scale=1e-6  --exp_name="optomech-action-scale-1e-6""
    "--seed=88887 --gpu_list=6 --action_scale=1e-7  --exp_name="optomech-action-scale-1e-7""
    "--seed=88887 --gpu_list=7 --action_scale=1e-8  --exp_name="optomech-action-scale-1e-8""

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
        --model_save_interval=10000 \
        --num_envs=1 \
        --tau=0.0001 \
        --exploration_noise=1.0 \
        --decay_rate=0.0001 \
        --reward_function=negastrehl \
        --reward_scale=10.0 \
        --buffer_size=1_000_000 \
        --learning_starts=100_000 \
        --prelearning_sample=scales \
        --policy_frequency=1 \
        --batch_size=32 \
        --max_episode_steps=200 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_fc_scale=256 \
        --qnetwork_fc_scale=256 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --actor_learning_rate=1e-4 \
        --critic_learning_rate=3e-4 \
        $VARIANT &  # Run in background
done

wait