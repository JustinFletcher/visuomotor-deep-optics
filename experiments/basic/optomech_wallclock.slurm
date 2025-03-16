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

    "--seed=123456 --gpu_list=0 --policy_frequency=1 --actor_learning_rate=1e-4 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-4-c1e-3-pf-1""
    "--seed=123456 --gpu_list=1 --policy_frequency=1 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-3-c1e-3-pf-1""
    "--seed=123456 --gpu_list=2 --policy_frequency=2 --actor_learning_rate=1e-4 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-4-c1e-3-pf-2""
    "--seed=123456 --gpu_list=3 --policy_frequency=2 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-3-c1e-3-pf-2""
    "--seed=123456 --gpu_list=4 --policy_frequency=4 --actor_learning_rate=1e-4 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-4-c1e-3-pf-4""
    "--seed=123456 --gpu_list=5 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-3-c1e-3-pf-4""
    "--seed=123456 --gpu_list=6 --policy_frequency=8 --actor_learning_rate=1e-4 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-4-c1e-3-pf-8""
    "--seed=123456 --gpu_list=7 --policy_frequency=8 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-wallclock-a1e-3-c1e-3-pf-8""


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
        --tau=0.0001 \
        --exploration_noise=1.0 \
        --decay_rate=0.0001 \
        --l2_reg=0.0 \
        --reward_function=negastrehl \
        --reward_scale=10.0 \
        --max_grad_norm=0.01 \
        --buffer_size=1_000_000 \
        --learning_starts=100_000 \
        --prelearning_sample=scales \
        --batch_size=32 \
        --max_episode_steps=200 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_fc_scale=256 \
        --qnetwork_fc_scale=256 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --action_scale=1e-2 \
        $VARIANT &  # Run in background
done

wait