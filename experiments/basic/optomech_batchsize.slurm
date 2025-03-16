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

    "--seed=123456 --gpu_list=0 --batch_size=64  --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-batch64-clr1e-3""
    "--seed=123456 --gpu_list=1 --batch_size=128 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-batch128-clr1e-3""
    "--seed=123456 --gpu_list=2 --batch_size=256 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-batch256-clr1e-3""
    "--seed=123456 --gpu_list=3 --batch_size=512 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=1e-3 --exp_name="optomech-batch512-clr1e-3""
    "--seed=123456 --gpu_list=4 --batch_size=64  --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=3e-3 --exp_name="optomech-batch64-clr3e-3""
    "--seed=123456 --gpu_list=5 --batch_size=128 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=3e-3 --exp_name="optomech-batch128-clr3e-3""
    "--seed=123456 --gpu_list=6 --batch_size=256 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=3e-3 --exp_name="optomech-batch256-clr3e-3""
    "--seed=123456 --gpu_list=7 --batch_size=512 --policy_frequency=4 --actor_learning_rate=1e-3 --critic_learning_rate=3e-3 --exp_name="optomech-batch512-clr3e-3""


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