#!/bin/bash
#SBATCH --job-name=omclip50           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(

    "--gpu_list=0 --exp_name="command_tip_tilt""
    "--gpu_list=1 --exp_name="command_tip_tilt""
    "--gpu_list=2 --exp_name="command_tip_tilt""
    "--gpu_list=3 --exp_name="command_tip_tilt""
    "--gpu_list=4 --exp_name="command_tip_tilt""
    "--gpu_list=5 --exp_name="command_tip_tilt""
    "--gpu_list=6 --exp_name="command_tip_tilt""
    "--gpu_list=7 --exp_name="command_tip_tilt""
    
    

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
        --command_tip_tilt \
        --target_smoothing \
        --incremental_control \
        --model_wind_diff_motion \
        --init_differential_motion \
        --save_model \
        --model_save_interval=100_000 \
        --num_envs=1 \
        --batch_size=32 \
        --gamma=0.95 \
        --l2_reg=1e-3 \
        --reward_function=strehl \
        --reward_scale=10.0 \
        --max_grad_norm=50.0 \
        --tau=0.0001 \
        --buffer_size=1_000_000 \
        --learning_starts=50_000 \
        --actor_training_delay=50_000 \
        --policy_frequency=1 \
        --max_episode_steps=200 \
        --critic_learning_rate=1e-3 \
        --actor_learning_rate=1e-4 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --action_scale=1e-3 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_fc_scale=256 \
        --qnetwork_fc_scale=256 \
        --exploration_noise=0.01 \
        --decay_rate=0.9993 \
        --writer_interval=1 \
        $VARIANT &  # Run in background
done

wait