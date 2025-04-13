#!/bin/bash
#SBATCH --job-name=omclip50           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(

    "--gpu_list=0 --action_scale=1e-2 --tau=0.0001 --actor_learning_rate=1e-4 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-4-as1e-2-tau1e-4""
    "--gpu_list=1 --action_scale=1e-2 --tau=0.0001 --actor_learning_rate=1e-4 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-4-as1e-2-tau1e-4""
    "--gpu_list=2 --action_scale=1e-2 --tau=0.0001 --actor_learning_rate=1e-5 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-5-as1e-2-tau1e-4""
    "--gpu_list=3 --action_scale=1e-2 --tau=0.0001 --actor_learning_rate=1e-5 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-5-as1e-2-tau1e-4""
    "--gpu_list=4 --action_scale=1.0 --tau=0.0005 --actor_learning_rate=1e-4 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-4-as1e-0-tau5e-4""
    "--gpu_list=5 --action_scale=1.0 --tau=0.0005 --actor_learning_rate=1e-4 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-4-as1e-0-tau5e-4""
    "--gpu_list=6 --action_scale=1.0 --tau=0.0005 --actor_learning_rate=1e-5 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-5-as1e-0-tau5e-4""
    "--gpu_list=7 --action_scale=1.0 --tau=0.0005 --actor_learning_rate=1e-5 --actor_type=impala --critic_type=impala --exp_name="optomech-nanoelf-clip50-lr1e-5-as1e-0-tau5e-4""
    
    

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
        --model_save_interval=100_000 \
        --num_envs=1 \
        --batch_size=32 \
        --gamma=0.99 \
        --l2_reg=0.0 \
        --reward_function=strehl \
        --reward_scale=1.0 \
        --max_grad_norm=50.0 \
        --buffer_size=1_000_000 \
        --learning_starts=50_000 \
        --actor_training_delay=100_000 \
        --policy_frequency=3 \
        --max_episode_steps=100 \
        --critic_learning_rate=1e-3 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --actor_fc_scale=256 \
        --qnetwork_fc_scale=256 \
        --exploration_noise=0.1 \
        --decay_rate=0.9993 \
        --writer_interval=100 \
        $VARIANT &  # Run in background
done

wait