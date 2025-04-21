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
    --env-id VisualPendulum-v1 \
    --num_envs=1 \
    --batch_size=32 \
    --tau=0.0001 \
    --policy_frequency=2 \
    --l2_reg=1e-3 \
    --reward_function=strehl \
    --reward_scale=1.0 \
    --max_grad_norm=50.0 \
    --buffer_size=1_000_000 \
    --learning_starts=50_000 \
    --actor_training_delay=10_000 \
    --max_episode_steps=300 \
    --actor_type=impala \
    --critic_type=impala \
    --actor_learning_rate=1e-3 \
    --critic_learning_rate=1e-3 \
    --qnetwork_channel_scale=16 \
    --actor_channel_scale=16 \
    --actor_fc_scale=256 \
    --qnetwork_fc_scale=256 \
    --action_scale=1.0 \
    --exploration_noise=0.1 \
    --decay_rate=0.999 \
    --target_smoothing \
    --writer_interval=1 \
    $VARIANT &  # Run in background
done

wait