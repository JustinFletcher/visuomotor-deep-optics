#!/bin/bash
#SBATCH --job-name=vispend           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-SMX4:8
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)

VARIANTS=(

    "--gpu_list=0 --batch_size=32 --actor_fc_scale=256 --qnetwork_fc_scale=256 --exp_name="vispend-batch32-fc256""
    "--gpu_list=1 --batch_size=256 --actor_fc_scale=256 --qnetwork_fc_scale=256 --exp_name="vispend-batch256-fc256""
    "--gpu_list=2 --batch_size=32 --actor_fc_scale=512 --qnetwork_fc_scale=512 --exp_name="vispend-batch32-fc512""
    "--gpu_list=3 --batch_size=256 --actor_fc_scale=512 --qnetwork_fc_scale=512 --exp_name="vispend-batch256-fc512""
    "--gpu_list=4 --batch_size=32 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --exp_name="vispend-batch32-fc1024""
    "--gpu_list=5 --batch_size=256 --actor_fc_scale=1024 --qnetwork_fc_scale=1024 --exp_name="vispend-batch256-fc1024""
    "--gpu_list=6 --batch_size=32 --actor_fc_scale=2048 --qnetwork_fc_scale=2048 --exp_name="vispend-batch32-fc2048""
    "--gpu_list=7 --batch_size=256 --actor_fc_scale=2048 --qnetwork_fc_scale=2048 --exp_name="vispend-batch256-fc2048""

    

)

cd visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_lstm.py \
    --env-id VisualPendulum-v1 \
    --num_envs=1 \
    --tau=0.0005 \
    --policy_frequency=2 \
    --model_save_interval=100_000 \
    --l2_reg=1e-3 \
    --reward_function=strehl \
    --reward_scale=1.0 \
    --max_grad_norm=100_000.0 \
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
    --action_scale=2.0 \
    --exploration_noise=1.0 \
    --decay_rate=0.999 \
    --target_smoothing \
    --writer_interval=100 \
    $VARIANT &  # Run in background
done

wait