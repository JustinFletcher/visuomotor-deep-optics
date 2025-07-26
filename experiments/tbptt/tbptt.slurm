#!/bin/bash
#SBATCH --job-name=tbptt           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       #
#SBATCH --output=./tbptt.out
#SBATCH --error=./tbptt.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)


# fdsgdfgs #SBATCH --nodelist=cr4mi2        # 
# fdsgdfgs #SBATCH --output=$LOCALWORKDIR/tbptt.out
# fdsgdfgs #SBATCH --error=$LOCALWORKDIR/tbptt.err

VARIANTS=(

    "--gpu_list=0 --exp_name="tbptt-steps256""
    "--gpu_list=1 --exp_name="tbptt-steps256""
    "--gpu_list=2 --exp_name="tbptt-steps256""
    "--gpu_list=3 --exp_name="tbptt-steps256""

)

cd ~/visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_tbptt_lstm.py \
        --env-id optomech-v1 \
        --object_type=single \
        --max_episode_steps=512 \
        --ao_interval_ms=5.0 \
        --control_interval_ms=5.0 \
        --frame_interval_ms=5.0 \
        --decision_interval_ms=5.0 \
        --num_atmosphere_layers=0 \
        --aperture_type=elf \
        --focal_plane_image_size_pixels=64 \
        --observation_mode=image_only \
        --command_secondaries \
        --init_differential_motion \
        --model_wind_diff_motion \
        --num_envs=1 \
        --reward_function=align \
        --batch_size=32 \
        --tau=0.0003 \
        --policy_frequency=1 \
        --reward_scale=1.0 \
        --max_grad_norm=100_000 \
        --buffer_size=1_000_000 \
        --learning_starts=20_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=10_000 \
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
        --exploration_noise=0.0001 \
        --decay_rate=0.998 \
        --writer_interval=100 \
        --save_model \
        --tbptt_seq_len=64 \
        --num_eval_rollouts=0 \
        --model_save_interval=10_000 \
        $VARIANT &  # Run in background
done

wait