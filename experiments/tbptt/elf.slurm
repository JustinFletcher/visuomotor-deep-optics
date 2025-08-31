#!/bin/bash
#SBATCH --job-name=tbptt           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       #
#SBATCH --output=./tbptt_elf.out
#SBATCH --error=./tbptt_elf.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=7-00:00:00                # Maximum runtime (D-HH:MM:SS)


# fdsgdfgs #SBATCH --nodelist=cr4mi2        # 
# fdsgdfgs #SBATCH --output=$LOCALWORKDIR/tbptt.out
# fdsgdfgs #SBATCH --error=$LOCALWORKDIR/tbptt.err
# This is a change git

# Open questions.
# Does batch size matter? What's the max at 512 and 256? rather would leave this be for now.
# Does q_bias matter? - This experiment is 5667
# What's the right tau? - trying 005 in 5667
# 0.999 noise decal is too slow.
# Should I have the LSTM directly in the loop?


VARIANTS=(
    "--gpu_list=0 --exp_name="0_optomech_longrun_direct""
    "--gpu_list=1 --incremental_control --exp_name="1_optomech_longrun_incremental""
)

cd ~/visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_manual_tbptt_lstm.py \
        --seed=9758 \
        --env-id optomech-v1 \
        --focal_plane_image_size_pixels=64 \
        --object_type=single \
        --max_episode_steps=1024 \
        --ao_interval_ms=5.0 \
        --control_interval_ms=5.0 \
        --frame_interval_ms=5.0 \
        --decision_interval_ms=5.0 \
        --num_atmosphere_layers=0 \
        --aperture_type=nanoelfplus \
        --observation_mode=image_only \
        --command_secondaries \
        --init_differential_motion \
        --model_wind_diff_motion \
        --num_envs=1 \
        --reward_function=align \
        --batch_size=8 \
        --gamma=0.99 \
        --tau=0.0001 \
        --policy_frequency=2 \
        --reward_scale=0.1 \
        --clip_gradients \
        --max_grad_norm=10000.0 \
        --buffer_size=1_000_000 \
        --learning_starts=200_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=0 \
        --actor_type=impala \
        --critic_type=impala \
        --critic_learning_rate=1e-3 \
        --actor_learning_rate=1e-5 \
        --qnetwork_channel_scale=32 \
        --actor_channel_scale=32 \
        --actor_fc_scale=128 \
        --qnetwork_fc_scale=128 \
        --lstm_hidden_dim=128 \
        --action_scale=1.0 \
        --exploration_noise=0.00001 \
        --decay_rate=0.99 \
        --writer_interval=1 \
        --tbptt_seq_len=8 \
        --target_smoothing \
        --save_model \
        --num_eval_rollouts=0 \
        --model_save_interval=100_000 \
        $VARIANT &  # Run in background
done

wait