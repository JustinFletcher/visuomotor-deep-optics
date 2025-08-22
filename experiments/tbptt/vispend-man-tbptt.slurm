#!/bin/bash
#SBATCH --job-name=tbptt           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       #
#SBATCH --output=./tbptt.out
#SBATCH --error=./tbptt.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=48:00:00                # Maximum runtime (D-HH:MM:SS)


# fdsgdfgs #SBATCH --nodelist=cr4mi2        # 
# fdsgdfgs #SBATCH --output=$LOCALWORKDIR/tbptt.out
# fdsgdfgs #SBATCH --error=$LOCALWORKDIR/tbptt.err
# This is a change git

VARIANTS=(

    "--gpu_list=0 --exp_name="0""
    "--gpu_list=1 --exp_name="1""
    "--gpu_list=2 --exp_name="2""
    "--gpu_list=3 --exp_name="3""
    "--gpu_list=4 --exp_name="4""
    "--gpu_list=5 --exp_name="5""
    "--gpu_list=6 --exp_name="6""
    "--gpu_list=7 --exp_name="7""
)

cd ~/visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_manual_tbptt_lstm.py \
        --seed=9758 \
        --env-id VisualPendulum-v1 \
        --object_type=single \
        --max_episode_steps=256 \
        --ao_interval_ms=5.0 \
        --control_interval_ms=5.0 \
        --frame_interval_ms=5.0 \
        --decision_interval_ms=5.0 \
        --num_atmosphere_layers=0 \
        --aperture_type=elf \
        --observation_mode=image_only \
        --command_secondaries \
        --init_differential_motion \
        --model_wind_diff_motion \
        --num_envs=1 \
        --reward_function=align \
        --batch_size=128 \
        --gamma=0.99 \
        --tau=0.0001 \
        --policy_frequency=2 \
        --reward_scale=0.1 \
        --clip_gradients \
        --max_grad_norm=10000.0 \
        --buffer_size=1_000_000 \
        --learning_starts=10_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=0 \
        --actor_type=impala \
        --critic_type=impala \
        --critic_learning_rate=1e-3 \
        --actor_learning_rate=1e-5 \
        --qnetwork_channel_scale=512 \
        --actor_channel_scale=512 \
        --actor_fc_scale=1024 \
        --qnetwork_fc_scale=1024 \
        --lstm_hidden_dim=256 \
        --action_scale=2.0 \
        --exploration_noise=0.1 \
        --decay_rate=0.9 \
        --writer_interval=100 \
        --tbptt_seq_len=2 \
        --target_smoothing \
        --save_model \
        --num_eval_rollouts=5 \
        --model_save_interval=100_000 \
        $VARIANT &  # Run in background
done

wait