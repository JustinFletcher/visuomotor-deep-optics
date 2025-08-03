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
#SBATCH --time=23:00:00                # Maximum runtime (D-HH:MM:SS)


# fdsgdfgs #SBATCH --nodelist=cr4mi2        # 
# fdsgdfgs #SBATCH --output=$LOCALWORKDIR/tbptt.out
# fdsgdfgs #SBATCH --error=$LOCALWORKDIR/tbptt.err
# This is a change git

VARIANTS=(

    "--gpu_list=0 --tbptt_seq_len=2 --use_q_bias --exp_name=vp-qbias-short"
    "--gpu_list=1 --tbptt_seq_len=2 --use_q_bias --exp_name=vp-qbias-short"
    "--gpu_list=2 --tbptt_seq_len=16 --use_q_bias --exp_name=vp-qbias-long"
    "--gpu_list=3 --tbptt_seq_len=16 --use_q_bias --exp_name=vp-qbias-long"
    "--gpu_list=4 --tbptt_seq_len=2 --exp_name=vp-noqbias-short"
    "--gpu_list=5 --tbptt_seq_len=2 --exp_name=vp-noqbias-short"
    "--gpu_list=6 --tbptt_seq_len=16 --exp_name=vp-noqbias-long"
    "--gpu_list=7 --tbptt_seq_len=16 --exp_name=vp-noqbias-long"

)

cd ~/visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_tbptt_lstm.py \
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
        --batch_size=16 \
        --tau=0.0001 \
        --policy_frequency=2 \
        --reward_scale=0.1 \
        --max_grad_norm=100.0 \
        --buffer_size=10_000 \
        --learning_starts=10_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=40_000 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-3 \
        --critic_learning_rate=1e-3 \
        --qnetwork_channel_scale=32 \
        --actor_channel_scale=32 \
        --actor_fc_scale=256 \
        --qnetwork_fc_scale=256 \
        --action_scale=2.0 \
        --exploration_noise=0.1 \
        --decay_rate=0.995 \
        --writer_interval=100 \
        --save_model \
        --num_eval_rollouts=10 \
        --model_save_interval=20_000 \
        $VARIANT &  # Run in background
done

wait