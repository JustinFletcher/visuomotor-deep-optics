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

    "--gpu_list=0 --policy_frequency=2 --exploration_noise=1.0 --use_q_bias --normalize_returns --exp_name=vp-biasnorm-noise1.0-pf2-200kstart"
    "--gpu_list=1 --policy_frequency=4 --exploration_noise=1.0 --use_q_bias --normalize_returns --exp_name=vp-biasnorm-noise1.0-pf4-200kstart"
    "--gpu_list=2 --policy_frequency=2 --exploration_noise=0.1 --use_q_bias --normalize_returns --exp_name=vp-biasnorm-noise0.1-pf2-200kstart"
    "--gpu_list=3 --policy_frequency=4 --exploration_noise=0.1 --use_q_bias --normalize_returns --exp_name=vp-biasnorm-noise0.1-pf4-200kstart"
    "--gpu_list=4 --policy_frequency=2 --exploration_noise=1.0 --exp_name=vp-nobiasnorm-noise1.0-pf4-200kstart"
    "--gpu_list=5 --policy_frequency=4 --exploration_noise=1.0 --exp_name=vp-nobiasnorm-noise1.0-pf2-200kstart"
    "--gpu_list=6 --policy_frequency=2 --exploration_noise=0.1 --exp_name=vp-nobiasnorm-noise0.1-pf2-200kstart"
    "--gpu_list=7 --policy_frequency=4 --exploration_noise=0.1 --exp_name=vp-nobiasnorm-noise0.1-pf4-200kstart"

)

cd ~/visuomotor-deep-optics/
# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    poetry run python ./optomech/ddpg_manual_tbptt_lstm.py \
        --env-id VisualPendulum-v1 \
        --save_dir=./runs/test \
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
        --gamma=0.99 \
        --l2_reg=0.0 \
        --l1_reg=0.0 \
        --tau=0.0001 \
        --reward_scale=0.1 \
        --max_grad_norm=1.0 \
        --buffer_size=1_000_000 \
        --learning_starts=200_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=0_000 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-4 \
        --critic_learning_rate=3e-4 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --actor_fc_scale=512 \
        --qnetwork_fc_scale=512 \
        --lstm_hidden_dim=64 \
        --action_scale=2.0 \
        --decay_rate=0.995 \
        --writer_interval=100 \
        --target_smoothing \
        --tbptt_seq_len=1 \
        --save_model \
        --num_eval_rollouts=1 \
        --model_save_interval=10_000 \
        $VARIANT &  # Run in background
done

wait