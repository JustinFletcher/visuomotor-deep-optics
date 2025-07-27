#!/bin/bash
#SBATCH --job-name=tbptt           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       #
#SBATCH --output=./tbptt.out
#SBATCH --error=./tbptt.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)


# fdsgdfgs #SBATCH --nodelist=cr4mi2        # 
# fdsgdfgs #SBATCH --output=$LOCALWORKDIR/tbptt.out
# fdsgdfgs #SBATCH --error=$LOCALWORKDIR/tbptt.err

VARIANTS=(

    "--gpu_list=0 --normalize_returns --exp_name="vp-norm""
    "--gpu_list=1 --normalize_returns --exp_name="vp-norm""
    "--gpu_list=2 --normalize_returns --exp_name="vp-norm""
    "--gpu_list=3 --normalize_returns --exp_name="vp-norm""
    "--gpu_list=4 --exp_name="vp""
    "--gpu_list=5 --exp_name="vp""
    "--gpu_list=6 --exp_name="vp""
    "--gpu_list=7 --exp_name="vp""

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
        --reward_scale=1.0 \
        --max_grad_norm=1.0 \
        --buffer_size=1_000_000 \
        --learning_starts=50_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=10_000 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-4 \
        --critic_learning_rate=1e-3 \
        --qnetwork_channel_scale=16 \
        --actor_channel_scale=16 \
        --actor_fc_scale=64 \
        --qnetwork_fc_scale=64 \
        --action_scale=2.0 \
        --use_q_bias \
        --exploration_noise=1.0 \
        --decay_rate=0.99 \
        --writer_interval=100 \
        --save_model \
        --tbptt_seq_len=32 \
        --num_eval_rollouts=0 \
        --model_save_interval=10_000 \
        $VARIANT &  # Run in background
done

wait