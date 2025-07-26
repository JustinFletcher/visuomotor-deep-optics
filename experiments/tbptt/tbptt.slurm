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

    "--gpu_list=0 --exp_name="norm-img256-tbptt-tau0.0005-actorlr3e-4-criticlr5e-4-channels32-fc512""
    "--gpu_list=1 --exp_name="norm-img256-tbptt-tau0.0005-actorlr3e-4-criticlr5e-4-channels32-fc512""
    "--gpu_list=2 --exp_name="norm-img256-tbptt-tau0.0005-actorlr3e-4-criticlr5e-4-channels32-fc512""
    "--gpu_list=3 --exp_name="norm-img256-tbptt-tau0.0005-actorlr3e-4-criticlr5e-4-channels32-fc512""

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
        --focal_plane_image_size_pixels=256 \
        --observation_mode=image_only \
        --command_secondaries \
        --init_differential_motion \
        --model_wind_diff_motion \
        --num_envs=1 \
        --reward_function=align \
        --batch_size=64 \
        --tau=0.0005 \
        --policy_frequency=2 \
        --reward_scale=1.0 \
        --max_grad_norm=100_000 \
        --buffer_size=1_000_000 \
        --learning_starts=50_000 \
        --experience_sampling_delay=0_000 \
        --actor_training_delay=5_000 \
        --actor_type=impala \
        --critic_type=impala \
        --actor_learning_rate=1e-3 \
        --critic_learning_rate=1e-3 \
        --qnetwork_channel_scale=64 \
        --actor_channel_scale=64 \
        --actor_fc_scale=1024 \
        --qnetwork_fc_scale=1024 \
        --action_scale=1.0 \
        --use_q_bias \
        --normalize_returns \
        --exploration_noise=0.001 \
        --decay_rate=0.99 \
        --writer_interval=100 \
        --save_model \
        --tbptt_seq_len=64 \
        --num_eval_rollouts=0 \
        --model_save_interval=10_000 \
        $VARIANT &  # Run in background
done

wait