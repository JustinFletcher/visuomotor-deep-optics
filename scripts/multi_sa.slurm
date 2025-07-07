#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)

cd visuomotor-deep-optics/

poetry run python optomech/multi_sa.py \
    --eval_save_path=$WORKDIR/output_buffers_2 \
    --env-id optomech-v1 \
    --object_type=single \
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
    --max_episode_steps=3_000 \
    --num_processes=10  &  # Run in background

wait

# sbatch 
# merge
