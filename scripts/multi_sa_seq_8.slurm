#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --time=1-00:00:00                # Maximum runtime (D-HH:MM:SS)

cd ~/visuomotor-deep-optics/

# Get the absolute path to this script
script_path="$(realpath "${BASH_SOURCE[0]}")"

# Hash this file's contents
hash=$(sha256sum "$script_path" | awk '{print $1}')

# Format the hash as a UUID (first 32 characters)
uuid="${hash:0:8}-${hash:8:4}-${hash:12:4}-${hash:16:4}-${hash:20:12}"

echo "UUID for this script ($script_path):"
echo "$uuid"

short_uuid=${uuid:0:8} 
i=1
while true; do
    echo "Number: $i"
    poetry run python optomech/multi_sa.py \
        --eval_save_path=$WORKDIR/buffer_pool_$short_uuid \
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
        --max_episode_steps=10_000 \
        --num_processes=8  # Run in background
    ((i++))
done

wait

# sbatch 
# merge
