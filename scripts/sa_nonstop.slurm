#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=4
#SBATCH --exclude=cr2arm[1-2],cr3arm[3-4],cr4arm[5-8],cr3cn[5-7]
#SBATCH --output=sa.out
#SBATCH --error=sa.err
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --time=7-00:00:00                # Maximum runtime (D-HH:MM:SS)

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
i=0

while true; do
  echo "Batch $i: launching 4 parallel runs"
  
  for n in {1..4}; do
    sleep 1  # Sleep for 1 second before starting each process  
    echo "  Starting process $n in batch $i"
    poetry run python optomech/multi_sa.py \
      --eval_save_path="$WORKDIR/sa_${short_uuid}" \
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
      --num_processes=1 &  # Each subprocess uses 1 process
  done

  wait  # Wait for all 16 background processes to finish

  ((i++))

  echo "Batch $i completed. Waiting for 60 seconds before starting the next batch..."

  sleep 60
done

# sbatch 
# merge
