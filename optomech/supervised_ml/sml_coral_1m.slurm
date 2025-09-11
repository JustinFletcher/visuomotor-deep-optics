#!/bin/bash
#SBATCH --job-name=sml-1m-dataset    # Job name
#SBATCH --partition=standard         # Partition or queue name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --account=MHPCC96650DAS      # Account
#SBATCH --output=$WORKDIR/sml_1m.out  # Standard output
#SBATCH --error=$WORKDIR/sml_1m.err   # Standard error
#SBATCH --time=24:00:00              # Maximum runtime (24 hours)

# SML Dataset Generation Job for Coral HPC
# Purpose: Generate 1M sample optical correction dataset
# Multiple identical jobs can be launched to contribute to the same dataset

echo "========================================"
echo "SML Dataset Generation Job Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Environment setup
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd ~/visuomotor-deep-optics/

# Ensure output directories exist
mkdir -p $WORKDIR/datasets
mkdir -p $WORKDIR/datasets/optomech-1m

echo "Working directory: $(pwd)"
echo "Dataset target: $WORKDIR/datasets/optomech-1m"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPU cores: $SLURM_CPUS_PER_TASK"

# Create job-specific config for 1M dataset
# Each job contributes 500 samples (2000 jobs total for 1M samples)
cat > /tmp/sml_1m_config_${SLURM_JOB_ID}.json << EOF
{
  "total_samples": 1000000,
  "samples_per_job": 500,
  "write_frequency": 100,
  "max_workers": 4,
  "timeout_minutes": 1440,
  "base_dataset_dir": "$WORKDIR/datasets",
  "dataset_name": "optomech-1m",
  "env_id": "optomech-v1",
  "object_type": "single",
  "aperture_type": "elf",
  "reward_function": "align",
  "observation_mode": "image_only",
  "focal_plane_image_size_pixels": 256,
  "environment_flags": [
    "--ao_interval_ms=5.0",
    "--control_interval_ms=5.0",
    "--frame_interval_ms=5.0",
    "--decision_interval_ms=5.0",
    "--num_atmosphere_layers=0",
    "--aperture_type=elf",
    "--focal_plane_image_size_pixels=256",
    "--observation_mode=image_only",
    "--command_secondaries",
    "--init_differential_motion",
    "--model_wind_diff_motion",
    "--reward_function=align",
    "--discrete_control=False",
    "--incremental_control=False",
    "--command_tensioners=False",
    "--command_tip_tilt=False",
    "--command_dm=False",
    "--ao_loop_active=False",
    "--simulate_differential_motion=False",
    "--model_gravity_diff_motion=False",
    "--model_temp_diff_motion=False",
    "--randomize_dm=False"
  ]
}
EOF

echo "Generated job config:"
cat /tmp/sml_1m_config_${SLURM_JOB_ID}.json

echo "========================================"
echo "Starting SML dataset generation..."
echo "Target: 500 samples per job"
echo "========================================"

# Launch the SML job manager
poetry run python optomech/supervised_ml/sml_job_manager_simple.py \
    --config /tmp/sml_1m_config_${SLURM_JOB_ID}.json \
    --total_samples 500 \
    --dataset_name optomech-1m

# Capture exit status
EXIT_STATUS=$?

echo "========================================"
echo "SML Dataset Generation Job Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit status: $EXIT_STATUS"
echo "End time: $(date)"
echo "========================================"

# Check if dataset was created successfully
if [ -d "$WORKDIR/datasets/optomech-1m" ]; then
    echo "Dataset directory exists"
    EPISODE_COUNT=$(find $WORKDIR/datasets/optomech-1m -name "episode_*.json" | wc -l)
    echo "Episode files created: $EPISODE_COUNT"
    
    # Show disk usage
    echo "Dataset size:"
    du -sh $WORKDIR/datasets/optomech-1m
else
    echo "WARNING: Dataset directory not found!"
fi

# Clean up temporary config
rm -f /tmp/sml_1m_config_${SLURM_JOB_ID}.json

exit $EXIT_STATUS
