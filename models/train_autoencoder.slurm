#!/bin/bash
#SBATCH --job-name=ae-training        # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./autoencoder_training.out
#SBATCH --error=./autoencoder_training.err
#SBATCH --gres=gpu:A100-SXM4:4        # Request 4 A100 GPUs
#SBATCH --time=3-00:00:00             # Maximum runtime (3 days)

# Print job information
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start Time: $(date)"
echo "============================================"

# Change to the project directory
cd ~/visuomotor-deep-optics/

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Define dataset path using WORKDIR
DATASET_PATH="${WORKDIR}/datasets/sa_dataset_100k"

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

echo "Dataset path: $DATASET_PATH"
echo "============================================"

# Training configuration variants
# You can add multiple variants to run different experiments
VARIANTS=(
    "--arch autoencoder_resnet --latent-dim 256 --batch-size 64 --num-epochs 50"
    # Add more variants here if needed:
    # "--arch autoencoder_resnet --latent-dim 512 --batch-size 32 --num-epochs 50"
    # "--arch autoencoder_cnn --latent-dim 256 --batch-size 64 --num-epochs 50"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    
    echo "============================================"
    echo "Starting training variant: $VARIANT"
    echo "============================================"
    
    # Launch autoencoder training
    poetry run python models/train_autoencoder.py \
        --dataset-path "$DATASET_PATH" \
        --input-crop-size 256 \
        --learning-rate 1e-4 \
        --optimizer adamw \
        --loss-function mse \
        --log-scale \
        --device cuda \
        --seed 42 \
        $VARIANT &  # Run in background
    
    # Get the PID of the background process
    PIDS+=($!)
    
done

# Wait for all background processes to complete
echo "============================================"
echo "Waiting for all training jobs to complete..."
echo "============================================"

for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Process $PID exited with code $EXIT_CODE"
    fi
done

echo "============================================"
echo "All training jobs completed"
echo "End Time: $(date)"
echo "============================================"
