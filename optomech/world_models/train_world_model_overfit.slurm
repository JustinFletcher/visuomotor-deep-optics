#!/bin/bash
#SBATCH --job-name=wm           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./wm_training.out
#SBATCH --error=./wm_training.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name world_model_experiment_overfit --seed 2"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/world_models/train_world_model.py \
        --config optomech/world_models/world_model_config_overfit.json \
        --runs-dir /wdata/scratch/fletch/runs \
        --max-examples 128 \
        --no-save \
        $VARIANT &  # Run in background
done


wait
