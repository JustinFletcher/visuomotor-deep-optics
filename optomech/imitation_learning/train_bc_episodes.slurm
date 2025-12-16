#!/bin/bash
#SBATCH --job-name=train-bc           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./bc_training.out
#SBATCH --error=./bc_training.err
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name bc_episode"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/imitation_learning/train_bc_episodes.py \
        --config optomech/imitation_learning/bc_config_sa_1m.json \
        --seed 42 \
       $VARIANT &  # Run in background
done


wait
