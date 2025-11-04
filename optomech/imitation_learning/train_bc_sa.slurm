#!/bin/bash
#SBATCH --job-name=bc_simple          # Job name
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
    "--run-name perfect_inc_8 --device cuda:0 --seed 8"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/imitation_learning/train_behavior_cloning.py \
        --dataset-path $WORKDIR/datasets/sa_dataset_100k \
        --pretrained-encoder $WORKDIR/models/trained_autoencoder.pth \
        --num-epochs 1000 \
        --batch-size 1024 \
        --learning-rate 1e-4 \
        --optimizer adamw \
        --loss-function log_cosh \
        --no-rollouts \
        --no-dataparallel \
        --force-incremental-mode \
        --target-action perfect_incremental_actions \
        --num-workers 16 \
       $VARIANT &  # Run in background
done


wait

