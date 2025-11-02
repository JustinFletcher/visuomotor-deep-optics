#!/bin/bash
#SBATCH --job-name=bc_simple          # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./bc_training.out
#SBATCH --error=./bc_training.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name sa_direct_0 --device cuda:0"
    "--run-name sa_direct_1 --device cuda:1"
    "--run-name sa_direct_2 --device cuda:2"
    "--run-name sa_direct_3 --device cuda:3"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/imitation_learning/train_behavior_cloning.py \
        --dataset-path $WORKDIR/datasets/sa_dataset_100k \
        --pretrained-encoder $WORKDIR/models/trained_autoencoder.pth \
        --num-epochs 1000 \
        --batch-size 64 \
        --learning-rate 1e-4 \
        --enable-rollouts \
        --rollout-interval 50 \
        --rollout-seeds 10 \
        --rollout-steps 1000 \
        --rollout-episode-steps 1000 \
        --no-dataparallel \
        --no-force-incremental-mode \
        --target-action sa_actions \
       $VARIANT &  # Run in background
done


wait

