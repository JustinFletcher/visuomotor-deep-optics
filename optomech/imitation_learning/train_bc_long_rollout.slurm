#!/bin/bash
#SBATCH --job-name=train-ae           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./ae_training.out
#SBATCH --error=./ae_training.err
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name finetune_perfect_1000"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/imitation_learning/train_behavior_cloning.py \
        --dataset-path $WORKDIR/datasets/sa_dataset_100k \
        --pretrained-encoder $WORKDIR/models/trained_autoencoder.pth \
        --num-epochs 200 \
        --batch-size 1024 \
        --learning-rate 1e-4 \
        --enable-rollouts \
        --rollout-seeds 5 \
        --rollout-steps 1000 \
        --rollout-episode-steps 1000 \
        --no-dataparallel \
        --target-action sal_actions \
       $VARIANT &  # Run in background
done


wait

