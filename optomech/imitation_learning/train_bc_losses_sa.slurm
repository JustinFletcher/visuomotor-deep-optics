#!/bin/bash
#SBATCH --job-name=bc_loss           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./ae_training.out
#SBATCH --error=./ae_training.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name incrollout_sa_logcosh  --target-action sa_incremental_action --loss-function log_cosh --device cuda:0"
    "--run-name incrollout_sa_huber  --target-action sa_incremental_action --loss-function huber --huber-delta 0.01 --device cuda:1"
    "--run-name incrollout_sa_adaptive_mse  --target-action sa_incremental_action --loss-function adaptive_mse --adaptive-scale 0.01 --device cuda:2"
    "--run-name incrollout_sa_logcosh  --target-action perfect_incremental_action --loss-function log_cosh --device cuda:3"
    "--run-name incrollout_sa_huber  --target-action perfect_incremental_action --loss-function huber --huber-delta 0.01 --device cuda:4"
    "--run-name incrollout_sa_adaptive_mse  --target-action perfect_incremental_action --loss-function adaptive_mse --adaptive-scale 0.01 --device cuda:5"
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
       $VARIANT &  # Run in background
done


wait

