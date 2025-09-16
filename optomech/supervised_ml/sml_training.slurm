#!/bin/bash
#SBATCH --job-name=sml-training       # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28           # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./sml_1m.out
#SBATCH --error=./sml_1m.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=3-00:00:00               # Maximum runtime (24 hours)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--device cuda:0 --num_samples 100_000"
    "--device cuda:1 --num_samples 200_000"
    "--device cuda:2 --num_samples 400_000"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch supervised ML training
    poetry run python optomech/supervised_ml/train_sml_model.py \
        --dataset_path /wdata/scratch/fletch/datasets/optomech-1m/optomech-1m \
        --model_arch sml_resnet_gn \
        --batch_size 2048 \
        --learning_rate 1e-3 \
        --num_epochs 10000 \
        --log_dir ./runs/ \
        --no_dataparallel \
        $VARIANT &  # Run in background
done


wait

