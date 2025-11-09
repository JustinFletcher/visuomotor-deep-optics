#!/bin/bash
#SBATCH --job-name=train-ae           # Job name
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
    "--arch autoencoder_resnet --latent-dim 256"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python models/train_autoencoder.py \
       --dataset-path $WORKDIR/datasets/sa_dataset_1m \
       --input-crop-size 256
       --batch-size 1024
       --learning-rate 1e-4
       --num-epochs 1000
       --optimizer adamw
       --loss-function mse
       --log-scale
       --load-in-memory
       --no-dataparallel
       --reconstruction-interval 10
       --checkpoint-interval 10
       --seed 42 
       $VARIANT &  # Run in background
done


wait
