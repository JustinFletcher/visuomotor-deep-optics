#!/bin/bash
#SBATCH --job-name=wm           # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1              # <-- ONE task
#SBATCH --cpus-per-task=28      # <-- 28 CPUs for that one task
#SBATCH --account=MHPCC96650DAS       # Account/project namesd
#SBATCH --output=./wm_training.out
#SBATCH --error=./wm_training.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days) 



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--run-name world_model_experiment_sa_1m_mse --loss-function mse --seed 2 --device cuda:0"
    "--run-name world_model_experiment_sa_1m_mae --loss-function mae --seed 2 --device cuda:1"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch autoencoder training
    poetry run python optomech/world_models/train_world_model.py \
        --runs-dir /wdata/scratch/fletch/runs \
        --config optomech/world_models/world_model_config_sa_1m.json \
        $VARIANT &  # Run in background
done


wait
