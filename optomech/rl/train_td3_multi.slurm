#!/bin/bash
#SBATCH --job-name=om-td3     # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./om-td3.out
#SBATCH --error=./om-td3.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=4-00:00:00             # Maximum runtime (5 days) 

# Change to the project directory
cd ~/visuomotor-deep-optics/


# SA Dataset Generation Variants
VARIANTS=(
    "--seed 1 --gpu_list 0"
    "--seed 2 --gpu_list 1"
    "--seed 3 --gpu_list 2"
    "--seed 4 --gpu_list 3"
    "--seed 5 --gpu_list 4"
    "--seed 6 --gpu_list 5"
    "--seed 7 --gpu_list 6"
    "--seed 8 --gpu_list 7"

)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    # Launch SA dataset generation
    poetry run python optomech/rl/ddpg_tbptt_lstm.py \
        --config optomech/rl/test_frozen_encoder_config.json \
        --sa_dataset_path /wdata/scratch/fletch/datasets/sa_dataset_1m \
        --sa_max_sequences 2048 \
        $VARIANT &
done

wait



