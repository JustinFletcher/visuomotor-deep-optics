#!/bin/bash
#SBATCH --job-name=sa-dataset-gen     # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./sa_dataset_1k,out
#SBATCH --error=./sa_dataset_1m.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=2-00:00:00             # Maximum runtime (2 days)

# Change to the project directory
cd ~/visuomotor-deep-optics/

# SA Dataset Generation Variants
VARIANTS=(
    "--total_samples 1000000 --dataset_name sa_dataset_1m"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch SA dataset generation
    poetry run python optomech/optimization/sa_job_manager_simple.py \
        --config optomech/optimization/sa_job_config.json \
        --dataset_dir /wdata/scratch/fletch/datasets/sa_dataset_1m \
        $VARIANT &  # Run in background
done

wait
