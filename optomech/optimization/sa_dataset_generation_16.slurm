#!/bin/bash
#SBATCH --job-name=sadatagen     # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./sa_dataset_1m.out
#SBATCH --error=./sa_dataset_1m.err
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=5-00:00:00             # Maximum runtime (5 days)

# Change to the project directory
cd ~/visuomotor-deep-optics/

# Limit each process to use only sequential operations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# SA Dataset Generation Variants
VARIANTS=(
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
    "--num-samples 1000000 --dataset-name sa_dataset_1m"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do
    # Launch SA dataset generation
    poetry run python optomech/optimization/build_sa_dataset.py \
        --job-config-file optomech/optimization/sa_dataset_config.json \
        --dataset-save-path /wdata/scratch/fletch/datasets \
        --dataset-name sa_dataset_tuning \
        $VARIANT &
done

wait




    poetry run python optomech/optimization/build_sa_dataset.py \
        --job-config-file optomech/optimization/sa_dataset_config.json \
        --dataset-save-path ./datasets \
        --dataset-name sa_dataset_tuning \
