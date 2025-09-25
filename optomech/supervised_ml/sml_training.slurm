#!/bin/bash
#SBATCH --job-name=sml-training       # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28           # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./sml_1m.out
#SBATCH --error=./sml_1m.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=2-00:00:00               # Maximum runtime (2 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--device cuda:0 --run_name sml_resnet_gn_1m_cosine_warmup --scheduler cosine_warmup --initial_lr 1e-4 --max_lr 2e-3"
    "--device cuda:1 --run_name sml_resnet_gn_1m_plateau --scheduler plateau --initial_lr 2e-3"
    "--device cuda:2 --run_name sml_resnet_gn_1m_onecycle --scheduler onecycle --initial_lr 1e-4 --max_lr 1e-2"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch supervised ML training
    poetry run python optomech/supervised_ml/train_sml_model.py \
        --dataset_path /wdata/scratch/fletch/datasets/optomech-1m/optomech-1m \
        --model_arch sml_resnet_gn \
        --batch_size 4096 \
        --num_epochs 500 \
        --log_dir ./runs/ \
        --no_dataparallel \
        --enable_rollouts \
        --max_examples 1_000_000 \
        --center_crop_size 128 \
        --warmup_epochs 25 \
        --patience 25 \
        $VARIANT &  # Run in background
done


wait

