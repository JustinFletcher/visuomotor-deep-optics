#!/bin/bash
#SBATCH --job-name=sml-training       # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28           # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       # Account/project name
#SBATCH --output=./sml_1m.out
#SBATCH --error=./sml_1m.err
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=4-00:00:00               # Maximum runtime (7 days)



# Change to the project directory
cd ~/visuomotor-deep-optics/

VARIANTS=(
    "--device cuda:0 --run_name sml_resnet_gn_1m_multi_0_cropped"
    "--device cuda:1 --run_name sml_resnet_gn_1m_multi_1_cropped"
    "--device cuda:2 --run_name sml_resnet_gn_1m_multi_2_cropped"
    "--device cuda:3 --run_name sml_resnet_gn_1m_multi_3_cropped"
    "--device cuda:4 --run_name sml_resnet_gn_1m_multi_4_cropped"
    "--device cuda:5 --run_name sml_resnet_gn_1m_multi_5_cropped"
    "--device cuda:6 --run_name sml_resnet_gn_1m_multi_6_cropped"
    "--device cuda:7 --run_name sml_resnet_gn_1m_multi_7_cropped"
)

# Run all variants in parallel
for VARIANT in "${VARIANTS[@]}"; do

    # Launch supervised ML training
    poetry run python optomech/supervised_ml/train_sml_model.py \
        --dataset_path /wdata/scratch/fletch/datasets/optomech-1m/optomech-1m \
        --model_arch sml_resnet_gn \
        --batch_size 4096 \
        --learning_rate 1e-3 \
        --num_epochs 500 \
        --log_dir ./runs/ \
        --no_dataparallel \
        --enable_rollouts \
        --max_examples 1_000_000 \
        --center_crop_size 128 \
        $VARIANT &  # Run in background
done


wait

