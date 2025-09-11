#!/bin/bash
#SBATCH --job-name=sml-1m-dataset    # Job name
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --nodelist=cr1dpu1,cr1dpu2,cr4mi1,cr4mi2,cr4mi3,cr4mi4
#SBATCH --ntasks-per-node=28          # Number of tasks per node
#SBATCH --account=MHPCC96650DAS       #
#SBATCH --output=./sml_1m.out
#SBATCH --error=./sml_1m.err
#SBATCH --time=7-00:00:00                # Maximum runtime (D-HH:MM:SS)

cd ~/visuomotor-deep-optics/

# Launch the SML dataset generation
poetry run python optomech/supervised_ml/sml_job_manager_simple.py \
    --config optomech/supervised_ml/sml_job_config.json \
    --total_samples 1000000 \
    --dataset_name optomech-1m \
    --dataset_dir $WORKDIR/datasets

