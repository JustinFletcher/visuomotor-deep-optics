#!/bin/bash
#SBATCH --job-name=elf-boot
#SBATCH --time=72:00:00
#SBATCH --account=MHPCC38870258
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --array=0-14
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

cd /p/home/fletch/visuomotor-deep-optics
poetry run python train/ppo/train_ppo_elf_bootstrap.py --hpc --no-eval --phased-count $SLURM_ARRAY_TASK_ID
