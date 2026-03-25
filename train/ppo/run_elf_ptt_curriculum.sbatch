#!/bin/bash
#SBATCH --job-name=elf-ptt-curr
#SBATCH --time=72:00:00
#SBATCH --account=MHPCC38870258
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /p/home/fletch/visuomotor-deep-optics
poetry run python train/ppo/train_ppo_elf_ptt_curriculum.py --hpc --no-eval
