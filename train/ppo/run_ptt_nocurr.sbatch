#!/bin/bash
#SBATCH --job-name=ptt-nocurr
#SBATCH --time=24:00:00
#SBATCH --account=MHPCC38870258
#SBATCH --partition=standard
#SBATCH --nodelist=cn[01-15]
#SBATCH --gres=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /p/home/fletch/visuomotor-deep-optics
poetry run python train/ppo/train_ppo_nanoelf_ptt_nocurr.py --hpc --no-eval
