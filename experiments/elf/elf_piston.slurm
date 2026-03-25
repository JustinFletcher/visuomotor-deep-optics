#!/bin/bash
#SBATCH --job-name=elf_piston        # Job name
#SBATCH --output=output_%j.txt       # Standard output file
#SBATCH --error=error_%j.txt         # Standard error file
#SBATCH --partition=standard         # Partition or queue name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --account=MHPCC96650DAS
#SBATCH --gres=gpu:A100-PCIe:1
#SBATCH --time=72:00:00              # Maximum runtime (D-HH:MM:SS)

cd visuomotor-deep-optics/

poetry run python train/ppo/train_ppo_elf_piston.py \
    --hpc --no-eval \
    --run-dir runs/elf_piston_$(date +%s)
