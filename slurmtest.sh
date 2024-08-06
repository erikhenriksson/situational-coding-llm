#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=project_2010911
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --partition=gpusmall
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source venv/bin/activate
srun python3 "$@"
