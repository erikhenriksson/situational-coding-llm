#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=project_2010911
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:2
#SBATCH --time=1:00:00

/users/ehenriks/bin/ollama serve > ollama_output.log 2>&1 &
source venv/bin/activate
srun python3 test.py