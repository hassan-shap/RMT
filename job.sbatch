#!/bin/bash
#SBATCH --partition=nc24-high
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=LNtrainingdata
#SBATCH --output=log-training-%J.txt
#SBATCH --exclusive


## run a code 
python3 generate_training.py > log_N_12.txt &
