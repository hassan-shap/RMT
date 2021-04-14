#!/bin/bash
#SBATCH --partition=nc24-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --job-name=U1_vs_La
#SBATCH --output=out_%A_%a
#SBATCH --error=err_%A_%a
#SBATCH --no-requeue

## run a code
/gscratch/home/hshapourian/miniconda3/bin/python3 LN_U1_symm.py
