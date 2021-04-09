#!/bin/bash
#SBATCH --partition=h16m-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=TPS
#SBATCH --output=out_%A_%a
#SBATCH --error=err_%A_%a
#SBATCH --no-requeue

## run a code
/gscratch/home/hshapourian/miniconda3/bin/python3 thermal_vs_Lc.py
