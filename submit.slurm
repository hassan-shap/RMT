#!/bin/bash
#SBATCH --partition=nc24-high
### SBATCH --array=1-70%500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --job-name=RMT_train
#SBATCH --output=out_%A_%a
#SBATCH --error=err_%A_%a
#SBATCH --no-requeue
export OMP_NUM_THREADS=20
export PYTHON_LOAD_PATH=/gscratch/home/hshapourian/RMT:$PYTHON_LOAD_PATH
export PARMFILE=/faithscratch/hshapourian`printf "%04d" $SLURM_ARRAY_TASK_ID`.jld
export OUTFILE=/faithscratch/hshapourian`printf "%04d" $SLURM_ARRAY_TASK_ID`.jld
echo $PARMFILE
echo $OUTFILE

/gscratch/home/hshapourian/miniconda3/bin/python3 /gscratch/home/hshapourian/RMT/generate_training.py

