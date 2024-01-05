#!/bin/bash

#SBATCH --job-name="CSC-4651 HF audio decorrelator"
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10

SCRIPT_NAME="CSC-4651 HF audio decorrelator"

echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun hostname; pwd; date;
srun singularity exec --nv -B /data:/data /data/containers/msoe-tensorflow-23.05-tf2-py3.sif python3 model.py
echo "END: " $SCRIPT_NAME

