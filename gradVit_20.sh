#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

# Make conda available: 
eval "$(conda shell.bash hook)"

# Activate a conda environment:
conda activate mjp     

python gradvit_eval_big_layer.py
