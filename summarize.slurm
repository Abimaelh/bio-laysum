#!/bin/bash

#SBATCH --job-name=llama3_summarization
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=abimaelh@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=40G
#SBATCH --gpus=1 
#SBATCH --time=01:00:00

#SBATCH --chdir=/mmfs1/home/abimaelh/miniconda3/envs/llama3
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abimaelh
#SBATCH --error=/mmfs1/home/abimaelh

# Modules to use (optional).
conda activate llama3

# Your programs to run.
python summarize.py
