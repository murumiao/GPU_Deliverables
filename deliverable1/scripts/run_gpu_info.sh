#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --job-name=INFO
#SBATCH --output=GPU_Deliverables/deliverable1/stdout/info-%j.out
#SBATCH --error=GPU_Deliverables/deliverable1/stderr/info-%j.err


module load CUDA/11.8.0
GPU_Deliverables/deliverable1/bin/gpu_info