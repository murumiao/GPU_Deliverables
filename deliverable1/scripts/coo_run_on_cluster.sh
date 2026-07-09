#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01
#SBATCH --job-name=COO
#SBATCH --output=GPU_Deliverables/deliverable1/stdout/COO2/COO-%j.out
#SBATCH --error=GPU_Deliverables/deliverable1/stderr/COO2/COO-%j.err

PATH_SCRIPT="GPU_Deliverables/deliverable1/bin/gpu_coo"
PATH_MATRIX="$1"
MODE="$2"

N_BLOCKSS_COO=(512 1024 2048 4096 8192 16384)
N_THREADSS_COO=(32 128 256 512 1024)
module load CUDA/11.8.0

for N_BLOCKS in "${N_BLOCKSS_COO[@]}"; do
    for N_THREADS in "${N_THREADSS_COO[@]}"; do
        $PATH_SCRIPT $PATH_MATRIX $MODE $N_BLOCKS $N_THREADS
    done
done