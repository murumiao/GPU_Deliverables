#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01
#SBATCH --job-name=CSR
#SBATCH --output=GPU_Deliverables/deliverable1/stdout/CSR3/CSR-%j.out
#SBATCH --error=GPU_Deliverables/deliverable1/stderr/CSR3/CSR-%j.err

PATH_SCRIPT="GPU_Deliverables/deliverable1/bin/gpu_csr"
PATH_MATRIX="$1"
MODE="$2"

# N_BLOCKSS_CSR=(2 8 16 32 128 256)
N_THREADSS_CSR=(32 128 256 512 1024)
SHARED_MEM=(4096 16384 32768 49152)

module load CUDA/11.8.0

for N_THREADS in "${N_THREADSS_CSR[@]}"; do
    if [[ $MODE -eq 2  || $MODE -eq 4 ]]; then
        for SHAMEM in "${SHARED_MEM[@]}"; do
            if (( N_THREADS == 1024 && SHARED > 32768 )); then
                continue
            fi
            $PATH_SCRIPT $PATH_MATRIX $MODE 0 $N_THREADS $SHAMEM
        done
    else
        $PATH_SCRIPT $PATH_MATRIX $MODE 0 $N_THREADS 0
    fi
done
