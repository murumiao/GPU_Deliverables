#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --job-name=CPU
#SBATCH --output=GPU_Deliverables/deliverable1/stdout/CPU-%j.out
#SBATCH --error=GPU_Deliverables/deliverable1/stderr/CPU-%j.err

PATH_SCRIPT1="GPU_Deliverables/deliverable1/bin/cpu_csr"
PATH_SCRIPT2="GPU_Deliverables/deliverable1/bin/cpu_coo"
PATH_MATRIX="$1"

$PATH_SCRIPT1 $PATH_MATRIX $MODE
$PATH_SCRIPT2 $PATH_MATRIX $MODE