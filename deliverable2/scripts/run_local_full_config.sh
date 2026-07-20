NUMBER_PROCESSES=2
FILE_PATH="./bin/distributed_spmv"
MATRIX_PATH="./data/tmp2.mtx"
# MATRIX_PATH="./data/sorted_ASIC_680ks.mtx"

# N_BLOCKSS_CSR=(2 8 16 32 128 256)
N_THREADSS_CSR=(32 128 256 512 1024)
SHARED_MEM=(4096 16384 32768 49152)
MODE=0
N_PROCESS=(2 8 32 128 512 1024)

for N_THREADS in "${N_THREADSS_CSR[@]}"; do
    if [[ $MODE -eq 2  || $MODE -eq 4 ]]; then
        for SHAMEM in "${SHARED_MEM[@]}"; do
            if (( N_THREADS == 1024 && SHARED > 32768 )); then
                continue
            fi
            mpiexec -n $NUMBER_PROCESSES $FILE_PATH $MATRIX_PATH $MODE $N_THREADS $SHAMEM
        done
    else
            mpiexec -n $NUMBER_PROCESSES $FILE_PATH $MATRIX_PATH $MODE $N_THREADS 0
    fi
done