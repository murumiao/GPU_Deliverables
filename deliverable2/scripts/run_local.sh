NUMBER_PROCESSES=4
FILE_PATH="./bin/distributed_spmv"
MATRIX_PATH="./data/tmp2.mtx"

mpiexec -n $NUMBER_PROCESSES $FILE_PATH $MATRIX_PATH