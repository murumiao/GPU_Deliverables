#ifndef DELIVERABLE1_H
#define DELIVERABLE1_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

void readMatrixFile(char* filePath, int** rowPtr, int** colIndexes, dtype** valCSR, int* n_row, int* n_col, int* nnz);

__global__ void csr_globmem_spmv_sequential(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result);
__global__ void csr_globmem_spmv_stride(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result);
__global__ void csr_sharmem_spmv_stride(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result);
__global__ void csr_sharmem_coalesced(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result);

#endif