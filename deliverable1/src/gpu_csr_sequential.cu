#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./TIMER_LIB/my_time_lib.h"

#define MAX_STRING 100
#define NUMBER_ITERATIONS 10
#define WARMUP_ITERATIONS 2

#define TIMED_ITERS NUMBER_ITERATIONS - WARMUP_ITERATIONS

void readMatrixFile(char* filePath, int** rowPtr, int** colIndexes, double** valCSR, int* n_row, int* n_col, int* nnz) {
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL) {
        fprintf(stderr, "File %s not found", filePath);
        exit(1);
    }
    char buffer[MAX_STRING];
    int i = 0;
    int last_row = -1;
    while (fgets(buffer, MAX_STRING, fp)) {
        if (buffer[0] == '%') continue;
        if (buffer[0] == ' ') continue;
        if (buffer[0] == '\n') continue;
        if (*n_row == -1 || *n_col == -1 || *nnz == -1) {
            char* token = strtok(buffer, " ");
            *n_row = atoi(token);
            token = strtok(NULL, " ");
            *n_col = atoi(token);
            token = strtok(NULL, " ");
            *nnz = atoi(token);

            cudaMallocManaged(rowPtr, (*n_row + 1) * sizeof(int));
            cudaMallocManaged(colIndexes, *nnz * sizeof(int));
            cudaMallocManaged(valCSR, *nnz * sizeof(double));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            double val = atof(token);
            (*valCSR)[i] = val;
            (*colIndexes)[i] = col;
            while (last_row < row) {
                last_row++;
                (*rowPtr)[last_row] = i;
            }
            i++;
        }
    }
    while (last_row < *n_row) {
        last_row++;
        (*rowPtr)[last_row] = i;
    }
    fclose(fp);
}

__global__ void csr_spmv(int* rowPtr, int* colIndexes, double* AVal, int rowLen, double* v, double* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= rowLen) {
        return;
    }

    int start = rowPtr[thread_id];
    int end = rowPtr[thread_id + 1];

    for (int i = start; i < end; i++) {
        double product = AVal[i] * v[colIndexes[i]];
        result[thread_id] += product;
    }
}

void checkCorrect(double* result, int n) {
    if (!(result[0] == 3.3 && result[1] == 2.5 && result[2] == 3)) {
        printf("Fail!\n");
        for (int j = 0; j < n; j++) {
            printf("%f,", result[j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    printf("CSR\n");
    if (argc != 2) {
        fprintf(stderr, "Usage %s <path_to_matrix>\n", argv[0]);
        exit(1);
    }

    //* CSR storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    double* AVal = NULL;

    printf("Loading matrix...\n");
    readMatrixFile(argv[1], &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
    printf("Matrix loaded!\n");

    // Create dense vector
    double* v;
    cudaMallocManaged(&v, n_col * sizeof(double));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    double* result;
    cudaMallocManaged(&result, n_row * sizeof(double));

    double timers[TIMED_ITERS];

    int threads_per_block = 256;
    int blocks_per_grid = (n_row + threads_per_block - 1) / threads_per_block;

    TIMER_DEF(0);
    for (int i = 0; i < NUMBER_ITERATIONS; i++) {
        cudaMemset(result, 0, n_row * sizeof(double));
        TIMER_START(0);
        csr_spmv<<<threads_per_block, blocks_per_grid>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        cudaDeviceSynchronize();
        TIMER_STOP(0);
        if (strcmp(argv[1], "test.mtx") == 0) checkCorrect(result, n_row);
        if (i > WARMUP_ITERATIONS) {
            timers[i - WARMUP_ITERATIONS] = TIMER_ELAPSED(0) / 1.e6;
        }
        printf("Run %d/%d done in %fs\n", i + 1, NUMBER_ITERATIONS, TIMER_ELAPSED(0) / 1.e6);
    }

    printf("Arithmetic mean on %d iterations: %fs\n", TIMED_ITERS, arithmetic_mean(timers, TIMED_ITERS));
    cudaFree(rowPtr);
    cudaFree(colIndexes);
    cudaFree(AVal);
    cudaFree(v);
    return 0;
}