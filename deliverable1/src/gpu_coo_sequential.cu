#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./TIMER_LIB/my_time_lib.h"

#define MAX_STRING 100
#define NUMBER_ITERATIONS 10
#define WARMUP_ITERATIONS 2

#define TIMED_ITERS NUMBER_ITERATIONS - WARMUP_ITERATIONS

void readMatrixFile(char* filePath, int** ARow, int** ACol, double** AVal, int* n_row, int* n_col, int* n_value) {
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL) {
        fprintf(stderr, "File %s not found", filePath);
        exit(1);
    }
    char buffer[MAX_STRING];
    int i = 0;
    while (fgets(buffer, MAX_STRING, fp)) {
        if (buffer[0] == '%') continue;
        if (buffer[0] == ' ') continue;
        if (buffer[0] == '\n') continue;
        if (*n_row == -1 || *n_col == -1 || *n_value == -1) {
            char* token = strtok(buffer, " ");
            *n_row = atoi(token);
            token = strtok(NULL, " ");
            *n_col = atoi(token);
            token = strtok(NULL, " ");
            *n_value = atoi(token);
            cudaMallocManaged(ARow, *n_value * sizeof(int));
            cudaMallocManaged(ACol, *n_value * sizeof(int));
            cudaMallocManaged(AVal, *n_value * sizeof(double));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            double val = atof(token);
            (*ARow)[i] = row;
            (*ACol)[i] = col;
            (*AVal)[i] = val;
            i++;
        }
    }
    fclose(fp);
}

__global__ void coo_spmv_sequential(int* ARow, int* ACol, double* AVal, double* v, int nnz, double* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (thread_id >= nnz) {
        return;
    }
    int nnz_per_thread = (nnz + total_threads - 1) / total_threads;
    int start = thread_id * nnz_per_thread;
    int end = min(start + nnz_per_thread, nnz);

    for (int i = start; i < end; i++) {
        if (i < nnz) {
            double product = AVal[i] * v[ACol[i]];
            atomicAdd(&result[ARow[i]], product);
        }
    }
}
__global__ void coo_spmv_stride(int* ARow, int* ACol, double* AVal, double* v, int nnz, double* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (thread_id >= nnz) {
        return;
    }
    for (int i = thread_id; i < nnz; i += total_threads) {
        double product = AVal[i] * v[ACol[i]];
        atomicAdd(&result[ARow[i]], product);
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
    if (argc != 3) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode[0=seq,X=stride]>\n", argv[0]);
        exit(1);
    }

    int mode = atoi(argv[2]);
    //* COO storage
    int n_row = -1, n_col = -1, n_value = -1;
    int *ARow = NULL, *ACol = NULL;
    double* Aval = NULL;

    printf("Loading matrix...\n");
    readMatrixFile(argv[1], &ARow, &ACol, &Aval, &n_row, &n_col, &n_value);
    printf("Matrix loaded!\n");

    // Create dense vector
    double* v;
    cudaMallocManaged(&v, n_col * sizeof(double));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    double* result;
    cudaMallocManaged(&result, n_value * sizeof(double));
    double timers[TIMED_ITERS];

    int threads_per_block = 256;
    int blocks_per_grid = (n_value + threads_per_block - 1) / threads_per_block;
    // int blocks_per_grid = 10;
    TIMER_DEF(0);
    for (int i = 0; i < NUMBER_ITERATIONS; i++) {
        cudaMemset(result, 0, n_row * sizeof(double));
        TIMER_START(0);
        if (mode == 0) {
            printf("Running Sequential\n");
            coo_spmv_sequential<<<threads_per_block, blocks_per_grid>>>(ARow, ACol, Aval, v, n_value, result);
        } else {
            printf("Running Stride\n");
            coo_spmv_stride<<<threads_per_block, blocks_per_grid>>>(ARow, ACol, Aval, v, n_value, result);
        }
        cudaDeviceSynchronize();

        if (strcmp(argv[1], "test.mtx") == 0) checkCorrect(result, n_col);
        TIMER_STOP(0);
        if (i > WARMUP_ITERATIONS) {
            timers[i - WARMUP_ITERATIONS] = TIMER_ELAPSED(0) / 1.e6;
        }
        printf("Run %d/%d done in %fs\n", i + 1, NUMBER_ITERATIONS, TIMER_ELAPSED(0) / 1.e6);
    }

    printf("Arithmetic mean on %d iterations: %fs\n", TIMED_ITERS, arithmetic_mean(timers, TIMED_ITERS));
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(Aval);
    cudaFree(v);
    return 0;
}