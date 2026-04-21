#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100
#define TOTAL_RUNS 10
#define WARMUP_RUNS 2

#define TIMED_RUNS TOTAL_RUNS - WARMUP_RUNS

void readMatrixFile(char* filePath, int** ARow, int** ACol, dtype** AVal, int* n_row, int* n_col, int* n_value) {
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
            cudaMallocManaged(AVal, *n_value * sizeof(dtype));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            dtype val = atof(token);
            (*ARow)[i] = row;
            (*ACol)[i] = col;
            (*AVal)[i] = val;
            i++;
        }
    }
    fclose(fp);
}

__global__ void coo_spmv_sequential(int* ARow, int* ACol, dtype* AVal, dtype* v, int nnz, dtype* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (thread_id >= nnz) {
        return;
    }
    int nnz_per_thread = (nnz + total_threads - 1) / total_threads;
    int start = thread_id * nnz_per_thread;
    int end = min(start + nnz_per_thread, nnz);

    if (start < nnz) {
        for (int i = start; i < end; i++) {
            if (i < nnz) {
                dtype product = AVal[i] * v[ACol[i]];
                atomicAdd(&result[ARow[i]], product);
            }
        }
    }
}
__global__ void coo_spmv_stride(int* ARow, int* ACol, dtype* AVal, dtype* v, int nnz, dtype* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (thread_id >= nnz) {
        return;
    }
    for (int i = thread_id; i < nnz; i += total_threads) {
        dtype product = AVal[i] * v[ACol[i]];
        atomicAdd(&result[ARow[i]], product);
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
    dtype* Aval = NULL;

    readMatrixFile(argv[1], &ARow, &ACol, &Aval, &n_row, &n_col, &n_value);
    if (mode == 0) {
        print_starting_info("COO GPU SEQUENTIAL", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else {
        print_starting_info("COO GPU STRIDE", argv[1], TIMED_RUNS, WARMUP_RUNS);
    }
    // Create dense vector
    dtype* v;
    cudaMallocManaged(&v, n_col * sizeof(dtype));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    dtype* result;
    cudaMallocManaged(&result, n_value * sizeof(dtype));
    double timer_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];

    int threads_per_block = 256;
    int blocks_per_grid = (n_value + threads_per_block - 1) / threads_per_block;
    cudaEvent_t start, stop;
    for (int i = 0; i < TOTAL_RUNS; i++) {
        cudaMemset(result, 0, n_row * sizeof(dtype));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        if (mode == 0) {
            coo_spmv_sequential<<<4, 256>>>(ARow, ACol, Aval, v, n_value, result);
        } else {
            coo_spmv_stride<<<blocks_per_grid, threads_per_block>>>(ARow, ACol, Aval, v, n_value, result);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start, stop);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = coo_calculate_bandwidthGBs(n_col, n_row, n_value, exec_time_s);
        double gflop = coo_calculate_gflop(n_value, exec_time_s);
        if (i > WARMUP_RUNS) {
            timer_arr[i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[i - WARMUP_RUNS] = bandwidth;
            gflops_arr[i - WARMUP_RUNS] = gflop;
        }
        print_run_stat(i, exec_time_s, bandwidth, gflop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    final_info_print(timer_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, result, n_row);
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(Aval);
    cudaFree(v);
    return 0;
}