#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100
#define TOTAL_RUNS 10
#define WARMUP_RUNS 1

#define TIMED_RUNS TOTAL_RUNS - WARMUP_RUNS

void readMatrixFile(char* filePath, int** rowPtr, int** colIndexes, dtype** valCSR, int* n_row, int* n_col, int* nnz) {
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
            cudaMallocManaged(valCSR, *nnz * sizeof(dtype));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            dtype val = atof(token);
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

__global__ void csr_globmem_spmv_sequential(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= rowLen) {
        return;
    }

    int start = rowPtr[thread_id];
    int end = rowPtr[thread_id + 1];

    for (int i = start; i < end; i++) {
        result[thread_id] += AVal[i] * v[colIndexes[i]];
    }
}

__global__ void csr_globmem_spmv_stride(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    int thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    // int total_threads = gridDim.x * blockDim.x;
    if (thread_id >= rowLen) {
        return;
    }
    int lane = threadIdx.x % warpSize;

    int start = rowPtr[thread_id];
    int end = rowPtr[thread_id + 1];

    dtype sum = 0.0;
    for (int i = start + lane; i < end; i += warpSize) {
        sum += AVal[i] * v[colIndexes[i]];
    }

    // reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        result[thread_id] = sum;
    }
}
__global__ void csr_sharmem_spmv_stride(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    extern __shared__ dtype vals[];

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane = thread_id & (32 - 1);
    int row = warp_id;
    vals[threadIdx.x] = 0.0;
    __syncthreads();

    if (row < rowLen) {
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        vals[threadIdx.x] = 0;
        for (int jj = row_start + lane; jj < row_end; jj += 32)
            vals[threadIdx.x] += AVal[jj] * v[colIndexes[jj]];
    }
    __syncthreads();
    if (lane < 16 && threadIdx.x + 16 < blockDim.x) {
        vals[threadIdx.x] += vals[threadIdx.x + 16];
    }
    __syncthreads();

    if (lane < 8 && threadIdx.x + 8 < blockDim.x) {
        vals[threadIdx.x] += vals[threadIdx.x + 8];
    }
    __syncthreads();

    if (lane < 4 && threadIdx.x + 4 < blockDim.x) {
        vals[threadIdx.x] += vals[threadIdx.x + 4];
    }
    __syncthreads();

    if (lane < 2 && threadIdx.x + 2 < blockDim.x) {
        vals[threadIdx.x] += vals[threadIdx.x + 2];
    }
    __syncthreads();

    if (lane < 1 && threadIdx.x + 1 < blockDim.x) {
        vals[threadIdx.x] += vals[threadIdx.x + 1];
    }
    __syncthreads();

    // Write result to global memory
    if (lane == 0 && row < rowLen) {
        atomicAdd(&result[row], vals[threadIdx.x]);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode[0=seq,1=stride]>\n", argv[0]);
        exit(1);
    }

    int mode = atoi(argv[2]);
    //* CSR storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    dtype* AVal = NULL;

    readMatrixFile(argv[1], &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
    if (mode == 0) {
        print_starting_info("CSR GPU SEQUENTIAL GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else if (mode == 1) {
        print_starting_info("CSR GPU STRIDE GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else if (mode == 2) {
        print_starting_info("CSR GPU STRIDE SHARED MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else {
        fprintf(stderr, "Wrong mode\n");
        exit(1);
    }

    // Create dense vector
    dtype* v;
    cudaMallocManaged(&v, n_col * sizeof(dtype));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    dtype* result;
    cudaMallocManaged(&result, n_row * sizeof(dtype));

    double timer_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];

    int threads_per_block = 256;
    int blocks_per_grid = 4;
    // int blocks_per_grid = (n_row + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    int shared_mem_size = 1024;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < TOTAL_RUNS; i++) {
        cudaMemset(result, 0, n_row * sizeof(dtype));

        cudaEventRecord(start);
        if (mode == 0) {
            csr_globmem_spmv_sequential<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 1) {
            csr_globmem_spmv_stride<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 2) {
            csr_sharmem_spmv_stride<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start, stop);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = csr_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
        double gflop = calculate_gflop(nnz, exec_time_s);
        if (i > WARMUP_RUNS) {
            timer_arr[i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[i - WARMUP_RUNS] = bandwidth;
            gflops_arr[i - WARMUP_RUNS] = gflop;
        }
        print_run_stat(i, exec_time_s, bandwidth, gflop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    final_info_print(timer_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, result, n_row);
    cudaFree(rowPtr);
    cudaFree(colIndexes);
    cudaFree(AVal);
    cudaFree(v);
    return 0;
}