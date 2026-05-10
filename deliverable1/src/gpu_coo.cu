#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100
#define TOTAL_RUNS 10
#define WARMUP_RUNS 1

#define TIMED_RUNS TOTAL_RUNS - WARMUP_RUNS

void readMatrixFile(char* filePath, int** ARow, int** ACol, dtype** AVal, int* n_row, int* n_col, int* nnz) {
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
        if (*n_row == -1 || *n_col == -1 || *nnz == -1) {
            char* token = strtok(buffer, " ");
            *n_row = atoi(token);
            token = strtok(NULL, " ");
            *n_col = atoi(token);
            token = strtok(NULL, " ");
            *nnz = atoi(token);
            cudaMallocManaged(ARow, *nnz * sizeof(int));
            cudaMallocManaged(ACol, *nnz * sizeof(int));
            cudaMallocManaged(AVal, *nnz * sizeof(dtype));
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

__global__ void coo_globmem_spmv_seq(int* ARow, int* ACol, dtype* AVal, dtype* v, int nnz, dtype* result) {
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
__global__ void coo_globmem_spmv_stride(int* ARow, int* ACol, dtype* AVal, dtype* v, int nnz, dtype* result) {
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

__global__ void coo_sharmem_spmv_seq(int* ARow, int* ACol, dtype* AVal, dtype* v, int nnz, dtype* result) {
    extern __shared__ dtype result_shared[];

    int tid = threadIdx.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int block_id = blockIdx.x;
    
    // Initialize shared memory
    result_shared[tid] = 0.0;
    __syncthreads();

    // Each thread processes a portion of non-zeros
    int nnz_per_thread = (nnz + total_threads - 1) / total_threads;
    int start = thread_id * nnz_per_thread;
    int end = min(start + nnz_per_thread, nnz);
    
    for (int idx = start; idx < end; idx++) {
        if (idx < nnz) {
            int row = ARow[idx];
            int col = ACol[idx];
            dtype val = AVal[idx];
            
            // Only accumulate if row maps to this block's range
            int row_block = row % gridDim.x;
            if (row_block == block_id) {
                int local_row = row / gridDim.x;
                if (local_row < blockDim.x) {
                    atomicAdd(&result_shared[local_row], val * v[col]);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write shared memory results back to global memory
    for (int i = tid; i < blockDim.x; i += blockDim.x) {
        if (result_shared[i] != 0.0) {
            int global_row = block_id + i * gridDim.x;
            atomicAdd(&result[global_row], result_shared[i]);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode[0=seq,1=stride,2=seq&shared]>\n", argv[0]);
        exit(1);
    }

    int mode = atoi(argv[2]);
    //* COO storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *ARow = NULL, *ACol = NULL;
    dtype* Aval = NULL;

    readMatrixFile(argv[1], &ARow, &ACol, &Aval, &n_row, &n_col, &nnz);
    if (mode == 0) {
        print_starting_info("COO GPU SEQUENTIAL GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else if (mode == 1) {
        print_starting_info("COO GPU STRIDE GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
    } else if (mode == 2) {
        print_starting_info("COO GPU STRIDE SHARED MEM", argv[1], TIMED_RUNS, WARMUP_RUNS);
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
    int shared_mem_size = threads_per_block * sizeof(dtype);  // 1024
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < TOTAL_RUNS; i++) {
        cudaMemset(result, 0, n_row * sizeof(dtype));
        cudaEventRecord(start);
        if (mode == 0) {
            coo_globmem_spmv_seq<<<blocks_per_grid, threads_per_block>>>(ARow, ACol, Aval, v, nnz, result);
        } else if (mode == 1) {
            coo_globmem_spmv_stride<<<blocks_per_grid, threads_per_block>>>(ARow, ACol, Aval, v, nnz, result);
        } else if (mode == 2) {
            coo_sharmem_spmv_seq<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(ARow, ACol, Aval, v, nnz, result);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start, stop);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = coo_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
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
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(Aval);
    cudaFree(v);
    return 0;
}