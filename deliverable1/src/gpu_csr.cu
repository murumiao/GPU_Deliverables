#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

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
            CUDA_CHECK(cudaMallocManaged(rowPtr, (*n_row + 1) * sizeof(int)));
            CUDA_CHECK(cudaMallocManaged(colIndexes, *nnz * sizeof(int)));
            CUDA_CHECK(cudaMallocManaged(valCSR, *nnz * sizeof(dtype)));
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

// load balance
__global__ void csr_warp_per_row_dynamic(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    // int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    // int lane = threadIdx.x & 31;

    // if (warp_id >= rowLen) return;

    // int row_start = rowPtr[warp_id];
    // int row_end = rowPtr[warp_id + 1];
    // dtype sum = 0.0;

    // for (int i = row_start + lane; i < row_end; i += 32) {
    //     sum += AVal[i] * v[colIndexes[i]];
    // }

    // // Warp shuffle reduction
    // for (int offset = 16; offset > 0; offset >>= 1) {
    //     sum += __shfl_down_sync(0xffffffff, sum, offset);
    // }

    // if (lane == 0) {
    //     result[warp_id] = sum;
    // }
}

// coalesced and shared (less bank conflicts)
__global__ void csr_sharmem_coalesced(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    extern __shared__ dtype shared_data[];

    int tid = threadIdx.x;
    int block_start_row = blockIdx.x * (blockDim.x / 32);
    int warp_id = tid / 32;
    int lane = tid & 31;

    if (block_start_row + warp_id >= rowLen) return;

    int row = block_start_row + warp_id;
    int row_start = rowPtr[row];
    int row_end = rowPtr[row + 1];

    // Coalesced load into shared memory
    dtype* warp_shared = &shared_data[warp_id * 32];
    warp_shared[lane] = 0.0;
    __syncthreads();

    // Accumulate values
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        warp_shared[lane] += AVal[idx] * v[colIndexes[idx]];
    }
    __syncthreads();

    // Parallel reduction within warp using shared memory
    if (lane < 16) warp_shared[lane] += warp_shared[lane + 16];
    __syncthreads();
    if (lane < 8) warp_shared[lane] += warp_shared[lane + 8];
    __syncthreads();
    if (lane < 4) warp_shared[lane] += warp_shared[lane + 4];
    __syncthreads();
    if (lane < 2) warp_shared[lane] += warp_shared[lane + 2];
    __syncthreads();
    if (lane < 1) warp_shared[lane] += warp_shared[lane + 1];
    __syncthreads();

    if (lane == 0) {
        result[row] = warp_shared[0];
    }
}
int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode> <n_blocks> <n_threads_per_block> <shared_mem_size>\n", argv[0]);
        exit(1);
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
    printf("deviceCount = %d\n", deviceCount);

    int mode = atoi(argv[2]);
    int threads_per_block = atoi(argv[4]);
    int shared_mem_size = atoi(argv[5]);
    //* CSR storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    dtype* AVal = NULL;

    readMatrixFile(argv[1], &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);

    int blocks_per_grid = (n_row + threads_per_block - 1) / threads_per_block;
    ;
    if (mode == 0) {
        print_starting_info("CSR GPU SEQUENTIAL GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, -1);
    } else if (mode == 1) {
        print_starting_info("CSR GPU STRIDE GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, -1);
    } else if (mode == 2) {
        print_starting_info("CSR GPU STRIDE SHARED MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
    } else if (mode == 3) {  // skipped
        print_starting_info("CSR GPU GLOBMEM WARP-PER-ROW DYNAMIC", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, -1);
    } else if (mode == 4) {
        print_starting_info("CSR GPU SHARED MEM COALESCED", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
    } else {
        fprintf(stderr, "Wrong mode\n");
        exit(1);
    }

    // Create dense vector
    dtype* v;
    CUDA_CHECK(cudaMallocManaged(&v, n_col * sizeof(dtype)));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    dtype* result;
    CUDA_CHECK(cudaMallocManaged(&result, n_row * sizeof(dtype)));

    double timer_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < TOTAL_RUNS; i++) {
        CUDA_CHECK(cudaMemset(result, 0, n_row * sizeof(dtype)));

        CUDA_CHECK(cudaEventRecord(start));
        if (mode == 0) {
            csr_globmem_spmv_sequential<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 1) {
            csr_globmem_spmv_stride<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 2) {
            csr_sharmem_spmv_stride<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 3) {
            csr_warp_per_row_dynamic<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        } else if (mode == 4) {
            csr_sharmem_coalesced<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float exec_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&exec_time_ms, start, stop));
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = csr_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
        double gflop = calculate_gflop(nnz, exec_time_s);
        if (i >= WARMUP_RUNS) {
            timer_arr[i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[i - WARMUP_RUNS] = bandwidth;
            gflops_arr[i - WARMUP_RUNS] = gflop;
        }
        print_run_stat(i, exec_time_s, bandwidth, gflop);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    final_info_print(timer_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, result, n_row);
    CUDA_CHECK(cudaFree(rowPtr));
    CUDA_CHECK(cudaFree(colIndexes));
    CUDA_CHECK(cudaFree(AVal));
    CUDA_CHECK(cudaFree(v));
    return 0;
}