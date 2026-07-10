#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/deliverable1.cuh"
#include "../include/spmv_utils.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void assign_GPU_to_rank(int my_rank) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error_id));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Assign GPU
    int my_device = my_rank % deviceCount;
    cudaSetDevice(my_device);

    // Check the asignement

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Process %d: Failed to initialize NVML: %s\n", my_rank, nvmlErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(my_device, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Process %d: Failed to get handle for device %d: %s\n", my_rank, my_device, nvmlErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void splitCSR(int total, int n_row, int n_col, int* rowPtr, int* colIndexes, dtype* AVal, int*** splitRowPtr, int*** splitColIndexes, dtype*** splitAVal, int** splitNRows, int** splitNNZ, int*** splitGlobalRows) {
    *splitRowPtr = (int**)malloc(total * sizeof(int*));
    *splitColIndexes = (int**)malloc(total * sizeof(int*));
    *splitAVal = (dtype**)malloc(total * sizeof(dtype*));
    *splitNRows = (int*)malloc(total * sizeof(int));
    *splitNNZ = (int*)malloc(total * sizeof(int));
    *splitGlobalRows = (int**)malloc(total * sizeof(int*));

    for (int p = 0; p < total; p++) {
        int localRows = 0;
        int localNNZ = 0;

        for (int r = p; r < n_row; r += total) {
            localRows++;
            localNNZ += rowPtr[r + 1] - rowPtr[r];
        }

        (*splitNRows)[p] = localRows;
        (*splitNNZ)[p] = localNNZ;
        (*splitGlobalRows)[p] = (int*)malloc(localRows * sizeof(int));
        (*splitRowPtr)[p] = (int*)malloc((localRows + 1) * sizeof(int));
        (*splitColIndexes)[p] = (int*)malloc(localNNZ * sizeof(int));
        (*splitAVal)[p] = (dtype*)malloc(localNNZ * sizeof(dtype));

        int localRow = 0;
        int localNNZOffset = 0;
        (*splitRowPtr)[p][0] = 0;

        for (int r = p; r < n_row; r += total) {
            (*splitGlobalRows)[p][localRow] = r;
            int rowNNZ = rowPtr[r + 1] - rowPtr[r];

            memcpy(&((*splitColIndexes)[p][localNNZOffset]), &colIndexes[rowPtr[r]], rowNNZ * sizeof(int));
            memcpy(&((*splitAVal)[p][localNNZOffset]), &AVal[rowPtr[r]], rowNNZ * sizeof(dtype));

            localNNZOffset += rowNNZ;
            localRow++;
            (*splitRowPtr)[p][localRow] = localNNZOffset;
        }
    }
}

void reconstruct_solution(dtype* received, int* recvCounts, dtype** reconstructed_solution, int total_len, int comm_size, int rank) {
    *reconstructed_solution = (dtype*)malloc(total_len * sizeof(dtype));
    int offset = 0;

    for (int rank = 0; rank < comm_size; rank++) {
        for (int local_index = 0; local_index < recvCounts[rank]; local_index++) {
            int global_index = rank + local_index * comm_size;
            (*reconstructed_solution)[global_index] = received[offset + local_index];
        }
        offset += recvCounts[rank];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage %s <path_to_matrix>\n", argv[0]);
        exit(1);
    }
    char* matrixPath = argv[1];

    int my_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    assign_GPU_to_rank(my_rank);

    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    dtype* AVal = NULL;
    int* globalRows = NULL;

    int* splitNRows = NULL;
    int total_nrows = -1;
    if (my_rank == 0) {
        // read file
        readMatrixFile(matrixPath, &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
        total_nrows = n_row;
        printf("Loaded data\n");

        // split data (1D)
        int **splitRowPtr = NULL, **splitColIndexes = NULL, **splitGlobalRows = NULL;
        dtype** splitAVal = NULL;
        int* splitNNZ = NULL;
        splitCSR(comm_size, n_row, n_col, rowPtr, colIndexes, AVal, &splitRowPtr, &splitColIndexes, &splitAVal, &splitNRows, &splitNNZ, &splitGlobalRows);
        printf("Splitted the data\n");
        // send data
        MPI_Request req;
        //** to allocate arrays
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Isend(&splitNRows[other_rank], 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(&n_col, 1, MPI_INT, other_rank, 1, MPI_COMM_WORLD, &req);
            MPI_Isend(&splitNNZ[other_rank], 1, MPI_INT, other_rank, 2, MPI_COMM_WORLD, &req);
        }
        // ** reconstruction info
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Isend(&total_nrows, 1, MPI_INT, other_rank, 4, MPI_COMM_WORLD, &req);
        }
        //** arrays
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Send(splitRowPtr[other_rank], splitNRows[other_rank] + 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
            MPI_Send(splitColIndexes[other_rank], splitNNZ[other_rank], MPI_INT, other_rank, 1, MPI_COMM_WORLD);
            MPI_Send(splitAVal[other_rank], splitNNZ[other_rank], MPI_DTYPE, other_rank, 2, MPI_COMM_WORLD);
            MPI_Send(splitGlobalRows[other_rank], splitNRows[other_rank], MPI_INT, other_rank, 3, MPI_COMM_WORLD);
        }
        // ** reconstruction info
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Isend(splitNRows, comm_size, MPI_INT, other_rank, 4, MPI_COMM_WORLD, &req);
        }
        // free memory
        free(rowPtr);
        free(colIndexes);
        free(AVal);
        n_row = splitNRows[0];
        nnz = splitNNZ[0];
        gpuErrchk(cudaMallocManaged(&rowPtr, (n_row + 1) * sizeof(int)));
        gpuErrchk(cudaMallocManaged(&colIndexes, nnz * sizeof(int)));
        gpuErrchk(cudaMallocManaged(&AVal, nnz * sizeof(int)));
        gpuErrchk(cudaMallocManaged(&globalRows, n_row * sizeof(int)));
        gpuErrchk(cudaMemcpy(rowPtr, splitRowPtr[0], (n_row + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(colIndexes, splitColIndexes[0], nnz * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(AVal, splitAVal[0], nnz * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(globalRows, splitGlobalRows[0], n_row * sizeof(int), cudaMemcpyHostToDevice));

        for (int p = 0; p < comm_size; p++) {
            free(splitRowPtr[p]);
            free(splitColIndexes[p]);
            free(splitAVal[p]);
            free(splitGlobalRows[p]);
        }
        free(splitRowPtr);
        free(splitColIndexes);
        free(splitAVal);
        free(splitGlobalRows);
        free(splitNNZ);
    }
    // other ranks
    else {
        // recieve size
        MPI_Recv(&n_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_col, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&total_nrows, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // allocate cudaMem
        gpuErrchk(cudaMallocManaged(&rowPtr, (n_row + 1) * sizeof(int)));
        gpuErrchk(cudaMallocManaged(&colIndexes, nnz * sizeof(int)));
        gpuErrchk(cudaMallocManaged(&AVal, nnz * sizeof(dtype)));

        // recieve CSR
        MPI_Recv(rowPtr, n_row + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(colIndexes, nnz, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(AVal, nnz, MPI_DTYPE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // allocate and receive globalRows
        globalRows = (int*)malloc(n_row * sizeof(int));
        MPI_Recv(globalRows, n_row, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // recieve for reconstruction
        splitNRows = (int*)malloc(comm_size * sizeof(int));
        MPI_Recv(splitNRows, comm_size, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Create dense vector & result vector
    dtype* v;
    gpuErrchk(cudaMallocManaged(&v, n_col * sizeof(dtype)));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }
    dtype* result;
    gpuErrchk(cudaMallocManaged(&result, n_row * sizeof(dtype)));

    int threads_per_block = 256;
    int blocks_per_grid = (n_row + threads_per_block - 1) / threads_per_block;

    double timer_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];
    cudaEvent_t start_spmv, stop_spmv;
    cudaEventCreate(&start_spmv);
    cudaEventCreate(&stop_spmv);

    for (int run_i = 0; run_i < TOTAL_RUNS; run_i++) {
        cudaMemset(result, 0, n_row * sizeof(dtype));

        cudaEventRecord(start_spmv);
        // Each rank does spmv
        csr_globmem_spmv_sequential<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, v, result);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaEventRecord(stop_spmv);
        cudaEventSynchronize(stop_spmv);

        // Gather the results
        int* displs = (int*)malloc(comm_size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < comm_size; i++)
            displs[i] = displs[i - 1] + splitNRows[i - 1];
        dtype* reciever_buffer = (dtype*)malloc(total_nrows * sizeof(dtype));
        MPI_Allgatherv(result, n_row, MPI_DTYPE, reciever_buffer, splitNRows, displs, MPI_DTYPE, MPI_COMM_WORLD);
        free(displs);

        dtype* total_solution = NULL;
        reconstruct_solution(reciever_buffer, splitNRows, &total_solution, total_nrows, comm_size, my_rank);

        // extract stats of the run
        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start_spmv, stop_spmv);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = csr_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
        double gflop = calculate_gflop(nnz, exec_time_s);
        if (run_i > WARMUP_RUNS) {
            timer_arr[run_i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[run_i - WARMUP_RUNS] = bandwidth;
            gflops_arr[run_i - WARMUP_RUNS] = gflop;
        }
        if (my_rank == 0) {
            print_run_stat(run_i, exec_time_s, bandwidth, gflop);
            if (run_i == TOTAL_RUNS - 1) {
                final_info_print(timer_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, reciever_buffer, n_row);
            }
        }
    }
    cudaEventDestroy(start_spmv);
    cudaEventDestroy(stop_spmv);

    nvmlShutdown();
    MPI_Finalize();
    cudaFree(rowPtr);
    cudaFree(colIndexes);
    cudaFree(AVal);
    cudaFree(v);
    return (0);
}