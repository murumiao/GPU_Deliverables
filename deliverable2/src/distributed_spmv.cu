#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/deliverable1.cuh"

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

void splitCSR(int total, int n_row, int n_col, int* rowPtr, int* colIndexes, dtype* AVal, int*** splitRowPtr, int*** splitColIndexes, dtype*** splitAVal, int** splitNRows, int** splitNNZ) {
    *splitRowPtr = (int**)malloc(total * sizeof(int*));
    *splitColIndexes = (int**)malloc(total * sizeof(int*));
    *splitAVal = (float**)malloc(total * sizeof(float*));
    *splitNRows = (int*)malloc(total * sizeof(int));
    *splitNNZ = (int*)malloc(total * sizeof(int));

    int base = n_row / total;
    int rem = n_row % total;

    for (int p = 0; p < total; p++) {
        int rows = base + (p < rem ? 1 : 0);
        int start = p * base + (p < rem ? p : rem);
        int end = start + rows;

        int firstNNZ = rowPtr[start];
        int lastNNZ = rowPtr[end];
        int nnzPart = lastNNZ - firstNNZ;

        (*splitNRows)[p] = rows;
        (*splitNNZ)[p] = nnzPart;

        (*splitRowPtr)[p] = (int*)malloc((rows + 1) * sizeof(int));
        (*splitColIndexes)[p] = (int*)malloc(nnzPart * sizeof(int));
        (*splitAVal)[p] = (float*)malloc(nnzPart * sizeof(float));

        for (int i = 0; i <= rows; i++) {
            (*splitRowPtr)[p][i] = rowPtr[start + i] - firstNNZ;
        }

        memcpy((*splitColIndexes)[p], &colIndexes[firstNNZ], nnzPart * sizeof(int));
        memcpy((*splitAVal)[p], &AVal[firstNNZ], nnzPart * sizeof(float));
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
    if (my_rank == 0) {
        // read file
        readMatrixFile(matrixPath, &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
        printf("Loaded data\n");

        // split data (1D)
        int **splitRowPtr = NULL, **splitColIndexes = NULL;
        dtype** splitAVal = NULL;
        int *splitNRows = NULL, *splitNNz = NULL;
        splitCSR(comm_size, n_row, n_col, rowPtr, colIndexes, AVal, &splitRowPtr, &splitColIndexes, &splitAVal, &splitNRows, &splitNNz);
        printf("Splitted the data\n");
        // send data
        MPI_Request req;
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Isend(&splitNRows[other_rank], 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(&n_col, 1, MPI_INT, other_rank, 1, MPI_COMM_WORLD, &req);
            MPI_Isend(&splitNNz[other_rank], 1, MPI_INT, other_rank, 2, MPI_COMM_WORLD, &req);
        }
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Send(splitRowPtr[other_rank], splitNRows[other_rank] + 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
            MPI_Send(splitColIndexes[other_rank], splitNNz[other_rank], MPI_INT, other_rank, 1, MPI_COMM_WORLD);
            MPI_Send(splitAVal[other_rank], splitNNz[other_rank], MPI_DTYPE, other_rank, 2, MPI_COMM_WORLD);
        }
        // free memory
        free(rowPtr);
        free(colIndexes);
        free(AVal);
        n_row = splitNRows[0];
        nnz = splitNNz[0];
        cudaMemcpy(rowPtr, splitRowPtr[0], splitNRows[0], cudaMemcpyHostToDevice);
        cudaMemcpy(colIndexes, splitColIndexes[0], nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(AVal, splitAVal[0], nnz, cudaMemcpyHostToDevice);
        free(splitRowPtr);
        free(splitColIndexes);
        free(splitNNz);
    } else {
        // recieve size
        MPI_Recv(&n_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_col, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // allocate cudaMem
        cudaMallocManaged(&rowPtr, (n_row + 1) * sizeof(int));
        cudaMallocManaged(&colIndexes, nnz * sizeof(int));
        cudaMallocManaged(&AVal, nnz * sizeof(dtype));

        // recieve CSR
        MPI_Recv(rowPtr, n_row + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(colIndexes, nnz, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(AVal, nnz, MPI_DTYPE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("[%d]: nrow: %d\n", my_rank, n_row);
    nvmlShutdown();
    MPI_Finalize();
    return (0);
}