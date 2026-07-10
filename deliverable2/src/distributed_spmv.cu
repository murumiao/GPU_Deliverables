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
    if (my_rank == 0) {
        // read file
        readMatrixFile(matrixPath, &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
        printf("Loaded data\n");

        // split data (1D)
        int **splitRowPtr = NULL, **splitColIndexes = NULL, **splitGlobalRows = NULL;
        dtype** splitAVal = NULL;
        int *splitNRows = NULL, *splitNNZ = NULL;
        splitCSR(comm_size, n_row, n_col, rowPtr, colIndexes, AVal, &splitRowPtr, &splitColIndexes, &splitAVal, &splitNRows, &splitNNZ, &splitGlobalRows);
        printf("Splitted the data\n");
        // send data
        MPI_Request req;
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Isend(&splitNRows[other_rank], 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(&n_col, 1, MPI_INT, other_rank, 1, MPI_COMM_WORLD, &req);
            MPI_Isend(&splitNNZ[other_rank], 1, MPI_INT, other_rank, 2, MPI_COMM_WORLD, &req);
        }
        for (int other_rank = 1; other_rank < comm_size; other_rank++) {
            MPI_Send(splitRowPtr[other_rank], splitNRows[other_rank] + 1, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
            MPI_Send(splitColIndexes[other_rank], splitNNZ[other_rank], MPI_INT, other_rank, 1, MPI_COMM_WORLD);
            MPI_Send(splitAVal[other_rank], splitNNZ[other_rank], MPI_DTYPE, other_rank, 2, MPI_COMM_WORLD);
            MPI_Send(splitGlobalRows[other_rank], splitNRows[other_rank], MPI_INT, other_rank, 3, MPI_COMM_WORLD);
        }
        // free memory
        free(rowPtr);
        free(colIndexes);
        free(AVal);
        n_row = splitNRows[0];
        nnz = splitNNZ[0];
        cudaMemcpy(rowPtr, splitRowPtr[0], (n_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(colIndexes, splitColIndexes[0], nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(AVal, splitAVal[0], nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(globalRows, splitGlobalRows[0], n_row * sizeof(int), cudaMemcpyHostToDevice);

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
        free(splitNRows);
        free(splitNNZ);
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
        MPI_Recv(&globalRows, n_row, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("[%d]: nrow: %d\n", my_rank, n_row);
    
    //todo each rank does spmv

    //todo distribute the results
    nvmlShutdown();
    MPI_Finalize();
    return (0);
}