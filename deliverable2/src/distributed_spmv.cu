#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/deliverable1.cuh"
#include "../include/spmv_utils.h"

static const int HOST_WARP_SIZE = 32;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        };
    }
}

typedef struct {
    int rank;
    size_t host_bytes;
    size_t gpu_bytes;
    int n_row;
    int n_col;
    int nnz;
} MemoryRecord;

int calculate_blocks_per_grid(int mode, int row_count, int threads_per_block) {
    if (row_count <= 0) {
        return 1;
    }

    if (mode == 0) {
        int blocks = (row_count + threads_per_block - 1) / threads_per_block;
        return blocks > 0 ? blocks : 1;
    }

    int warps_per_block = threads_per_block / HOST_WARP_SIZE;
    if (warps_per_block <= 0) {
        warps_per_block = 1;
    }

    int blocks = (row_count + warps_per_block - 1) / warps_per_block;
    return blocks > 0 ? blocks : 1;
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
    } else {
        fprintf(stdout, "Rank %d has device: %d\n", my_rank, my_device);
    }
}

void splitCSR(int comm_size, int n_row, int n_col, int* rowPtr, int* colIndexes, dtype* AVal, int*** splitRowPtr, int*** splitColIndexes, dtype*** splitAVal, int** splitNRows, int** splitNNZ, int*** splitGlobalRows) {
    *splitRowPtr = (int**)malloc(comm_size * sizeof(int*));
    *splitColIndexes = (int**)malloc(comm_size * sizeof(int*));
    *splitAVal = (dtype**)malloc(comm_size * sizeof(dtype*));
    *splitNRows = (int*)malloc(comm_size * sizeof(int));
    *splitNNZ = (int*)malloc(comm_size * sizeof(int));
    *splitGlobalRows = (int**)malloc(comm_size * sizeof(int*));

    for (int p = 0; p < comm_size; p++) {
        int localRows = 0;
        int localNNZ = 0;

        for (int r = p; r < n_row; r += comm_size) {
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

        for (int r = p; r < n_row; r += comm_size) {
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

void write_memory_footprint_csv(const char* filename, int rank, int threads, int sharedmem, size_t host_bytes, size_t gpu_bytes, int n_row, int n_col, int nnz) {
    MemoryRecord local = {rank, host_bytes, gpu_bytes, n_row, n_col, nnz};

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MemoryRecord* all = NULL;
    if (rank == 0)
        all = (MemoryRecord*)malloc(comm_size * sizeof(MemoryRecord));

    MPI_Gather(&local, sizeof(MemoryRecord), MPI_BYTE, all, sizeof(MemoryRecord), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "fopen cant find %s", filename);
            free(all);
            return;
        }
        fprintf(fp, "rank, threads, sharedmem, n_row, n_col, nnz, host_memory_bytes, gpu_memory_bytes, total_memory_bytes, total_memory_megabytes\n");
        for (int i = 0; i < comm_size; i++) {
            fprintf(fp, "%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%f\n", all[i].rank, threads, sharedmem, all[i].n_row, all[i].n_col, all[i].nnz, all[i].host_bytes, all[i].gpu_bytes, all[i].host_bytes + all[i].gpu_bytes, (all[i].host_bytes + all[i].gpu_bytes) / 1e6);
        }
        fclose(fp);
        free(all);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
//! MAIN FUNCTION HERE
//! MAIN FUNCTION HERE
//! MAIN FUNCTION HERE
//! MAIN FUNCTION HERE
//! MAIN FUNCTION HERE
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
/* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */

int main(int argc, char* argv[]) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode[0,1,2,3]> <n_threads_per_block> <shared_mem_size> <save path, optional>. Got %d args\n", argv[0], argc);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char* matrixPath = argv[1];
    int mode = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);
    int shared_mem_size = atoi(argv[4]);
    char* save_folder_path;
    if (argc == 5) {
        // save_folder_path = "./GPU_Deliverables/deliverable2/results/";
        save_folder_path = "results_1D/";
    } else {
        save_folder_path = argv[5];
    }

    int my_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    assign_GPU_to_rank(my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        fprintf(stderr, "Doing matrix=%s, mode=%d, threads=%d, shared_mem=%d\n", matrixPath, mode, threads_per_block, shared_mem_size);

    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    dtype* AVal = NULL;
    int* globalRows = NULL;

    int* splitNRows = NULL;
    int total_nrows = -1;

    dtype* dense_vector;
    dtype* result;

    size_t host_bytes = 0, gpu_bytes = 0;
    double communication_time_arr[TIMED_RUNS], exec_time_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];
    cudaEvent_t start_spmv, stop_spmv, start_reading_data, stop_reading_data;
    dtype* total_solution = NULL;
    gpuErrchk(cudaEventCreate(&start_spmv));
    gpuErrchk(cudaEventCreate(&stop_spmv));
    gpuErrchk(cudaEventCreate(&start_reading_data));
    gpuErrchk(cudaEventCreate(&stop_reading_data));

    int first_run = 0;
    for (int run_i = 0; run_i < TOTAL_RUNS; run_i++) {
        if (my_rank == 0)
            fprintf(stderr, "Run: %d/%d\n", run_i, TOTAL_RUNS);
        gpuErrchk(cudaEventRecord(start_reading_data));
        if (my_rank == 0) {
            // read file
            n_row = -1;
            n_col = -1;
            nnz = -1;
            readMatrixFile(matrixPath, &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
            total_nrows = n_row;

            // dense vector
            gpuErrchk(cudaMallocManaged(&dense_vector, n_col * sizeof(dtype)));
            for (int i = 0; i < n_col; i++) {
                dense_vector[i] = 1.0;
            }
            // split data (1D)
            int **splitRowPtr = NULL, **splitColIndexes = NULL, **splitGlobalRows = NULL;
            dtype** splitAVal = NULL;
            int* splitNNZ = NULL;
            splitCSR(comm_size, n_row, n_col, rowPtr, colIndexes, AVal, &splitRowPtr, &splitColIndexes, &splitAVal, &splitNRows, &splitNNZ, &splitGlobalRows);
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
            // communicate dense vector
            for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                MPI_Isend(dense_vector, n_col, MPI_DTYPE, other_rank, 5, MPI_COMM_WORLD, &req);
            }
            // calculate memoery footprint before free
            if (first_run == 0) {
                first_run = 1;
                host_bytes += (comm_size) * sizeof(int*) * 3;
                host_bytes += (comm_size) * sizeof(dtype*);
                host_bytes += (comm_size) * sizeof(int) * 2;

                for (int p = 0; p < comm_size; p++) {
                    host_bytes += (splitNRows[p] + 1) * sizeof(int) * 2;
                    host_bytes += (splitNNZ[p] + 1) * sizeof(int) * 2;
                }
                host_bytes += (total_nrows) * sizeof(dtype);
                host_bytes += (total_nrows) * sizeof(int);
                host_bytes += (comm_size) * sizeof(int) * 2;
                host_bytes += (total_nrows) * sizeof(dtype);

                gpu_bytes += n_col * sizeof(dtype);
                gpu_bytes += (n_row + 1) * sizeof(int);
                gpu_bytes += nnz * sizeof(int);
                gpu_bytes += nnz * sizeof(dtype);
                gpu_bytes += n_row * sizeof(int);
                gpu_bytes += n_row * sizeof(dtype);
            }
            // free memory
            free(rowPtr);
            free(colIndexes);
            free(AVal);

            n_row = splitNRows[0];
            nnz = splitNNZ[0];
            gpuErrchk(cudaMallocManaged(&rowPtr, (n_row + 1) * sizeof(int)));
            gpuErrchk(cudaMallocManaged(&colIndexes, nnz * sizeof(int)));
            gpuErrchk(cudaMallocManaged(&AVal, nnz * sizeof(dtype)));
            gpuErrchk(cudaMallocManaged(&globalRows, n_row * sizeof(int)));
            gpuErrchk(cudaMemcpy(rowPtr, splitRowPtr[0], (n_row + 1) * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(colIndexes, splitColIndexes[0], nnz * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(AVal, splitAVal[0], nnz * sizeof(dtype), cudaMemcpyHostToDevice));
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

            // recieve for dense vector
            gpuErrchk(cudaMallocManaged(&dense_vector, n_col * sizeof(dtype)));
            MPI_Recv(dense_vector, n_col, MPI_DTYPE, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (first_run == 0) {
                first_run = 1;
                host_bytes += (n_row + 1) * sizeof(int);
                host_bytes += (nnz) * sizeof(int);
                host_bytes += (nnz) * sizeof(dtype);

                host_bytes += (n_row) * sizeof(int);
                host_bytes += (total_nrows) * sizeof(int);
                host_bytes += (comm_size) * sizeof(int);
                host_bytes += (n_col) * sizeof(dtype);

                gpu_bytes += (n_row + 1) * sizeof(int);
                gpu_bytes += nnz * sizeof(int);
                gpu_bytes += nnz * sizeof(dtype);
                gpu_bytes += n_col * sizeof(dtype);
                gpu_bytes += n_row * sizeof(dtype);
            }
        }

        gpuErrchk(cudaEventRecord(stop_reading_data));
        gpuErrchk(cudaEventSynchronize(stop_reading_data));

        int blocks_per_grid = calculate_blocks_per_grid(mode, n_row, threads_per_block);

        if (mode != 0 && threads_per_block < HOST_WARP_SIZE) {
            fprintf(stderr, "Rank %d: mode %d requires at least %d threads per block, got %d\n", my_rank, mode, HOST_WARP_SIZE, threads_per_block);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (mode != 0 && threads_per_block % HOST_WARP_SIZE != 0) {
            fprintf(stderr, "Rank %d: mode %d requires threads per block to be a multiple of %d, got %d\n", my_rank, mode, HOST_WARP_SIZE, threads_per_block);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (my_rank == 0 && run_i == 0) {
            if (mode == 0) {
                print_starting_info("(mode0)CSR GPU SEQUENTIAL GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, 0);
            } else if (mode == 1) {
                print_starting_info("(mode1)CSR GPU STRIDE GLOBAL MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, 0);
            } else if (mode == 2) {
                print_starting_info("(mode2)CSR GPU STRIDE SHARED MEM", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
            } else if (mode == 3) {
                print_starting_info("(mode3)CSR GPU SHARED MEM COALESCED", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
            } else {
                fprintf(stderr, "Wrong mode\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        //! FINISHED SPLITTING THE DATA^^. NOW PERFORM THE SPMV vv.
        //! FINISHED SPLITTING THE DATA^^. NOW PERFORM THE SPMV vv.
        //! FINISHED SPLITTING THE DATA^^. NOW PERFORM THE SPMV vv.
        //! FINISHED SPLITTING THE DATA^^. NOW PERFORM THE SPMV vv.
        //! FINISHED SPLITTING THE DATA^^. NOW PERFORM THE SPMV vv.
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
        /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */

        // Create result vector
        gpuErrchk(cudaMallocManaged(&result, n_row * sizeof(dtype)));
        cudaMemset(result, 0, n_row * sizeof(dtype));

        cudaEventRecord(start_spmv);
        // Each rank does spmv
        if (mode == 0) {
            csr_globmem_spmv_sequential<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, dense_vector, result);
        } else if (mode == 1) {
            csr_globmem_spmv_stride<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, n_row, dense_vector, result);
        } else if (mode == 2) {
            csr_sharmem_spmv_stride<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, n_row, dense_vector, result);
        } else if (mode == 3) {
            csr_sharmem_coalesced<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, n_row, dense_vector, result);
        }
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaEventRecord(stop_spmv));
        gpuErrchk(cudaEventSynchronize(stop_spmv));

        // Gather the results
        int* displs = (int*)malloc(comm_size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < comm_size; i++)
            displs[i] = displs[i - 1] + splitNRows[i - 1];
        dtype* reciever_buffer = (dtype*)malloc(total_nrows * sizeof(dtype));
        MPI_Allgatherv(result, n_row, MPI_DTYPE, reciever_buffer, splitNRows, displs, MPI_DTYPE, MPI_COMM_WORLD);
        free(displs);

        if (total_solution != NULL) {
            free(total_solution);
            total_solution = NULL;
        }
        reconstruct_solution(reciever_buffer, splitNRows, &total_solution, total_nrows, comm_size, my_rank);
        free(reciever_buffer);

        // extract stats of the run
        float communication_time_ms;
        gpuErrchk(cudaEventElapsedTime(&communication_time_ms, start_reading_data, stop_reading_data));
        double communication_time_s = communication_time_ms / 1000.0;

        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start_spmv, stop_spmv);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = csr_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
        double gflop = calculate_gflop(nnz, exec_time_s);
        if (run_i >= WARMUP_RUNS) {
            communication_time_arr[run_i - WARMUP_RUNS] = communication_time_s;
            exec_time_arr[run_i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[run_i - WARMUP_RUNS] = bandwidth;
            gflops_arr[run_i - WARMUP_RUNS] = gflop;
        }
    }
    gpuErrchk(cudaEventDestroy(start_spmv));
    gpuErrchk(cudaEventDestroy(stop_spmv));
    gpuErrchk(cudaEventDestroy(start_reading_data));
    gpuErrchk(cudaEventDestroy(stop_reading_data));

    final_info_print(my_rank, communication_time_arr, exec_time_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, total_solution, total_nrows);
    /***********************************************************************
     Each rank saves own stats (per rank efficency = t_max/t_rank)
    ***********************************************************************/
    // create folder
    char tmp[512];
    char folder_name[512];
    char* matrix_name = strrchr(matrixPath, '/');
    matrix_name = matrix_name + 1;

    sprintf(folder_name, save_folder_path);
    // sprintf(folder_name, "results/");
    make_dir(folder_name, my_rank);
    strcpy(tmp, folder_name);

    sprintf(folder_name, "%s/%s", tmp, matrix_name);
    make_dir(folder_name, my_rank);
    strcpy(tmp, folder_name);

    sprintf(folder_name, "%s/%d ranks", tmp, comm_size);
    make_dir(folder_name, my_rank);
    strcpy(tmp, folder_name);

    sprintf(folder_name, "%s/m%d", tmp, mode);
    make_dir(folder_name, my_rank);
    strcpy(tmp, folder_name);

    sprintf(folder_name, "%s/t%d", tmp, threads_per_block);
    make_dir(folder_name, my_rank);
    strcpy(tmp, folder_name);

    sprintf(folder_name, "%s/s%d", tmp, shared_mem_size);
    make_dir(folder_name, my_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    char file_name[512];
    sprintf(file_name, "%s/rank_%d.csv", folder_name, my_rank);

    save_statistics(file_name, TIMED_RUNS, communication_time_arr, exec_time_arr, bandwidth_arr, gflops_arr);

    char mem_footprint_file_name[512];
    sprintf(mem_footprint_file_name, "%s/memory_footprint.csv", folder_name);
    write_memory_footprint_csv(mem_footprint_file_name, my_rank, threads_per_block, shared_mem_size, host_bytes, gpu_bytes, n_row, n_col, nnz);

    gpuErrchk(cudaFree(rowPtr));
    gpuErrchk(cudaFree(colIndexes));
    gpuErrchk(cudaFree(AVal));
    gpuErrchk(cudaFree(dense_vector));

    nvmlShutdown();
    MPI_Finalize();
    return (0);
}
