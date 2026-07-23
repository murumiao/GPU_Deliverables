#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void partition_range(int total, int parts, int coord, int* start, int* count) {
    int base = total / parts;
    int remainder = total % parts;

    *count = base + (coord < remainder ? 1 : 0);
    *start = coord * base + (coord < remainder ? coord : remainder);
}

typedef struct {
    int row_start;
    int row_count;
    int col_start;
    int col_count;
    int nnz;
    int* rowPtr;
    int* colIndexes;
    dtype* AVal;
} LocalCsrBlock;

static void free_local_csr_block(LocalCsrBlock* block) {
    if (block->rowPtr != NULL) {
        free(block->rowPtr);
    }
    if (block->colIndexes != NULL) {
        free(block->colIndexes);
    }
    if (block->AVal != NULL) {
        free(block->AVal);
    }

    block->rowPtr = NULL;
    block->colIndexes = NULL;
    block->AVal = NULL;
    block->nnz = 0;
    block->row_start = 0;
    block->row_count = 0;
    block->col_start = 0;
    block->col_count = 0;
}

static void build_local_csr_block(int row_start, int row_count, int col_start, int col_count,
                                  const int* rowPtr, const int* colIndexes, const dtype* AVal,
                                  LocalCsrBlock* block) {
    block->row_start = row_start;
    block->row_count = row_count;
    block->col_start = col_start;
    block->col_count = col_count;
    block->nnz = 0;
    block->rowPtr = (int*)malloc((row_count + 1) * sizeof(int));
    block->colIndexes = NULL;
    block->AVal = NULL;

    block->rowPtr[0] = 0;
    for (int local_row = 0; local_row < row_count; local_row++) {
        int global_row = row_start + local_row;
        int row_nnz = 0;
        for (int idx = rowPtr[global_row]; idx < rowPtr[global_row + 1]; idx++) {
            int global_col = colIndexes[idx];
            if (global_col >= col_start && global_col < col_start + col_count) {
                row_nnz++;
            }
        }
        block->nnz += row_nnz;
        block->rowPtr[local_row + 1] = block->nnz;
    }

    if (block->nnz > 0) {
        block->colIndexes = (int*)malloc(block->nnz * sizeof(int));
        block->AVal = (dtype*)malloc(block->nnz * sizeof(dtype));
    }

    int nnz_offset = 0;
    for (int local_row = 0; local_row < row_count; local_row++) {
        int global_row = row_start + local_row;
        for (int idx = rowPtr[global_row]; idx < rowPtr[global_row + 1]; idx++) {
            int global_col = colIndexes[idx];
            if (global_col >= col_start && global_col < col_start + col_count) {
                block->colIndexes[nnz_offset] = global_col - col_start;
                block->AVal[nnz_offset] = AVal[idx];
                nnz_offset++;
            }
        }
    }
}

static void build_allgatherv_layout(MPI_Comm grid_comm, int comm_size, int global_row_count,
                                    int process_rows, int process_cols,
                                    int* recvcounts, int* displs) {
    (void)process_cols;

    for (int rank = 0; rank < comm_size; rank++) {
        int coords[2];
        MPI_Cart_coords(grid_comm, rank, 2, coords);

        int row_start = 0;
        int row_count = 0;
        partition_range(global_row_count, process_rows, coords[0], &row_start, &row_count);

        recvcounts[rank] = (coords[1] == 0) ? row_count : 0;
        displs[rank] = (coords[1] == 0) ? row_start : 0;
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
    } else {
        fprintf(stdout, "Rank %d has device: %d\n", my_rank, my_device);
    }
}

void splitCSR(int total, int n_row, int n_col, int* rowPtr, int* colIndexes, dtype* AVal, int*** splitRowPtr, int*** splitColIndexes, dtype*** splitAVal, int** splitNRows, int** splitNNZ, int*** splitGlobalRows) {
    (void)total;
    (void)n_row;
    (void)n_col;
    (void)rowPtr;
    (void)colIndexes;
    (void)AVal;
    (void)splitRowPtr;
    (void)splitColIndexes;
    (void)splitAVal;
    (void)splitNRows;
    (void)splitNNZ;
    (void)splitGlobalRows;
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
        save_folder_path = "results/";
    } else {
        save_folder_path = argv[5];
    }

    int my_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    assign_GPU_to_rank(my_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    int grid_dims[2] = {0, 0};
    MPI_Dims_create(comm_size, 2, grid_dims);

    int periods[2] = {0, 0};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dims, periods, 0, &grid_comm);

    int my_coords[2] = {0, 0};
    MPI_Cart_coords(grid_comm, my_rank, 2, my_coords);

    int row_remain_dims[2] = {0, 1};
    int col_remain_dims[2] = {1, 0};
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
    MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);

    int global_n_row = -1;
    int global_n_col = -1;
    int global_nnz = -1;

    int local_n_row = -1;
    int local_n_col = -1;
    int local_nnz = -1;
    int local_row_start = 0;
    int local_col_start = 0;

    int* rowPtr = NULL;
    int* colIndexes = NULL;
    dtype* AVal = NULL;
    int* global_rowPtr = NULL;
    int* global_colIndexes = NULL;
    dtype* global_AVal = NULL;

    dtype* dense_vector_full = NULL;
    dtype* dense_vector = NULL;
    dtype* result = NULL;
    dtype* row_reduced_result = NULL;

    double communication_time_arr[TIMED_RUNS], exec_time_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];
    cudaEvent_t start_spmv, stop_spmv, start_reading_data, stop_reading_data;
    dtype* total_solution = NULL;
    gpuErrchk(cudaEventCreate(&start_spmv));
    gpuErrchk(cudaEventCreate(&stop_spmv));
    gpuErrchk(cudaEventCreate(&start_reading_data));
    gpuErrchk(cudaEventCreate(&stop_reading_data));

    for (int run_i = 0; run_i < TOTAL_RUNS; run_i++) {
        gpuErrchk(cudaEventRecord(start_reading_data));
        if (my_rank == 0) {
            readMatrixFile(matrixPath, &global_rowPtr, &global_colIndexes, &global_AVal, &global_n_row, &global_n_col, &global_nnz);

            gpuErrchk(cudaMallocManaged(&dense_vector_full, global_n_col * sizeof(dtype)));
            for (int i = 0; i < global_n_col; i++) {
                dense_vector_full[i] = 1.0;
            }

            MPI_Bcast(&global_n_row, 1, MPI_INT, 0, grid_comm);
            MPI_Bcast(&global_n_col, 1, MPI_INT, 0, grid_comm);

            partition_range(global_n_row, grid_dims[0], my_coords[0], &local_row_start, &local_n_row);
            partition_range(global_n_col, grid_dims[1], my_coords[1], &local_col_start, &local_n_col);

            LocalCsrBlock local_block = {0};
            build_local_csr_block(local_row_start, local_n_row, local_col_start, local_n_col, global_rowPtr, global_colIndexes, global_AVal, &local_block);
            local_nnz = local_block.nnz;

            if (local_block.nnz > 0) {
                gpuErrchk(cudaMallocManaged(&rowPtr, (local_n_row + 1) * sizeof(int)));
                gpuErrchk(cudaMallocManaged(&colIndexes, local_block.nnz * sizeof(int)));
                gpuErrchk(cudaMallocManaged(&AVal, local_block.nnz * sizeof(dtype)));
                gpuErrchk(cudaMemcpy(rowPtr, local_block.rowPtr, (local_n_row + 1) * sizeof(int), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(colIndexes, local_block.colIndexes, local_block.nnz * sizeof(int), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(AVal, local_block.AVal, local_block.nnz * sizeof(dtype), cudaMemcpyHostToDevice));
            } else {
                gpuErrchk(cudaMallocManaged(&rowPtr, (local_n_row + 1) * sizeof(int)));
                colIndexes = NULL;
                AVal = NULL;
                for (int i = 0; i <= local_n_row; i++) {
                    rowPtr[i] = 0;
                }
            }

            dense_vector = dense_vector_full + local_col_start;

            free_local_csr_block(&local_block);

            for (int other_rank = 1; other_rank < comm_size; other_rank++) {
                int other_coords[2];
                MPI_Cart_coords(grid_comm, other_rank, 2, other_coords);

                int other_row_start = 0;
                int other_row_count = 0;
                int other_col_start = 0;
                int other_col_count = 0;
                partition_range(global_n_row, grid_dims[0], other_coords[0], &other_row_start, &other_row_count);
                partition_range(global_n_col, grid_dims[1], other_coords[1], &other_col_start, &other_col_count);

                LocalCsrBlock other_block = {0};
                build_local_csr_block(other_row_start, other_row_count, other_col_start, other_col_count, global_rowPtr, global_colIndexes, global_AVal, &other_block);

                MPI_Send(&other_block.nnz, 1, MPI_INT, other_rank, 0, grid_comm);
                if (other_block.nnz > 0) {
                    MPI_Send(other_block.rowPtr, other_row_count + 1, MPI_INT, other_rank, 1, grid_comm);
                    MPI_Send(other_block.colIndexes, other_block.nnz, MPI_INT, other_rank, 2, grid_comm);
                    MPI_Send(other_block.AVal, other_block.nnz, MPI_DTYPE, other_rank, 3, grid_comm);
                    MPI_Send(dense_vector_full + other_col_start, other_col_count, MPI_DTYPE, other_rank, 4, grid_comm);
                } else {
                    int* zero_row_ptr = other_block.rowPtr;
                    MPI_Send(zero_row_ptr, other_row_count + 1, MPI_INT, other_rank, 1, grid_comm);
                    MPI_Send(other_block.colIndexes, 0, MPI_INT, other_rank, 2, grid_comm);
                    MPI_Send(other_block.AVal, 0, MPI_DTYPE, other_rank, 3, grid_comm);
                    MPI_Send(dense_vector_full + other_col_start, other_col_count, MPI_DTYPE, other_rank, 4, grid_comm);
                }

                free_local_csr_block(&other_block);
            }
        }
        // other ranks
        else {
            MPI_Bcast(&global_n_row, 1, MPI_INT, 0, grid_comm);
            MPI_Bcast(&global_n_col, 1, MPI_INT, 0, grid_comm);

            partition_range(global_n_row, grid_dims[0], my_coords[0], &local_row_start, &local_n_row);
            partition_range(global_n_col, grid_dims[1], my_coords[1], &local_col_start, &local_n_col);

            MPI_Recv(&local_nnz, 1, MPI_INT, 0, 0, grid_comm, MPI_STATUS_IGNORE);

            gpuErrchk(cudaMallocManaged(&rowPtr, (local_n_row + 1) * sizeof(int)));
            if (local_nnz > 0) {
                gpuErrchk(cudaMallocManaged(&colIndexes, local_nnz * sizeof(int)));
                gpuErrchk(cudaMallocManaged(&AVal, local_nnz * sizeof(dtype)));
            } else {
                colIndexes = NULL;
                AVal = NULL;
            }

            MPI_Recv(rowPtr, local_n_row + 1, MPI_INT, 0, 1, grid_comm, MPI_STATUS_IGNORE);
            if (local_nnz > 0) {
                MPI_Recv(colIndexes, local_nnz, MPI_INT, 0, 2, grid_comm, MPI_STATUS_IGNORE);
                MPI_Recv(AVal, local_nnz, MPI_DTYPE, 0, 3, grid_comm, MPI_STATUS_IGNORE);
            }

            if (local_n_col > 0) {
                gpuErrchk(cudaMallocManaged(&dense_vector, local_n_col * sizeof(dtype)));
                MPI_Recv(dense_vector, local_n_col, MPI_DTYPE, 0, 4, grid_comm, MPI_STATUS_IGNORE);
            } else {
                dense_vector = NULL;
                dtype dummy_vector = 0.0f;
                MPI_Recv(&dummy_vector, 0, MPI_DTYPE, 0, 4, grid_comm, MPI_STATUS_IGNORE);
            }
        }

        gpuErrchk(cudaEventRecord(stop_reading_data));
        gpuErrchk(cudaEventSynchronize(stop_reading_data));

        int blocks_per_grid = calculate_blocks_per_grid(mode, local_n_row, threads_per_block);

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
                print_starting_info("(mode0)CSR GPU SEQUENTIAL 2D GRID", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, 0);
            } else if (mode == 1) {
                print_starting_info("(mode1)CSR GPU STRIDE 2D GRID", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, 0);
            } else if (mode == 2) {
                print_starting_info("(mode2)CSR GPU STRIDE SHARED MEM 2D GRID", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
            } else if (mode == 3) {
                print_starting_info("(mode3)CSR GPU SHARED MEM COALESCED 2D GRID", argv[1], TIMED_RUNS, WARMUP_RUNS, blocks_per_grid, threads_per_block, shared_mem_size);
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
        if (local_n_row > 0) {
            gpuErrchk(cudaMallocManaged(&result, local_n_row * sizeof(dtype)));
            cudaMemset(result, 0, local_n_row * sizeof(dtype));
        } else {
            result = NULL;
        }

        cudaEventRecord(start_spmv);
        // Each rank does spmv
        if (mode == 0) {
            csr_globmem_spmv_sequential<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, local_n_row, dense_vector, result);
        } else if (mode == 1) {
            csr_globmem_spmv_stride<<<blocks_per_grid, threads_per_block>>>(rowPtr, colIndexes, AVal, local_n_row, dense_vector, result);
        } else if (mode == 2) {
            csr_sharmem_spmv_stride<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, local_n_row, dense_vector, result);
        } else if (mode == 3) {
            csr_sharmem_coalesced<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(rowPtr, colIndexes, AVal, local_n_row, dense_vector, result);
        }
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaEventRecord(stop_spmv));
        gpuErrchk(cudaEventSynchronize(stop_spmv));

        // Reduce partial row-block results across the process columns, then gather all row blocks.
        if (my_coords[1] == 0) {
            if (local_n_row > 0) {
                row_reduced_result = (dtype*)malloc(local_n_row * sizeof(dtype));
            } else {
                row_reduced_result = NULL;
            }
        }

        MPI_Reduce(result, row_reduced_result, local_n_row, MPI_DTYPE, MPI_SUM, 0, row_comm);

        int* gather_recvcounts = (int*)malloc(comm_size * sizeof(int));
        int* gather_displs = (int*)malloc(comm_size * sizeof(int));
        build_allgatherv_layout(grid_comm, comm_size, global_n_row, grid_dims[0], grid_dims[1], gather_recvcounts, gather_displs);

        if (total_solution != NULL) {
            free(total_solution);
            total_solution = NULL;
        }

        dtype* gather_send_buffer = (my_coords[1] == 0) ? row_reduced_result : result;
        int gather_send_count = (my_coords[1] == 0) ? local_n_row : 0;
        total_solution = (dtype*)malloc(global_n_row * sizeof(dtype));
        MPI_Allgatherv(gather_send_buffer, gather_send_count, MPI_DTYPE, total_solution, gather_recvcounts, gather_displs, MPI_DTYPE, grid_comm);

        free(gather_recvcounts);
        free(gather_displs);

        if (row_reduced_result != NULL) {
            free(row_reduced_result);
            row_reduced_result = NULL;
        }

        // extract stats of the run
        float communication_time_ms;
        gpuErrchk(cudaEventElapsedTime(&communication_time_ms, start_reading_data, stop_reading_data));
        double communication_time_s = communication_time_ms / 1000.0;

        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start_spmv, stop_spmv);
        double exec_time_s = exec_time_ms / 1000.0;

        double bandwidth = csr_calculate_bandwidthGBs(local_n_col, local_n_row, local_nnz, exec_time_s);
        double gflop = calculate_gflop(local_nnz, exec_time_s);
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

    final_info_print(my_rank, communication_time_arr, exec_time_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, total_solution, global_n_row);
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

    if (rowPtr != NULL) {
        gpuErrchk(cudaFree(rowPtr));
    }
    if (colIndexes != NULL) {
        gpuErrchk(cudaFree(colIndexes));
    }
    if (AVal != NULL) {
        gpuErrchk(cudaFree(AVal));
    }
    if (my_rank == 0) {
        if (dense_vector_full != NULL) {
            gpuErrchk(cudaFree(dense_vector_full));
        }
        free(global_rowPtr);
        free(global_colIndexes);
        free(global_AVal);
    } else {
        if (dense_vector != NULL) {
            gpuErrchk(cudaFree(dense_vector));
        }
    }

    if (total_solution != NULL) {
        free(total_solution);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    nvmlShutdown();
    MPI_Finalize();
    return (0);
}
