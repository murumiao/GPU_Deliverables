#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <nccl.h>
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

typedef struct {
    int rank;
    size_t host_bytes;
    size_t gpu_bytes;
    int n_row;
    int n_col;
    int nnz;
} MemoryRecord;

typedef enum {
    LOAD_BALANCE_UNIFORM = 0,
    LOAD_BALANCE_NNZ = 1,
    LOAD_BALANCE_COMM = 2,
} LoadBalanceType;

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

static void build_weighted_partition(const double* weights, int total, int parts, int* starts, int* counts) {
    if (parts <= 0) {
        return;
    }

    double total_weight = 0.0;
    for (int i = 0; i < total; i++) {
        total_weight += weights[i];
    }

    if (total_weight <= 0.0) {
        partition_range(total, parts, 0, &starts[0], &counts[0]);
        for (int p = 1; p < parts; p++) {
            partition_range(total, parts, p, &starts[p], &counts[p]);
        }
        return;
    }

    starts[0] = 0;
    int current_part = 0;
    double next_cut_weight = total_weight / parts;
    double running_weight = 0.0;

    for (int i = 0; i < total; i++) {
        if (current_part < parts - 1) {
            int remaining_parts = parts - current_part - 1;
            int remaining_items = total - i;
            if (running_weight >= next_cut_weight && remaining_items >= remaining_parts) {
                counts[current_part] = i - starts[current_part];
                current_part++;
                starts[current_part] = i;
                next_cut_weight = total_weight * (current_part + 1) / parts;
            }
        }
        running_weight += weights[i];
    }

    counts[current_part] = total - starts[current_part];
    for (int p = current_part + 1; p < parts; p++) {
        starts[p] = total;
        counts[p] = 0;
    }
}

double weighted_sum_range(const double* weights, int start, int count) {
    double sum = 0.0;
    for (int i = start; i < start + count; i++) {
        sum += weights[i];
    }
    return sum;
}
void write_memory_footprint_csv(const char* filename, int rank, LoadBalanceType lb_mode, int threads, int sharedmem, size_t host_bytes, size_t gpu_bytes, int n_row, int n_col, int nnz) {
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
        fprintf(fp, "rank, load balance mode, threads, sharedmem, n_row, n_col, nnz, host_memory_bytes, gpu_memory_bytes, total_memory_bytes, total_memory_megabytes\n");
        for (int i = 0; i < comm_size; i++) {
            fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%f\n", all[i].rank, lb_mode, threads, sharedmem, all[i].n_row, all[i].n_col, all[i].nnz, all[i].host_bytes, all[i].gpu_bytes, all[i].host_bytes + all[i].gpu_bytes, (all[i].host_bytes + all[i].gpu_bytes) / 1e6);
        }
        fclose(fp);
        free(all);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_balance_summary(const char* label, double local_value, MPI_Comm comm) {
    double min_value = 0.0;
    double max_value = 0.0;
    double sum_value = 0.0;
    MPI_Allreduce(&local_value, &min_value, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_value, &sum_value, 1, MPI_DOUBLE, MPI_SUM, comm);

    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    double avg_value = sum_value / comm_size;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Load balance (%s): min=%g max=%g avg=%g\n", label, min_value, max_value, avg_value);
    }
}

void build_allgatherv_layout(MPI_Comm grid_comm, int comm_size, int global_row_count, const int* row_starts, const int* row_counts, int* recvcounts, int* displs) {
    for (int rank = 0; rank < comm_size; rank++) {
        int coords[2];
        MPI_Cart_coords(grid_comm, rank, 2, coords);
        int row_coord = coords[0];
        recvcounts[rank] = (coords[1] == 0) ? row_counts[row_coord] : 0;
        displs[rank] = (coords[1] == 0) ? row_starts[row_coord] : 0;
    }
}

int parse_triplet_line(const char* line, int line_len, int* row, int* col, dtype* val) {
    char* buffer = (char*)malloc((size_t)line_len + 1);
    if (buffer == NULL) {
        return 0;
    }

    memcpy(buffer, line, (size_t)line_len);
    buffer[line_len] = '\0';

    char* token = strtok(buffer, " \t\r");
    if (token == NULL) {
        free(buffer);
        return 0;
    }

    char* endptr = NULL;
    long parsed_row = strtol(token, &endptr, 10);
    if (endptr == token) {
        free(buffer);
        return 0;
    }

    token = strtok(NULL, " \t\r");
    if (token == NULL) {
        free(buffer);
        return 0;
    }
    long parsed_col = strtol(token, &endptr, 10);
    if (endptr == token) {
        free(buffer);
        return 0;
    }

    token = strtok(NULL, " \t\r");
    if (token == NULL) {
        free(buffer);
        return 0;
    }
    dtype parsed_val = (dtype)strtod(token, &endptr);
    if (endptr == token) {
        free(buffer);
        return 0;
    }

    *row = (int)parsed_row - 1;
    *col = (int)parsed_col - 1;
    *val = parsed_val;

    free(buffer);
    return 1;
}

static int parse_matrix_market_header_from_buffer(const char* buffer, int buffer_len, int* n_row, int* n_col, int* nnz, int* header_bytes) {
    int offset = 0;
    int header_found = 0;

    while (offset < buffer_len) {
        int line_start = offset;
        while (offset < buffer_len && buffer[offset] != '\n') {
            offset++;
        }
        int line_len = offset - line_start;
        if (offset < buffer_len && buffer[offset] == '\n') {
            offset++;
        }

        if (line_len <= 0) {
            continue;
        }
        if (buffer[line_start] == '%') {
            continue;
        }

        char* line = (char*)malloc((size_t)line_len + 1);
        if (line == NULL) {
            return 0;
        }
        memcpy(line, &buffer[line_start], (size_t)line_len);
        line[line_len] = '\0';

        char* token = strtok(line, " \t\r");
        if (token == NULL) {
            free(line);
            continue;
        }

        *n_row = atoi(token);
        token = strtok(NULL, " \t\r");
        if (token == NULL) {
            free(line);
            return 0;
        }
        *n_col = atoi(token);
        token = strtok(NULL, " \t\r");
        if (token == NULL) {
            free(line);
            return 0;
        }
        *nnz = atoi(token);
        *header_bytes = offset;
        header_found = 1;
        free(line);
        break;
    }

    return header_found;
}

static void load_local_csr_block_mpiio(const char* matrix_path, MPI_Comm comm, int comm_size, int my_rank,
                                       int grid_dims[2], int my_coords[2], LoadBalanceType load_balance_mode,
                                       int* global_n_row, int* global_n_col, int* global_nnz,
                                       int* row_partition_starts, int* row_partition_counts,
                                       int* col_partition_starts, int* col_partition_counts,
                                       int* local_row_start, int* local_n_row,
                                       int* local_col_start, int* local_n_col,
                                       int* local_nnz, double* local_balance_metric,
                                       int** rowPtr, int** colIndexes, dtype** AVal, ncclComm_t nccl_comm) {
    MPI_File fh;
    MPI_Status status;
    MPI_Offset file_size = 0;
    MPI_Offset read_start = 0;
    MPI_Offset read_end = 0;
    MPI_Offset data_start = 0;
    MPI_Offset data_end = 0;
    int header_bytes = 0;
    int header_parsed = 0;

    MPI_File_open(comm, (char*)matrix_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &file_size);

    int prefix_size = (int)((file_size < 65536) ? file_size : 65536);
    while (!header_parsed) {
        char* prefix = (char*)malloc((size_t)prefix_size + 1);
        if (prefix == NULL) {
            fprintf(stderr, "Rank %d: failed to allocate MPI-IO prefix buffer\n", my_rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        MPI_File_read_at_all(fh, 0, prefix, prefix_size, MPI_BYTE, &status);
        prefix[prefix_size] = '\0';
        header_parsed = parse_matrix_market_header_from_buffer(prefix, prefix_size, global_n_row, global_n_col, global_nnz, &header_bytes);
        free(prefix);

        if (!header_parsed) {
            if (prefix_size >= file_size) {
                fprintf(stderr, "Rank %d: could not parse matrix header\n", my_rank);
                MPI_Abort(comm, EXIT_FAILURE);
            }
            prefix_size = prefix_size * 2;
            if (prefix_size > file_size) {
                prefix_size = (int)file_size;
            }
        }
    }

    data_start = header_bytes;
    data_end = file_size;
    MPI_Offset data_bytes = data_end - data_start;
    MPI_Offset chunk_start = data_start + (data_bytes * my_rank) / comm_size;
    MPI_Offset chunk_end = data_start + (data_bytes * (my_rank + 1)) / comm_size;

    const MPI_Offset overlap = 4096;
    read_start = (chunk_start > data_start + overlap) ? chunk_start - overlap : data_start;
    read_end = (chunk_end + overlap < file_size) ? chunk_end + overlap : file_size;
    if (read_end < read_start) {
        read_end = read_start;
    }

    int read_len = (int)(read_end - read_start);
    char* chunk = (char*)malloc((size_t)read_len + 1);
    if (chunk == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate MPI-IO chunk buffer\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_File_read_at_all(fh, read_start, chunk, read_len, MPI_BYTE, &status);
    chunk[read_len] = '\0';
    MPI_File_close(&fh);

    int* local_row_nnz = (int*)calloc((size_t)(*global_n_row), sizeof(int));
    int* local_col_nnz = (int*)calloc((size_t)(*global_n_col), sizeof(int));
    int* local_row_min_col = (int*)malloc((size_t)(*global_n_row) * sizeof(int));
    int* local_row_max_col = (int*)malloc((size_t)(*global_n_row) * sizeof(int));
    int* local_col_min_row = (int*)malloc((size_t)(*global_n_col) * sizeof(int));
    int* local_col_max_row = (int*)malloc((size_t)(*global_n_col) * sizeof(int));
    if (local_row_nnz == NULL || local_col_nnz == NULL || local_row_min_col == NULL || local_row_max_col == NULL || local_col_min_row == NULL || local_col_max_row == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate load-balance buffers\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int i = 0; i < *global_n_row; i++) {
        local_row_min_col[i] = *global_n_col;
        local_row_max_col[i] = -1;
    }
    for (int i = 0; i < *global_n_col; i++) {
        local_col_min_row[i] = *global_n_row;
        local_col_max_row[i] = -1;
    }

    *local_nnz = 0;
    MPI_Offset pos = 0;
    while (pos < read_len) {
        MPI_Offset line_start = pos;
        while (pos < read_len && chunk[pos] != '\n') {
            pos++;
        }
        int line_len = (int)(pos - line_start);
        if (pos < read_len && chunk[pos] == '\n') {
            pos++;
        }

        if (line_len <= 0 || chunk[line_start] == '%') {
            continue;
        }

        MPI_Offset absolute_line_start = read_start + line_start;
        if (absolute_line_start < chunk_start || absolute_line_start >= chunk_end) {
            continue;
        }

        int row = -1;
        int col = -1;
        dtype val = 0;
        if (!parse_triplet_line(&chunk[line_start], line_len, &row, &col, &val)) {
            continue;
        }

        if (row < 0 || row >= *global_n_row || col < 0 || col >= *global_n_col) {
            continue;
        }

        local_row_nnz[row]++;
        local_col_nnz[col]++;
        if (col < local_row_min_col[row]) local_row_min_col[row] = col;
        if (col > local_row_max_col[row]) local_row_max_col[row] = col;
        if (row < local_col_min_row[col]) local_col_min_row[col] = row;
        if (row > local_col_max_row[col]) local_col_max_row[col] = row;
        (*local_nnz)++;
    }

    int* global_row_nnz = (int*)malloc((size_t)(*global_n_row) * sizeof(int));
    int* global_col_nnz = (int*)malloc((size_t)(*global_n_col) * sizeof(int));
    int* global_row_min_col = (int*)malloc((size_t)(*global_n_row) * sizeof(int));
    int* global_row_max_col = (int*)malloc((size_t)(*global_n_row) * sizeof(int));
    int* global_col_min_row = (int*)malloc((size_t)(*global_n_col) * sizeof(int));
    int* global_col_max_row = (int*)malloc((size_t)(*global_n_col) * sizeof(int));
    if (global_row_nnz == NULL || global_col_nnz == NULL || global_row_min_col == NULL || global_row_max_col == NULL || global_col_min_row == NULL || global_col_max_row == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate global load-balance buffers\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    ncclGroupStart();
    ncclAllReduce(local_row_nnz, global_row_nnz, *global_n_row, ncclInt, ncclSum, nccl_comm, 0);
    ncclAllReduce(local_col_nnz, global_col_nnz, *global_n_col, ncclInt, ncclSum, nccl_comm, 0);
    ncclAllReduce(local_row_min_col, global_row_min_col, *global_n_row, ncclInt, ncclMin, nccl_comm, 0);
    ncclAllReduce(local_row_max_col, global_row_max_col, *global_n_row, ncclInt, ncclMax, nccl_comm, 0);
    ncclAllReduce(local_col_min_row, global_col_min_row, *global_n_col, ncclInt, ncclMin, nccl_comm, 0);
    ncclAllReduce(local_col_max_row, global_col_max_row, *global_n_col, ncclInt, ncclMax, nccl_comm, 0);
    ncclGroupEnd();
    double* row_weights = (double*)malloc((size_t)(*global_n_row) * sizeof(double));
    double* col_weights = (double*)malloc((size_t)(*global_n_col) * sizeof(double));
    if (row_weights == NULL || col_weights == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate weighted partition buffers\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for (int i = 0; i < *global_n_row; i++) {
        if (load_balance_mode == LOAD_BALANCE_UNIFORM) {
            row_weights[i] = 1.0;
        } else if (load_balance_mode == LOAD_BALANCE_NNZ) {
            row_weights[i] = (double)global_row_nnz[i];
        } else {
            row_weights[i] = (global_row_min_col[i] <= global_row_max_col[i]) ? (double)(global_row_max_col[i] - global_row_min_col[i] + 1) : 0.0;
        }
    }
    for (int i = 0; i < *global_n_col; i++) {
        if (load_balance_mode == LOAD_BALANCE_UNIFORM) {
            col_weights[i] = 1.0;
        } else if (load_balance_mode == LOAD_BALANCE_NNZ) {
            col_weights[i] = (double)global_col_nnz[i];
        } else {
            col_weights[i] = (global_col_min_row[i] <= global_col_max_row[i]) ? (double)(global_col_max_row[i] - global_col_min_row[i] + 1) : 0.0;
        }
    }

    build_weighted_partition(row_weights, *global_n_row, grid_dims[0], row_partition_starts, row_partition_counts);
    build_weighted_partition(col_weights, *global_n_col, grid_dims[1], col_partition_starts, col_partition_counts);

    *local_row_start = row_partition_starts[my_coords[0]];
    *local_n_row = row_partition_counts[my_coords[0]];
    *local_col_start = col_partition_starts[my_coords[1]];
    *local_n_col = col_partition_counts[my_coords[1]];

    data_start = header_bytes;
    data_end = file_size;
    data_bytes = data_end - data_start;
    chunk_start = data_start + (data_bytes * my_rank) / comm_size;
    chunk_end = data_start + (data_bytes * (my_rank + 1)) / comm_size;
    read_start = (chunk_start > data_start + overlap) ? chunk_start - overlap : data_start;
    read_end = (chunk_end + overlap < file_size) ? chunk_end + overlap : file_size;
    if (read_end < read_start) {
        read_end = read_start;
    }

    read_len = (int)(read_end - read_start);
    chunk = (char*)malloc((size_t)read_len + 1);
    if (chunk == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate MPI-IO chunk buffer\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_File_open(comm, (char*)matrix_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at_all(fh, read_start, chunk, read_len, MPI_BYTE, &status);
    chunk[read_len] = '\0';
    MPI_File_close(&fh);

    int* row_counts = (int*)calloc((size_t)(*local_n_row), sizeof(int));
    if (row_counts == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate row counts\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    pos = 0;
    while (pos < read_len) {
        MPI_Offset line_start = pos;
        while (pos < read_len && chunk[pos] != '\n') {
            pos++;
        }
        int line_len = (int)(pos - line_start);
        if (pos < read_len && chunk[pos] == '\n') {
            pos++;
        }

        if (line_len <= 0 || chunk[line_start] == '%') {
            continue;
        }

        MPI_Offset absolute_line_start = read_start + line_start;
        if (absolute_line_start < chunk_start || absolute_line_start >= chunk_end) {
            continue;
        }

        int row = -1;
        int col = -1;
        dtype val = 0;
        if (!parse_triplet_line(&chunk[line_start], line_len, &row, &col, &val)) {
            continue;
        }

        if (row >= *local_row_start && row < *local_row_start + *local_n_row && col >= *local_col_start && col < *local_col_start + *local_n_col) {
            row_counts[row - *local_row_start]++;
        }
    }

    *rowPtr = (int*)malloc((size_t)(*local_n_row + 1) * sizeof(int));
    *colIndexes = NULL;
    *AVal = NULL;
    (*rowPtr)[0] = 0;
    for (int i = 0; i < *local_n_row; i++) {
        (*rowPtr)[i + 1] = (*rowPtr)[i] + row_counts[i];
    }

    if (*local_nnz > 0) {
        *colIndexes = (int*)malloc((size_t)(*local_nnz) * sizeof(int));
        *AVal = (dtype*)malloc((size_t)(*local_nnz) * sizeof(dtype));
    }

    int* next_index = (int*)malloc((size_t)(*local_n_row) * sizeof(int));
    if (next_index == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate local fill indices\n", my_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (int i = 0; i < *local_n_row; i++) {
        next_index[i] = (*rowPtr)[i];
    }

    pos = 0;
    while (pos < read_len) {
        MPI_Offset line_start = pos;
        while (pos < read_len && chunk[pos] != '\n') {
            pos++;
        }
        int line_len = (int)(pos - line_start);
        if (pos < read_len && chunk[pos] == '\n') {
            pos++;
        }

        if (line_len <= 0 || chunk[line_start] == '%') {
            continue;
        }

        MPI_Offset absolute_line_start = read_start + line_start;
        if (absolute_line_start < chunk_start || absolute_line_start >= chunk_end) {
            continue;
        }

        int row = -1;
        int col = -1;
        dtype val = 0;
        if (!parse_triplet_line(&chunk[line_start], line_len, &row, &col, &val)) {
            continue;
        }

        if (row >= *local_row_start && row < *local_row_start + *local_n_row && col >= *local_col_start && col < *local_col_start + *local_n_col) {
            int local_row = row - *local_row_start;
            int out_index = next_index[local_row]++;
            (*colIndexes)[out_index] = col - *local_col_start;
            (*AVal)[out_index] = val;
        }
    }

    *local_balance_metric = weighted_sum_range(row_weights, *local_row_start, *local_n_row) + weighted_sum_range(col_weights, *local_col_start, *local_n_col);

    free(local_row_nnz);
    free(local_col_nnz);
    free(local_row_min_col);
    free(local_row_max_col);
    free(local_col_min_row);
    free(local_col_max_row);
    free(global_row_nnz);
    free(global_col_nnz);
    free(global_row_min_col);
    free(global_row_max_col);
    free(global_col_min_row);
    free(global_col_max_row);
    free(row_weights);
    free(col_weights);

    free(next_index);
    free(row_counts);
    free(chunk);
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
    if (argc < 5) {
        fprintf(stderr, "Usage %s <path_to_matrix> <mode[0,1,2,3]> <n_threads_per_block> <shared_mem_size> [--lb <0|1|2>] [--save <path>]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char* matrixPath = argv[1];
    int mode = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);
    int shared_mem_size = atoi(argv[4]);
    LoadBalanceType load_balance_mode = LOAD_BALANCE_UNIFORM;
    char* save_folder_path;
    save_folder_path = "results_2D/";

    for (int arg_i = 5; arg_i < argc; arg_i++) {
        if (strcmp(argv[arg_i], "--lb") == 0 && arg_i + 1 < argc) {
            load_balance_mode = (LoadBalanceType)atoi(argv[++arg_i]);
        } else if (strncmp(argv[arg_i], "--lb=", 5) == 0) {
            load_balance_mode = (LoadBalanceType)atoi(argv[arg_i] + 5);
        } else if (strcmp(argv[arg_i], "--save") == 0 && arg_i + 1 < argc) {
            save_folder_path = argv[++arg_i];
        } else if (strncmp(argv[arg_i], "--save=", 7) == 0) {
            save_folder_path = argv[arg_i] + 7;
        } else if (argv[arg_i][0] != '-' && load_balance_mode == LOAD_BALANCE_UNIFORM) {
            load_balance_mode = (LoadBalanceType)atoi(argv[arg_i]);
        } else if (argv[arg_i][0] != '-') {
            save_folder_path = argv[arg_i];
        }
    }

    int my_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    assign_GPU_to_rank(my_rank);
    ncclUniqueId id;

    if (my_rank == 0)
        ncclGetUniqueId(&id);

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, comm_size, id, my_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        fprintf(stderr, "Doing matrix=%s, mode=%d, threads=%d, shared_mem=%d, lb_mode=%d\n", matrixPath, mode, threads_per_block, shared_mem_size, load_balance_mode);
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

    int* row_partition_starts = (int*)malloc((size_t)grid_dims[0] * sizeof(int));
    int* row_partition_counts = (int*)malloc((size_t)grid_dims[0] * sizeof(int));
    int* col_partition_starts = (int*)malloc((size_t)grid_dims[1] * sizeof(int));
    int* col_partition_counts = (int*)malloc((size_t)grid_dims[1] * sizeof(int));
    if (row_partition_starts == NULL || row_partition_counts == NULL || col_partition_starts == NULL || col_partition_counts == NULL) {
        fprintf(stderr, "Rank %d: failed to allocate partition arrays\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int* rowPtr = NULL;
    int* colIndexes = NULL;
    dtype* AVal = NULL;
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

    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    //! START BENCHMARKING
    //! START BENCHMARKING
    //! START BENCHMARKING
    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    /* *** *** *** *** *** *** *** *** *** *** *** *** *** *** */
    for (int run_i = 0; run_i < TOTAL_RUNS; run_i++) {
        if (my_rank == 0)
            fprintf(stderr, "Run: %d/%d\n", run_i + 1, TOTAL_RUNS);
        gpuErrchk(cudaEventRecord(start_reading_data));
        double local_balance_metric = 0.0;
        load_local_csr_block_mpiio(matrixPath, grid_comm, comm_size, my_rank, grid_dims, my_coords, load_balance_mode, &global_n_row, &global_n_col, &global_nnz, row_partition_starts, row_partition_counts, col_partition_starts, col_partition_counts, &local_row_start, &local_n_row, &local_col_start, &local_n_col, &local_nnz, &local_balance_metric, &rowPtr, &colIndexes, &AVal, nccl_comm);

        if (run_i == 0) {
            if (load_balance_mode == LOAD_BALANCE_NNZ) {
                print_balance_summary("NNZ", (double)local_nnz, grid_comm);
            } else if (load_balance_mode == LOAD_BALANCE_COMM) {
                print_balance_summary("COMM", local_balance_metric, grid_comm);
            }
        }

        int* host_rowPtr = rowPtr;
        int* host_colIndexes = colIndexes;
        dtype* host_AVal = AVal;

        gpuErrchk(cudaMallocManaged(&rowPtr, (local_n_row + 1) * sizeof(int)));
        if (local_nnz > 0) {
            gpuErrchk(cudaMallocManaged(&colIndexes, local_nnz * sizeof(int)));
            gpuErrchk(cudaMallocManaged(&AVal, local_nnz * sizeof(dtype)));
            gpuErrchk(cudaMemcpy(rowPtr, host_rowPtr, (local_n_row + 1) * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(colIndexes, host_colIndexes, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(AVal, host_AVal, local_nnz * sizeof(dtype), cudaMemcpyHostToDevice));
        } else {
            colIndexes = NULL;
            AVal = NULL;
            for (int i = 0; i <= local_n_row; i++) {
                rowPtr[i] = host_rowPtr[i];
            }
        }

        free(host_rowPtr);
        free(host_colIndexes);
        free(host_AVal);

        if (local_n_col > 0) {
            gpuErrchk(cudaMallocManaged(&dense_vector, local_n_col * sizeof(dtype)));
            for (int i = 0; i < local_n_col; i++) {
                dense_vector[i] = 1.0;
            }
        } else {
            dense_vector = NULL;
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

        // Reduce partial row-block results across columns, then gather all row blocks.
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
        build_allgatherv_layout(grid_comm, comm_size, global_n_row, row_partition_starts, row_partition_counts, gather_recvcounts, gather_displs);

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

    char stats_file_name[512];
    sprintf(stats_file_name, "%s/rank_%d.csv", folder_name, my_rank);

    save_statistics(stats_file_name, TIMED_RUNS, communication_time_arr, exec_time_arr, bandwidth_arr, gflops_arr);
    size_t host_bytes = 0;
    size_t gpu_bytes = 0;
    host_bytes += (local_n_row + 1) * sizeof(int);
    host_bytes += local_nnz * sizeof(int);
    host_bytes += local_nnz * sizeof(double);
    host_bytes += global_n_row * sizeof(int) * 8;
    host_bytes += local_n_row * sizeof(int);
    host_bytes += grid_dims[0] * sizeof(int) * 2;
    host_bytes += grid_dims[1] * sizeof(int) * 2;
    host_bytes += local_n_row * sizeof(dtype);
    host_bytes += comm_size * sizeof(int) * 2;
    host_bytes += global_n_row * sizeof(dtype);

    gpu_bytes += (local_n_row + 1) * sizeof(dtype);
    gpu_bytes += local_nnz * sizeof(dtype) * 2;
    gpu_bytes += local_n_col * sizeof(dtype);
    gpu_bytes += local_n_row * sizeof(dtype);

    char mem_footprint_file_name[512];
    sprintf(mem_footprint_file_name, "%s/memory_footprint.csv", folder_name);

    write_memory_footprint_csv(mem_footprint_file_name, my_rank, load_balance_mode, threads_per_block, shared_mem_size, host_bytes, gpu_bytes, local_n_row, local_n_col, local_nnz);

    // cleanup
    if (rowPtr != NULL) {
        gpuErrchk(cudaFree(rowPtr));
    }
    if (colIndexes != NULL) {
        gpuErrchk(cudaFree(colIndexes));
    }
    if (AVal != NULL) {
        gpuErrchk(cudaFree(AVal));
    }
    if (dense_vector != NULL) {
        gpuErrchk(cudaFree(dense_vector));
    }

    free(row_partition_starts);
    free(row_partition_counts);
    free(col_partition_starts);
    free(col_partition_counts);

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
