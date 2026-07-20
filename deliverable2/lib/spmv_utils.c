#include "../include/spmv_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../include/my_time_lib.h"

struct stat st = {0};

void print_nnz_head_spmv(dtype* result_arr, int len_result, int n) {
    printf("First %d nnz values result\t", (int)fmin(n, len_result));
    double toPrint[n];
    int counter = 0;
    for (int i = 0; i < n && i < len_result; i++) {
        if (result_arr[i] != 0) {
            toPrint[counter++] = result_arr[i];
        }
    }
    if (counter != 0) {
        for (int i = 0; i < counter; i++) {
            printf("%f | ", toPrint[i]);
        }
        printf("\n");
    } else {
        printf("All elements were 0s\n");
    }
}

void print_starting_info(const char* type, const char* matrix_name, int amount_runs, int amount_warmpup, int block_size, int threads_per_block, int sharedmem_size) {
    printf("========%s========\n", type);
    printf("Matrix: %s\n", matrix_name);
    printf("Timed runs\t %d\n", amount_runs);
    printf("Warmup runs\t %d\n", amount_warmpup);
    printf("Blocks\t %d\n", block_size);
    printf("Threads\t %d\n", threads_per_block);
    printf("SharMem \t %d\n", sharedmem_size);
}
void print_run_stat(int runid, double exec_time, double bandwidth, double gflops) {
    printf("========Run #%d========\n", runid + 1);
    printf("Execution TIME (s)\t\t!%f\n", exec_time);
    printf("Estimated BANDWIDTH(GB/s)\t!%f\n", bandwidth);
    printf("Estimated GFLOPS\t\t!%f\n", gflops);
}

void final_info_print(double* timers, double* bandwidths, double* gflops, int amount_runs, dtype* results, int len_results) {
    printf("========STATS========\n");
    printf("Arithmetic mean TIME \t\t!%f s\n", arithmetic_mean(timers, amount_runs));
    printf("Arithmetic mean BANDWIDTH\t!%f GB/s\n", arithmetic_mean(bandwidths, amount_runs));
    printf("Arithmetic mean FLOPS\t\t!%f GFLOPS\n", arithmetic_mean(gflops, amount_runs));
    print_nnz_head_spmv(results, len_results, 5);
}

double coo_calculate_bandwidthGBs(int n_col, int n_row, int nnz, double time_s) {
    // effective bandwith = ((Byte_read+Byte_written)/10^9)/time
    if (time_s == 0.0) {
        return INFINITY;
    }

    long long int byte_read_matrix = 2 * nnz * sizeof(int) + nnz * sizeof(dtype);
    long long int byte_read_vector = n_col * sizeof(dtype);  // assuming worst cache
    long long int byte_read = byte_read_matrix + byte_read_vector;

    long long int byte_written_to_result = n_row * sizeof(dtype);

    long double gigabyte_used = (byte_read + byte_written_to_result) / 1.e9;
    return gigabyte_used / time_s;
}

double csr_calculate_bandwidthGBs(int n_col, int n_row, int nnz, double time_s) {
    // effective bandwith = ((Byte_read+Byte_written)/10^9)/time
    if (time_s == 0.0) {
        return INFINITY;
    }
    long long int byte_read_matrix = nnz * sizeof(int) + nnz * sizeof(dtype);
    long long int byte_read_row_ptr = (n_row + 1) * sizeof(int);
    long long int byte_read_vector = nnz * sizeof(dtype);  // assuming worst cache

    long long int bytes_read = byte_read_matrix + byte_read_row_ptr + byte_read_vector;

    long long int byte_written_to_result = n_row * sizeof(dtype);
    long double gigabyte_used = (bytes_read + byte_written_to_result) / 1.e9;

    return gigabyte_used / time_s;
}

double calculate_gflop(int nnz, double time_s) {
    // GFLOP=2*operations / time_s
    return (2 * nnz / 1.e9) / time_s;
}

void save_statistics(char* file_name, int len, double* communication, double* exec, double* bandwidth, double* gflops) {
    FILE* fptr;
    fptr = fopen(file_name, "w");
    fprintf(fptr, "ID, Communication(s), Execution(s), Bandwidth(GB/s), FLOPS(GFLOPS)\n");
    for (int i = 0; i < len; i++) {
        fprintf(fptr, "%d, %f, %f, %f, %f\n", i, communication[i], exec[i], bandwidth[i], gflops[i]);
    }
    fclose(fptr);
}
void make_dir(char* dirname, int my_rank) {
    if (my_rank == 0) {
        if (stat(dirname, &st) == -1) {
            mkdir(dirname, 0777);
        }
    }
}