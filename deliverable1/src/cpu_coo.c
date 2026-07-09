#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100


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
            *ARow = malloc(*nnz * sizeof(int));
            *ACol = malloc(*nnz * sizeof(int));
            *AVal = malloc(*nnz * sizeof(dtype));
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
void coo_globmem_spmv_seq(int* ARow, int* ACol, dtype* AVal, dtype* v, int n, dtype* result) {
    for (int i = 0; i < n; i++) {
        result[ARow[i]] += AVal[i] * v[ACol[i]];
    }
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage %s <path_to_matrix>\n", argv[0]);
        exit(1);
    }

    //* COO storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *ARow = NULL, *ACol = NULL;
    dtype* Aval = NULL;

    readMatrixFile(argv[1], &ARow, &ACol, &Aval, &n_row, &n_col, &nnz);

    print_starting_info("COO CPU", argv[1], TIMED_RUNS, WARMUP_RUNS,-1,-1,-1);
    // Create dense vector
    dtype* v = malloc(n_col * sizeof(dtype));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    dtype* result = malloc(n_row * sizeof(dtype));
    double timer_arr[TIMED_RUNS], bandwidth_arr[TIMED_RUNS], gflops_arr[TIMED_RUNS];

    TIMER_DEF(0);
    for (int i = 0; i < TOTAL_RUNS; i++) {
        memset(result, 0, n_row * sizeof(dtype));
        TIMER_START(0);
        coo_globmem_spmv_seq(ARow, ACol, Aval, v, nnz, result);
        TIMER_STOP(0);
        double exec_time_s = TIMER_ELAPSED(0) / 1.e6;
        double bandwidth = coo_calculate_bandwidthGBs(n_col, n_row, nnz, exec_time_s);
        double gflop = calculate_gflop(nnz, exec_time_s);
        if (i > WARMUP_RUNS) {
            timer_arr[i - WARMUP_RUNS] = exec_time_s;
            bandwidth_arr[i - WARMUP_RUNS] = bandwidth;
            gflops_arr[i - WARMUP_RUNS] = gflop;
        }
        print_run_stat(i, exec_time_s, bandwidth, gflop);
    }
    final_info_print(timer_arr, bandwidth_arr, gflops_arr, TIMED_RUNS, result, n_row);
    free(ARow);
    free(ACol);
    free(Aval);
    free(v);
    return 0;
}