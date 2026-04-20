#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_time_lib.h"
#include "../include/spmv_utils.h"

#define MAX_STRING 100
#define TOTAL_RUNS 10
#define WARMUP_RUNS 2

#define TIMED_RUNS TOTAL_RUNS - WARMUP_RUNS

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
            *rowPtr = malloc((*n_row + 1) * sizeof(int));
            *colIndexes = malloc(*nnz * sizeof(int));
            *valCSR = malloc(*nnz * sizeof(dtype));
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

void csr_spmv(int* rowPtr, int* colIndexes, dtype* AVal, int rowLen, dtype* v, dtype* result) {
    for (int i = 0; i < rowLen; i++) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            result[i] += AVal[j] * v[colIndexes[j]];
        }
    }
}

void checkCorrect(dtype* result, int n) {
    if (!(result[0] == 3.3 && result[1] == 2.5 && result[2] == 3)) {
        printf("Fail!\n");
        for (int j = 0; j < n; j++) {
            printf("%f,", result[j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    printf("CSR\n");
    if (argc != 2) {
        fprintf(stderr, "Usage %s <path_to_matrix>\n", argv[0]);
        exit(1);
    }

    //* CSR storage
    int n_row = -1, n_col = -1, nnz = -1;
    int *rowPtr = NULL, *colIndexes = NULL;
    dtype* AVal = NULL;

    printf("Loading matrix...\n");
    readMatrixFile(argv[1], &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
    printf("Matrix loaded!\n");

    // for (int i = 0; i < nnz; i++) {
    //     printf("%d\t%f\n", colIndexes[i], AVal[i]);
    // }
    // printf("\n");

    // Create dense vector
    dtype* v = malloc(n_col * sizeof(dtype));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    dtype* result = malloc(n_row * sizeof(dtype));
    double timers[TIMED_RUNS];

    TIMER_DEF(0);
    for (int i = 0; i < TOTAL_RUNS; i++) {
        memset(result, 0, n_row * sizeof(dtype));
        TIMER_START(0);
        csr_spmv(rowPtr, colIndexes, AVal, n_row, v, result);
        TIMER_STOP(0);
        if (strcmp(argv[1], "test.mtx") == 0) checkCorrect(result, n_row);
        if (i > WARMUP_RUNS) {
            timers[i - WARMUP_RUNS] = TIMER_ELAPSED(0) / 1.e6;
        }
    }
    free(rowPtr);
    free(colIndexes);
    free(AVal);
    free(v);
    return 0;
}