#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./TIMER_LIB/my_time_lib.h"

#define MAX_STRING 100
#define NUMBER_ITERATIONS 10
#define WARMUP_ITERATIONS 2

#define TIMED_ITERS NUMBER_ITERATIONS - WARMUP_ITERATIONS
void sortFile(char* filePath) {
}
void readMatrixFile(char* filePath, int** rowPtr, int** colIndexes, double** valCSR, int* n_row, int* n_col, int* nnz) {
    sortFile(filePath);
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL) {
        fprintf(stderr, "File %s not found", filePath);
        exit(1);
    }
    char buffer[MAX_STRING];
    int i = 0, j = 0;
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
            *valCSR = malloc(*nnz * sizeof(double));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            double val = atof(token);
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

void csr_spmv(int* rowPtr, int* colIndexes, double* AVal, int nnz, int rowLen, double* v, double* result) {
    for (int i = 0; i < rowLen; i++) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            result[i] += AVal[j] * v[colIndexes[j]];
        }
    }
}

void checkCorrect(double* result, int n) {
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
    double* AVal = NULL;

    printf("Loading matrix...\n");
    readMatrixFile(argv[1], &rowPtr, &colIndexes, &AVal, &n_row, &n_col, &nnz);
    printf("Matrix loaded!\n");

    // for (int i = 0; i < nnz; i++) {
    //     printf("%d\t%f\n", colIndexes[i], AVal[i]);
    // }
    // printf("\n");

    // Create dense vector
    double* v = malloc(n_col * sizeof(double));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }

    double* result = malloc(n_row * sizeof(double));
    double timers[TIMED_ITERS];

    TIMER_DEF(0);
    for (int i = 0; i < NUMBER_ITERATIONS; i++) {
        memset(result, 0, n_row * sizeof(double));
        TIMER_START(0);
        csr_spmv(rowPtr, colIndexes, AVal, nnz, n_row, v, result);
        TIMER_STOP(0);
        if (strcmp(argv[1], "test.mtx") == 0) checkCorrect(result, n_row);
        if (i > WARMUP_ITERATIONS) {
            timers[i - WARMUP_ITERATIONS] = TIMER_ELAPSED(0) / 1.e6;
        }
        printf("Run %d/%d done in %fs\n", i + 1, NUMBER_ITERATIONS, TIMER_ELAPSED(0) / 1.e6);
    }

    printf("Arithmetic mean on %d iterations: %fs\n", TIMED_ITERS, arithmetic_mean(timers, TIMED_ITERS));
    free(rowPtr);
    free(colIndexes);
    free(AVal);
    free(v);
    return 0;
}