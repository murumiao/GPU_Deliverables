#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING 100
#define NUMBER_ITERATIONS 1

void readMatrixFile(char* filePath, int** ARow, int** ACol, double** AVal, int* n_row, int* n_col, int* n_value) {
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL) {
        fprintf(stderr, "File %s not found", filePath);
        exit(1);
    }
    char buffer[MAX_STRING];
    int i = 0;
    while (fgets(buffer, MAX_STRING, fp)) {
        if (buffer[0] == '%') continue;
        if (*n_row == -1 || *n_col == -1 || *n_value == -1) {
            char* token = strtok(buffer, " ");
            *n_row = atoi(token);
            token = strtok(NULL, " ");
            *n_col = atoi(token);
            token = strtok(NULL, " ");
            *n_value = atoi(token);
            *ARow = malloc(*n_value * sizeof(int));
            *ACol = malloc(*n_value * sizeof(int));
            *AVal = malloc(*n_value * sizeof(double));
        } else {
            char* token = strtok(buffer, " ");
            int row = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            int col = atoi(token) - 1;  // 1-based index
            token = strtok(NULL, " ");
            double val = atof(token);
            (*ARow)[i] = row;
            (*ACol)[i] = col;
            (*AVal)[i] = val;
            i++;
        }
    }
    fclose(fp);
}
void compute_multiplication(int* ARow, int* ACol, double* AVal, double* v, int nnz, double* result) {
    for (int i = 0; i < nnz; i++) {
        result[ARow[i]] += AVal[i] * v[ACol[i]];
    }
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage %s <path_to_matrix>\n", argv[0]);
        exit(1);
    }

    int n_row = -1, n_col = -1, n_value = -1;
    int *ARow = NULL, *ACol = NULL;
    double* Aval = NULL;  // COO storage

    readMatrixFile(argv[1], &ARow, &ACol, &Aval, &n_row, &n_col, &n_value);
    // for (int i = 0; i < n_value; i++) {
    //     printf("%d\t%d\t%f\n", ARow[i], ACol[i], Aval[i]);
    // }

    // Create dense vector
    double* v = malloc(n_col * sizeof(double));
    for (int i = 0; i < n_col; i++) {
        v[i] = 1.0;
    }
    double* result = malloc(n_value * sizeof(double));
    for (int i = 0; i < NUMBER_ITERATIONS; i++) {
        memset(result, 0, n_row * sizeof(double));
        compute_multiplication(ARow, ACol, Aval, v, n_value, result);

        printf("---[%d]\n", i);
        for (int j = 0; j < n_row; j++) {
            printf("%f\n", result[j]);
        }
    }
    free(ARow);
    free(ACol);
    free(Aval);
    return 0;
}