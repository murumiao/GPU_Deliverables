#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double val;
} Entry;

static int cmp(const void *a, const void *b) {
    const Entry *ea = (const Entry *)a;
    const Entry *eb = (const Entry *)b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.mtx\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    FILE *fin = fopen(input_path, "r");
    if (!fin) {
        perror("fopen input");
        return 1;
    }

    char buf[512];
    do {
        if (!fgets(buf, sizeof(buf), fin)) {
            fprintf(stderr, "Invalid or empty file.\n");
            return 1;
        }
    } while (buf[0] == '%');

    int nrows, ncols, nnz;
    if (sscanf(buf, "%d %d %d", &nrows, &ncols, &nnz) != 3) {
        fprintf(stderr, "Invalid matrix size line.\n");
        return 1;
    }

    Entry *entries = malloc(sizeof(Entry) * nnz);
    if (!entries) {
        perror("malloc");
        return 1;
    }

    for (int i = 0; i < nnz; ++i) {
        if (fscanf(fin, "%d %d %lf", &entries[i].row, &entries[i].col, &entries[i].val) != 3) {
            fprintf(stderr, "Error reading entry at index %d.\n%d %d %lf\n", i, entries[i].row, entries[i].col, entries[i].val);
            free(entries);
            return 1;
        }
    }
    fclose(fin);

    qsort(entries, nnz, sizeof(Entry), cmp);

    // Create output file name
    char output_path[512];
    int last_position_slash = -1;
    for (int i = 0; input_path[i] != '\0'; i++) {
        if (input_path[i] == '/') {
            last_position_slash = i;
        }
    }
    last_position_slash++;
    strncpy(output_path, input_path, last_position_slash);
    strcat(output_path, "sorted_");
    strcat(output_path, (input_path + last_position_slash));

    FILE *fout = fopen(output_path, "w");
    if (!fout) {
        perror("fopen output");
        free(entries);
        return 1;
    }

    fprintf(fout, "%%%%Sorted matrix file\n");
    fprintf(fout, "%d %d %d\n", nrows, ncols, nnz);
    for (int i = 0; i < nnz; ++i)
        fprintf(fout, "%d %d %.12g\n", entries[i].row, entries[i].col, entries[i].val);

    fclose(fout);
    free(entries);
    printf("Wrote sorted matrix to %s\n", output_path);
    return 0;
}