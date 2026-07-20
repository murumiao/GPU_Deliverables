#ifndef SPMV_UTILS
#define SPMV_UTILS

#ifdef __cplusplus
extern "C" {
#endif
#define dtype float
#define MPI_DTYPE MPI_FLOAT

void print_starting_info(const char* type, const char* matrix_name, int amount_runs, int amount_warmpup, int block_size, int threads_per_block, int sharedmem_size);
void print_run_stat(int runid, double exec_time, double bandwidth, double gflops);
void final_info_print(double* timers, double* bandwidths, double* gflops, int amount_runs, dtype* results, int len_results);

double coo_calculate_bandwidthGBs(int n_col, int n_row, int nnz, double time_s);
double csr_calculate_bandwidthGBs(int n_col, int n_row, int nnz, double time_s);
double calculate_gflop(int nnz, double time_s);

void save_statistics(char* file_name, int len, double* communication, double* exec, double* bandwidth, double* gflops);
void make_dir(char* dirname, int my_rank);
#ifdef __cplusplus
}
#endif
#endif
