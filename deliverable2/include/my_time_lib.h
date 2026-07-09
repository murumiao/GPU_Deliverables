#ifndef LAB1_EX2_LIB
#define LAB1_EX2_LIB

#include <math.h>
#include <sys/time.h>

#define TOTAL_RUNS 55
#define WARMUP_RUNS 5
#define TIMED_RUNS TOTAL_RUNS - WARMUP_RUNS

#define STR(s) #s
#define XSTR(s) STR(s)

#define TIMER_DEF(n) struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 + (temp_2_##n.tv_usec - temp_1_##n.tv_usec))
#define TIMER_PRINT(n)                                                        \
    do {                                                                      \
        int rk;                                                               \
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);                                   \
        if (rk == 0) printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n) / 1e6); \
        fflush(stdout);                                                       \
        sleep(0.5);                                                           \
        MPI_Barrier(MPI_COMM_WORLD);                                          \
    } while (0);

#ifdef __cplusplus
extern "C" {
#endif

double arithmetic_mean(double* v, int len);
double geometric_mean(double* v, int len);
double sigma_fn_sol(double* v, double mu, int len);

#ifdef __cplusplus
}
#endif

#endif