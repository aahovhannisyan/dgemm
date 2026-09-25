#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "src/gemm_local.h"

/* Problem size & threading defaults (override at runtime: ./dgemm [N] [threads]) */
#ifndef N_DEFAULT
#define N_DEFAULT 4096 /* matrix order */
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 12 /* override with -DNUM_THREADS=n */
#endif

/* helpers */
static double wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static void *x_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : N_DEFAULT;
    int nthreads = argc > 2 ? atoi(argv[2]) : NUM_THREADS;
    if (n <= 0 || nthreads <= 0) {
        fprintf(stderr, "usage: %s [N] [threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* aligned allocation (64 B) */
    size_t nn = (size_t)n * n;
    double *A = x_aligned_alloc(64, nn * sizeof(double));
    double *B = x_aligned_alloc(64, nn * sizeof(double));
    double *C = x_aligned_alloc(64, nn * sizeof(double));

    /* initialise */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[(size_t)i * n + j] = (double)(i + j);
            B[(size_t)i * n + j] = (double)(i - j);
            C[(size_t)i * n + j] = 0.0;
        }
    }

    double t0 = wall_seconds();
    dgemm_local(n, n, n, A, n, B, n, C, n, nthreads);
    double t1 = wall_seconds();

    printf("Time: %.3f s (%.1f GFLOP/s, %d threads)\n", t1 - t0,
           2.0 * n * n * n / (t1 - t0) * 1e-9, nthreads);

#ifdef VERIFY
    /* exact check of every entry: sum_k (i+k)(k-j) = (i-j)*S1 - n*i*j + S2,
       evaluated in integers so -ffast-math cannot reassociate it */
    long long S1 = (long long)n * (n - 1) / 2;
    long long S2 = (long long)(n - 1) * n * (2LL * n - 1) / 6;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double ref = (double)((i - j) * S1 - (long long)n * i * j + S2);
            double err = fabs(ref - C[(size_t)i * n + j]) / (fabs(ref) + 1.0);
            if (err > maxerr) maxerr = err;
        }
    }
    printf("Verify: max rel err %.3e (%s)\n", maxerr, maxerr < 1e-9 ? "OK" : "FAIL");
    if (maxerr >= 1e-9) return 1;
#endif

    free(A);
    free(B);
    free(C);
    return 0;
}
