/*
    Correctness tests for dgemm_local: odd shapes (edges in m, n and k),
    submatrix views (ld > width), accumulation into a non-zero C, and
    thread counts from 1 to more threads than there is work.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/gemm_local.h"

#define SENTINEL 12345.678

static double rnd(void) { return 2.0 * rand() / RAND_MAX - 1.0; }

static int run_case(int m, int n, int k, int pad, int nthreads)
{
    int lda = k + pad, ldb = n + pad, ldc = n + pad;
    double *A = malloc((size_t)(m ? m : 1) * lda * sizeof(double));
    double *B = malloc((size_t)(k ? k : 1) * ldb * sizeof(double));
    double *C = malloc((size_t)(m ? m : 1) * ldc * sizeof(double));
    double *C0 = malloc((size_t)(m ? m : 1) * ldc * sizeof(double));

    for (size_t i = 0; i < (size_t)m * lda; ++i) A[i] = rnd();
    for (size_t i = 0; i < (size_t)k * ldb; ++i) B[i] = rnd();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j)
            C0[(size_t)i * ldc + j] = C[(size_t)i * ldc + j] = j < n ? rnd() : SENTINEL;

    dgemm_local(m, n, k, A, lda, B, ldb, C, ldc, nthreads);

    double maxerr = 0.0;
    int clobbered = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double ref = C0[(size_t)i * ldc + j], scale = fabs(ref);
            for (int p = 0; p < k; ++p) {
                double t = A[(size_t)i * lda + p] * B[(size_t)p * ldb + j];
                ref += t;
                scale += fabs(t);
            }
            double err = fabs(ref - C[(size_t)i * ldc + j]) / (scale + 1e-300);
            if (err > maxerr) maxerr = err;
        }
        for (int j = n; j < ldc; ++j)
            if (C[(size_t)i * ldc + j] != SENTINEL) clobbered = 1;
    }

    int ok = maxerr < 1e-13 && !clobbered;
    if (!ok)
        printf("FAIL m=%d n=%d k=%d pad=%d threads=%d: max rel err %.3e%s\n",
               m, n, k, pad, nthreads, maxerr, clobbered ? ", wrote outside C" : "");
    free(A); free(B); free(C); free(C0);
    return ok;
}

int main(void)
{
    static const int shapes[][3] = {
        {0, 5, 5}, {5, 0, 5}, {5, 5, 0},            /* empty: must be a no-op */
        {1, 1, 1}, {1, 1, 17}, {5, 7, 3}, {6, 8, 1}, /* smaller than one MR × NR tile */
        {13, 9, 11}, {7, 9, 385}, {12, 16, 769},     /* edges in m and n, k past KC */
        {100, 257, 300}, {257, 513, 129},            /* n past NC */
        {1000, 8, 50}, {8, 1000, 50}, {1, 999, 7},   /* tall and wide */
        {600, 600, 600}, {1031, 777, 413},
    };
    static const int threads[] = {1, 3, 16, 64};
    static const int pads[] = {0, 5};

    srand(42);
    int total = 0, passed = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s)
        for (size_t t = 0; t < sizeof threads / sizeof threads[0]; ++t)
            for (size_t p = 0; p < sizeof pads / sizeof pads[0]; ++p) {
                ++total;
                passed += run_case(shapes[s][0], shapes[s][1], shapes[s][2],
                                   pads[p], threads[t]);
            }

    printf("%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
