#ifndef GEMM_LOCAL_H
#define GEMM_LOCAL_H

/*
    C += A * B for row-major matrices on a single shared-memory node.
    A: m × k (leading dimension lda)
    B: k × n (leading dimension ldb)
    C: m × n (leading dimension ldc)
    Work is split across nthreads POSIX threads (nthreads < 1 means 1).
*/
void dgemm_local(int m, int n, int k,
                 const double *A, int lda,
                 const double *B, int ldb,
                 double *C, int ldc,
                 int nthreads);

#endif
