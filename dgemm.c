#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <immintrin.h>

/* Problem size & threading */
#ifndef N
#define N 4096 /* matrix order */
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 12
#endif

/* Blocking parameters (tune once per CPU) */
#ifndef KC
#define KC 256
#endif
#ifndef MC
#define MC 6
#endif
#ifndef NC
#define NC 64
#endif
#define MR 6
#define NR 8

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/*
    6×8 AVX2 / FMA micro‑kernel
    A: MR × kc
    B: kc × NR
    C: MR × NR
*/
static inline void micro_kernel_6x8(
    int kc,
    const double *restrict A,
    const double *restrict B, int ldb,
    double *restrict C, int ldc
)
{
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (int p = 0; p < kc; ++p) {
        __m256d b0 = _mm256_loadu_pd(B + p * ldb + 0);
        __m256d b1 = _mm256_loadu_pd(B + p * ldb + 4);

        __m256d a;
        a = _mm256_broadcast_sd(A + 0 * kc + p);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);

        a = _mm256_broadcast_sd(A + 1 * kc + p);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);

        a = _mm256_broadcast_sd(A + 2 * kc + p);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);

        a = _mm256_broadcast_sd(A + 3 * kc + p);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);

        a = _mm256_broadcast_sd(A + 4 * kc + p);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);

        a = _mm256_broadcast_sd(A + 5 * kc + p);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);
    }

    _mm256_storeu_pd(
        C + 0 * ldc + 0,
        _mm256_add_pd(c00, _mm256_loadu_pd(C + 0 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 0 * ldc + 4,
        _mm256_add_pd(c01, _mm256_loadu_pd(C + 0 * ldc + 4))
    );

    _mm256_storeu_pd(
        C + 1 * ldc + 0,
        _mm256_add_pd(c10, _mm256_loadu_pd(C + 1 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 1 * ldc + 4,
        _mm256_add_pd(c11, _mm256_loadu_pd(C + 1 * ldc + 4))
    );

    _mm256_storeu_pd(
        C + 2 * ldc + 0,
        _mm256_add_pd(c20, _mm256_loadu_pd(C + 2 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 2 * ldc + 4,
        _mm256_add_pd(c21, _mm256_loadu_pd(C + 2 * ldc + 4))
    );

    _mm256_storeu_pd(
        C + 3 * ldc + 0,
        _mm256_add_pd(c30, _mm256_loadu_pd(C + 3 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 3 * ldc + 4,
        _mm256_add_pd(c31, _mm256_loadu_pd(C + 3 * ldc + 4))
    );

    _mm256_storeu_pd(
        C + 4 * ldc + 0,
        _mm256_add_pd(c40, _mm256_loadu_pd(C + 4 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 4 * ldc + 4,
        _mm256_add_pd(c41, _mm256_loadu_pd(C + 4 * ldc + 4))
    );

    _mm256_storeu_pd(
        C + 5 * ldc + 0,
        _mm256_add_pd(c50, _mm256_loadu_pd(C + 5 * ldc + 0))
    );
    _mm256_storeu_pd(
        C + 5 * ldc + 4,
        _mm256_add_pd(c51, _mm256_loadu_pd(C + 5 * ldc + 4))
    );
}

/* pack A (mc × kc) into contiguous buffer */
static inline void pack_A(int mc, int kc, const double *A, int lda, double *Apack)
{
    for (int i = 0; i < mc; ++i) {
        memcpy(Apack + i * kc, A + i * lda, kc * sizeof(double));
    }
}

/* pack B (kc × nc) into contiguous buffer */
static inline void pack_B(int kc, int nc, const double *B, int ldb, double *Bpack)
{
    for (int p = 0; p < kc; ++p) {
        memcpy(Bpack + p * nc, B + p * ldb, nc * sizeof(double));
    }
}

/* Threading */
typedef struct {
    double *A;
    double *B;
    double *C;
    int      jc_start, jc_end;
    double  *Apack;
    double  *Bpack;
} thread_arg_t;

/* blocked DGEMM for columns jc_start ... jc_end‑1 */
static void dgemm_slice(thread_arg_t *ts)
{
    for (int jc = ts->jc_start; jc < ts->jc_end; jc += NC) {
        int nc = MIN(NC, N - jc);

        for (int pc = 0; pc < N; pc += KC) {
            int kc = MIN(KC, N - pc);
            pack_B(kc, nc, ts->B + pc * N + jc, N, ts->Bpack);

            for (int ic = 0; ic < N; ic += MC) {
                int mc = MIN(MC, N - ic);
                pack_A(mc, kc, ts->A + ic * N + pc, N, ts->Apack);

                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                        micro_kernel_6x8(
                            kc,
                            ts->Apack + ir * kc,
                            ts->Bpack + jr,
                            nc,
                            ts->C + (ic + ir) * N + jc + jr,
                            N
                        );
                    }
                }
            }
        }
    }
}

static void *worker(void *arg)
{
    dgemm_slice((thread_arg_t *)arg);
    return NULL;
}

/* helpers */
static double wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static void *x_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

int main(void)
{
    /* aligned allocation (64 B) */
    double *A, *B, *C;
    A = x_aligned_alloc(64, (size_t)N * N * sizeof(double));
    B = x_aligned_alloc(64, (size_t)N * N * sizeof(double));
    C = x_aligned_alloc(64, (size_t)N * N * sizeof(double));

    /* initialise */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double)(i + j);
            B[i * N + j] = (double)(i - j);
            C[i * N + j] = 0.0;
        }
    }

    /* assign column blocks to threads */
    int jc_blocks = (N + NC - 1) / NC;
    int blocks_per_thr = (jc_blocks + NUM_THREADS - 1) / NUM_THREADS;

    pthread_t thr[NUM_THREADS];
    thread_arg_t  arg[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; ++t) {
        arg[t].A = A;
        arg[t].B = B;
        arg[t].C = C;

        arg[t].jc_start = t * blocks_per_thr * NC;
        arg[t].jc_end = (t + 1) * blocks_per_thr * NC;
        if (arg[t].jc_end > N) arg[t].jc_end = N;

        arg[t].Apack = x_aligned_alloc(64, MC * KC * sizeof(double));
        arg[t].Bpack = x_aligned_alloc(64, KC * NC * sizeof(double));
    }

    double t0 = wall_seconds();

    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_create(&thr[t], NULL, worker, &arg[t]);
    }

    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(thr[t], NULL);
    }

    double t1 = wall_seconds();
    printf("Time: %.3f s\n", t1 - t0);

    return 0;
}
