#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#else
#error "Unsupported architecture: need AArch64 NEON or x86 AVX2+FMA"
#endif

#include "gemm_local.h"

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

_Static_assert(MC % MR == 0, "MC must be a multiple of MR");
_Static_assert(NC % NR == 0, "NC must be a multiple of NR");

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))
#define ROUND_UP(a,b) (CEIL_DIV(a,b) * (b))

#if defined(__aarch64__)
/*
    6×8 NEON micro-kernel (128-bit, 2 doubles per vector)
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
    float64x2_t c[MR][4];
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < 4; ++j)
            c[i][j] = vdupq_n_f64(0.0);

    for (int p = 0; p < kc; ++p) {
        const double *bp = B + p * ldb;
        float64x2_t b0 = vld1q_f64(bp + 0);
        float64x2_t b1 = vld1q_f64(bp + 2);
        float64x2_t b2 = vld1q_f64(bp + 4);
        float64x2_t b3 = vld1q_f64(bp + 6);

        for (int i = 0; i < MR; ++i) {
            float64x2_t a = vld1q_dup_f64(A + i * kc + p);
            c[i][0] = vfmaq_f64(c[i][0], a, b0);
            c[i][1] = vfmaq_f64(c[i][1], a, b1);
            c[i][2] = vfmaq_f64(c[i][2], a, b2);
            c[i][3] = vfmaq_f64(c[i][3], a, b3);
        }
    }

    for (int i = 0; i < MR; ++i) {
        double *cp = C + i * ldc;
        for (int j = 0; j < 4; ++j)
            vst1q_f64(cp + 2 * j, vaddq_f64(c[i][j], vld1q_f64(cp + 2 * j)));
    }
}
#else
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

#endif

/* pack A (mc × kc) into contiguous buffer, zero-padding rows up to a multiple of MR */
static inline void pack_A(int mc, int kc, const double *A, int lda, double *Apack)
{
    for (int i = 0; i < mc; ++i) {
        memcpy(Apack + i * kc, A + (size_t)i * lda, kc * sizeof(double));
    }
    int mcp = ROUND_UP(mc, MR);
    if (mcp > mc)
        memset(Apack + mc * kc, 0, (size_t)(mcp - mc) * kc * sizeof(double));
}

/* pack B (kc × nc) into a buffer with row stride ncp, zero-padding columns nc ... ncp-1 */
static inline void pack_B(int kc, int nc, int ncp, const double *B, int ldb, double *Bpack)
{
    for (int p = 0; p < kc; ++p) {
        memcpy(Bpack + p * ncp, B + (size_t)p * ldb, nc * sizeof(double));
        if (ncp > nc)
            memset(Bpack + p * ncp + nc, 0, (ncp - nc) * sizeof(double));
    }
}

/* Threading */
typedef struct {
    const double *A;
    const double *B;
    double       *C;
    int lda, ldb, ldc, k;
    int i_start, i_end;     /* rows of C owned by this thread */
    int jc_start, jc_end;   /* columns of C owned by this thread */
} thread_arg_t;

/* blocked DGEMM for C[i_start ... i_end-1][jc_start ... jc_end-1] */
static void dgemm_slice(const thread_arg_t *ts, double *Apack, double *Bpack)
{
    for (int jc = ts->jc_start; jc < ts->jc_end; jc += NC) {
        int nc = MIN(NC, ts->jc_end - jc);
        int ncp = ROUND_UP(nc, NR);

        for (int pc = 0; pc < ts->k; pc += KC) {
            int kc = MIN(KC, ts->k - pc);
            pack_B(kc, nc, ncp, ts->B + (size_t)pc * ts->ldb + jc, ts->ldb, Bpack);

            for (int ic = ts->i_start; ic < ts->i_end; ic += MC) {
                int mc = MIN(MC, ts->i_end - ic);
                pack_A(mc, kc, ts->A + (size_t)ic * ts->lda + pc, ts->lda, Apack);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = MIN(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = MIN(MR, mc - ir);
                        double *Cp = ts->C + (size_t)(ic + ir) * ts->ldc + jc + jr;
                        if (mr == MR && nr == NR) {
                            micro_kernel_6x8(kc, Apack + ir * kc,
                                             Bpack + jr, ncp, Cp, ts->ldc);
                        } else {
                            /* edge tile: the kernel always writes MR × NR, so
                               accumulate into scratch and copy the valid part */
                            double tmp[MR * NR] = {0};
                            micro_kernel_6x8(kc, Apack + ir * kc,
                                             Bpack + jr, ncp, tmp, NR);
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    Cp[(size_t)i * ts->ldc + j] += tmp[i * NR + j];
                        }
                    }
                }
            }
        }
    }
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

static void *worker(void *arg)
{
    double *Apack = x_aligned_alloc(64, MC * KC * sizeof(double));
    double *Bpack = x_aligned_alloc(64, KC * NC * sizeof(double));
    dgemm_slice((const thread_arg_t *)arg, Apack, Bpack);
    free(Apack);
    free(Bpack);
    return NULL;
}

/*
    Pick a tr × tc thread grid (tr * tc <= nthreads) minimising the largest
    per-thread tile, measured in NC-wide column blocks × MR-high row blocks.
    Ties go to more column groups: threads then share A and pack disjoint B.
*/
static void thread_grid(int m, int n, int nthreads, int *tr, int *tc)
{
    int nbc = CEIL_DIV(n, NC), nbr = CEIL_DIV(m, MR);
    long best = LONG_MAX;
    *tr = *tc = 1;
    for (int c = MIN(nbc, nthreads); c >= 1; --c) {
        int r = MIN(nthreads / c, nbr);
        long cost = (long)CEIL_DIV(nbc, c) * CEIL_DIV(nbr, r);
        if (cost < best) {
            best = cost;
            *tr = r;
            *tc = c;
        }
    }
}

void dgemm_local(int m, int n, int k,
                 const double *A, int lda,
                 const double *B, int ldb,
                 double *C, int ldc,
                 int nthreads)
{
    if (m <= 0 || n <= 0 || k <= 0) return;
    if (nthreads < 1) nthreads = 1;

    int tr, tc;
    thread_grid(m, n, nthreads, &tr, &tc);
    int nt = tr * tc;
    int nbc = CEIL_DIV(n, NC), nbr = CEIL_DIV(m, MR);

    thread_arg_t *arg = malloc(nt * sizeof(*arg));
    pthread_t *thr = malloc(nt * sizeof(*thr));
    if (!arg || !thr) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    /* balanced split: NC-aligned column blocks, MR-aligned row blocks */
    for (int t = 0; t < nt; ++t) {
        int r = t / tc, c = t % tc;
        arg[t] = (thread_arg_t){
            .A = A, .B = B, .C = C,
            .lda = lda, .ldb = ldb, .ldc = ldc, .k = k,
            .i_start  = MIN(m, (int)((long)r * nbr / tr) * MR),
            .i_end    = MIN(m, (int)((long)(r + 1) * nbr / tr) * MR),
            .jc_start = MIN(n, (int)((long)c * nbc / tc) * NC),
            .jc_end   = MIN(n, (int)((long)(c + 1) * nbc / tc) * NC),
        };
    }

    if (nt == 1) {
        worker(&arg[0]);
    } else {
        for (int t = 0; t < nt; ++t) {
            if (pthread_create(&thr[t], NULL, worker, &arg[t]) != 0) {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }
        }
        for (int t = 0; t < nt; ++t) {
            pthread_join(thr[t], NULL);
        }
    }

    free(arg);
    free(thr);
}
