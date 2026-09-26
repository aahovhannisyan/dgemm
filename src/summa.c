/*
    Distributed DGEMM (C = A × B, all N × N) with SUMMA over a pr × pc MPI grid.

    Every matrix is 2D block-distributed: rank (r, c) owns rows part(r, pr)
    and columns part(c, pc).  For each k-panel, the process column owning
    that slice of A broadcasts it along its process row, the process row
    owning that slice of B broadcasts it along its process column, and every
    rank accumulates C_loc += A_panel × B_panel with the threaded
    dgemm_local().  Panel broadcasts for step s+1 are posted (MPI_Ibcast)
    before computing step s, so communication can overlap computation.

    A and B are generated in place from their global indices (A = i + k,
    B = k - j), so no scatter is needed and --verify checks every local C
    entry against its closed form.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <getopt.h>
#include <mpi.h>

#include "gemm_local.h"

#ifndef NUM_THREADS
#define NUM_THREADS 12 /* per node; split across the ranks sharing a node */
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* start of part i when splitting n items into p nearly equal parts */
static int part_start(int i, int p, int n) { return (int)((long)i * n / p); }

typedef struct {
    int k;      /* global start of the panel */
    int w;      /* panel width */
    int a_col;  /* process column owning A[:, k:k+w] */
    int b_row;  /* process row owning B[k:k+w, :] */
} step_t;

/* Panel boundaries must respect both A's column split (over pc) and
   B's row split (over pr), so a panel never spans two owners. */
static step_t *make_schedule(int n, int b, int pr, int pc, int *nsteps)
{
    step_t *steps = malloc(((size_t)n / b + pr + pc + 1) * sizeof(*steps));
    int s = 0, ca = 0, rb = 0;
    for (int k = 0; k < n; ) {
        while (part_start(ca + 1, pc, n) <= k) ++ca;
        while (part_start(rb + 1, pr, n) <= k) ++rb;
        int end = MIN(k + b, MIN(part_start(ca + 1, pc, n), part_start(rb + 1, pr, n)));
        steps[s++] = (step_t){ .k = k, .w = end - k, .a_col = ca, .b_row = rb };
        k = end;
    }
    *nsteps = s;
    return steps;
}

/* this rank's share of the problem */
typedef struct {
    int my_r, my_c;
    int mloc, nloc;         /* local C is mloc × nloc */
    int ka0, kaloc;         /* local A holds global columns ka0 ... ka0+kaloc-1 */
    int kb0;                /* local B holds global rows kb0 ... */
    double *A, *B;
    double *bufA[2], *bufB[2];
    MPI_Comm row_comm, col_comm;
} summa_t;

/* A panel of step st: owners use their local block, others the receive buffer */
static const double *panel_A(const summa_t *z, const step_t *st, int slot, int *lda)
{
    if (z->my_c == st->a_col) {
        *lda = z->kaloc;
        return z->A + (st->k - z->ka0);
    }
    *lda = st->w;
    return z->bufA[slot];
}

static const double *panel_B(const summa_t *z, const step_t *st, int slot)
{
    return z->my_r == st->b_row ? z->B + (size_t)(st->k - z->kb0) * z->nloc : z->bufB[slot];
}

/* Post the two panel broadcasts of step st into buffer slot `slot`.  Owners
   send straight from their local blocks (A through a strided type). */
static void post_step(const summa_t *z, const step_t *st, int slot, MPI_Request req[2])
{
    if (z->my_c == st->a_col) {
        MPI_Datatype panel;
        MPI_Type_vector(z->mloc, st->w, z->kaloc, MPI_DOUBLE, &panel);
        MPI_Type_commit(&panel);
        MPI_Ibcast(z->A + (st->k - z->ka0), 1, panel, st->a_col, z->row_comm, &req[0]);
        MPI_Type_free(&panel);
    } else {
        MPI_Ibcast(z->bufA[slot], z->mloc * st->w, MPI_DOUBLE, st->a_col, z->row_comm, &req[0]);
    }
    MPI_Ibcast((void *)panel_B(z, st, slot), st->w * z->nloc, MPI_DOUBLE,
               st->b_row, z->col_comm, &req[1]);
}

static void *x_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size ? size : alignment) != 0) {
        perror("posix_memalign");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

typedef struct {
    int n, b, nthreads, reps, warmup, verify, csv, overlap;
    int grid_r, grid_c;     /* 0 = let MPI_Dims_create choose */
    double peak_node;       /* GFLOP/s per node for %-of-peak; 0 = unknown */
} options_t;

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [options]\n"
        "  -n, --n N            matrix order (default 4096)\n"
        "  -b, --b B            SUMMA panel width (default 512)\n"
        "  -t, --threads T      threads per rank (default %d / ranks per node)\n"
        "  -g, --grid PRxPC     process grid (default: MPI_Dims_create)\n"
        "  -r, --reps R         timed repetitions (default 5)\n"
        "  -w, --warmup W       untimed warm-up runs (default 1)\n"
        "  -p, --peak GFLOPS    per-node peak, for %% of peak in the CSV\n"
        "      --no-overlap     blocking broadcasts (no comm/compute overlap)\n"
        "      --verify         check every entry of C against its closed form\n"
        "      --csv            also print impl,nodes,ranks,threads,N,b,time_s,gflops,pct_peak\n",
        prog, NUM_THREADS);
}

static int parse_options(int argc, char **argv, options_t *o)
{
    *o = (options_t){ .n = 4096, .b = 512, .reps = 5, .warmup = 1, .overlap = 1 };
    static const struct option longopts[] = {
        { "n",          required_argument, NULL, 'n' },
        { "b",          required_argument, NULL, 'b' },
        { "threads",    required_argument, NULL, 't' },
        { "grid",       required_argument, NULL, 'g' },
        { "reps",       required_argument, NULL, 'r' },
        { "warmup",     required_argument, NULL, 'w' },
        { "peak",       required_argument, NULL, 'p' },
        { "no-overlap", no_argument,       NULL, 'O' },
        { "verify",     no_argument,       NULL, 'V' },
        { "csv",        no_argument,       NULL, 'C' },
        { "help",       no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };
    int ch;
    while ((ch = getopt_long(argc, argv, "n:b:t:g:r:w:p:h", longopts, NULL)) != -1) {
        switch (ch) {
        case 'n': o->n = atoi(optarg); break;
        case 'b': o->b = atoi(optarg); break;
        case 't': o->nthreads = atoi(optarg); break;
        case 'g':
            if (sscanf(optarg, "%dx%d", &o->grid_r, &o->grid_c) != 2) return 0;
            break;
        case 'r': o->reps = atoi(optarg); break;
        case 'w': o->warmup = atoi(optarg); break;
        case 'p': o->peak_node = atof(optarg); break;
        case 'O': o->overlap = 0; break;
        case 'V': o->verify = 1; break;
        case 'C': o->csv = 1; break;
        default: return 0;
        }
    }
    return o->n > 0 && o->b > 0 && o->reps > 0 && o->warmup >= 0 && optind == argc;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int nranks, wrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    options_t opt;
    if (!parse_options(argc, argv, &opt)) {
        if (wrank == 0) usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ranks per node and node count, for default threads and reporting */
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int ranks_per_node, node_rank, nodes;
    MPI_Comm_size(node_comm, &ranks_per_node);
    MPI_Comm_rank(node_comm, &node_rank);
    int is_leader = node_rank == 0;
    MPI_Allreduce(&is_leader, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_free(&node_comm);
    if (opt.nthreads <= 0) opt.nthreads = MAX(1, NUM_THREADS / ranks_per_node);

    /* process grid */
    int dims[2] = { opt.grid_r, opt.grid_c };
    if (dims[0] * dims[1] != 0 && dims[0] * dims[1] != nranks) {
        if (wrank == 0)
            fprintf(stderr, "grid %dx%d does not match %d ranks\n", dims[0], dims[1], nranks);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    MPI_Dims_create(nranks, 2, dims);
    int pr = dims[0], pc = dims[1];
    if (opt.n < MAX(pr, pc)) {
        if (wrank == 0) fprintf(stderr, "N=%d is smaller than the %dx%d grid\n", opt.n, pr, pc);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm grid, row_comm, col_comm;
    int periods[2] = { 0, 0 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int grank, coords[2];
    MPI_Comm_rank(grid, &grank);
    MPI_Cart_coords(grid, grank, 2, coords);
    int my_r = coords[0], my_c = coords[1];
    MPI_Cart_sub(grid, (int[]){ 0, 1 }, &row_comm);  /* rank in row_comm == my_c */
    MPI_Cart_sub(grid, (int[]){ 1, 0 }, &col_comm);  /* rank in col_comm == my_r */

    /* local blocks: rows of A and C split over pr, columns of B and C over pc,
       the shared k dimension over pc for A and over pr for B */
    const int n = opt.n;
    int i0  = part_start(my_r, pr, n), mloc  = part_start(my_r + 1, pr, n) - i0;
    int j0  = part_start(my_c, pc, n), nloc  = part_start(my_c + 1, pc, n) - j0;
    int ka0 = part_start(my_c, pc, n), kaloc = part_start(my_c + 1, pc, n) - ka0;
    int kb0 = part_start(my_r, pr, n), kbloc = part_start(my_r + 1, pr, n) - kb0;
    int b = MIN(opt.b, n);

    if ((long)mloc * b > INT_MAX || (long)b * nloc > INT_MAX) {
        if (wrank == 0) fprintf(stderr, "panel too large for an MPI count; reduce -b\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double *A = x_aligned_alloc(64, (size_t)mloc * kaloc * sizeof(double));
    double *B = x_aligned_alloc(64, (size_t)kbloc * nloc * sizeof(double));
    double *C = x_aligned_alloc(64, (size_t)mloc * nloc * sizeof(double));
    double *bufA[2], *bufB[2];
    for (int i = 0; i < 2; ++i) {
        bufA[i] = x_aligned_alloc(64, (size_t)mloc * b * sizeof(double));
        bufB[i] = x_aligned_alloc(64, (size_t)b * nloc * sizeof(double));
    }

    for (int i = 0; i < mloc; ++i)
        for (int k = 0; k < kaloc; ++k)
            A[(size_t)i * kaloc + k] = (double)((i0 + i) + (ka0 + k));
    for (int k = 0; k < kbloc; ++k)
        for (int j = 0; j < nloc; ++j)
            B[(size_t)k * nloc + j] = (double)((kb0 + k) - (j0 + j));

    int nsteps;
    step_t *steps = make_schedule(n, b, pr, pc, &nsteps);

    summa_t z = {
        .my_r = my_r, .my_c = my_c, .mloc = mloc, .nloc = nloc,
        .ka0 = ka0, .kaloc = kaloc, .kb0 = kb0, .A = A, .B = B,
        .bufA = { bufA[0], bufA[1] }, .bufB = { bufB[0], bufB[1] },
        .row_comm = row_comm, .col_comm = col_comm,
    };
    MPI_Request req[2];

    double *times = malloc((size_t)opt.reps * sizeof(double));
    double best_wait = 0.0, best_comp = 0.0, best_time = DBL_MAX;

    for (int rep = -opt.warmup; rep < opt.reps; ++rep) {
        memset(C, 0, (size_t)mloc * nloc * sizeof(double));
        double t_wait = 0.0, t_comp = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        if (opt.overlap) post_step(&z, &steps[0], 0, req);
        for (int s = 0; s < nsteps; ++s) {
            const step_t *st = &steps[s];
            double tw = MPI_Wtime();
            if (!opt.overlap) post_step(&z, st, s % 2, req);
            MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
            if (opt.overlap && s + 1 < nsteps) post_step(&z, &steps[s + 1], (s + 1) % 2, req);
            double tc = MPI_Wtime();
            t_wait += tc - tw;

            int lda;
            const double *Ap = panel_A(&z, st, s % 2, &lda);
            const double *Bp = panel_B(&z, st, s % 2);
            dgemm_local(mloc, nloc, st->w, Ap, lda, Bp, nloc, C, nloc, opt.nthreads);
            t_comp += MPI_Wtime() - tc;
        }

        double t_local = MPI_Wtime() - t0, t_max, sum_wait, sum_comp;
        MPI_Allreduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Reduce(&t_wait, &sum_wait, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_comp, &sum_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rep < 0) continue;
        times[rep] = t_max;
        if (t_max < best_time) {
            best_time = t_max;
            best_wait = sum_wait / nranks;
            best_comp = sum_comp / nranks;
        }
    }

    int failed = 0;
    if (wrank == 0) {
        double flops = 2.0 * n * n * n;
        double gflops = flops / best_time * 1e-9;
        qsort(times, opt.reps, sizeof(double), cmp_double);
        printf("SUMMA N=%d b=%d grid=%dx%d ranks=%d nodes=%d threads/rank=%d %s\n",
               n, b, pr, pc, nranks, nodes, opt.nthreads,
               opt.overlap ? "overlap" : "no-overlap");
        printf("Time: best %.3f s (%.1f GFLOP/s), median %.3f s; "
               "per rank avg: compute %.3f s, comm wait %.3f s\n",
               best_time, gflops, times[opt.reps / 2], best_comp, best_wait);
        if (opt.csv) {
            printf("impl,nodes,ranks,threads,N,b,time_s,gflops,pct_peak\n");
            if (opt.peak_node > 0)
                printf("summa,%d,%d,%d,%d,%d,%.6f,%.2f,%.2f\n", nodes, nranks, opt.nthreads,
                       n, b, best_time, gflops, 100.0 * gflops / (opt.peak_node * nodes));
            else
                printf("summa,%d,%d,%d,%d,%d,%.6f,%.2f,\n", nodes, nranks, opt.nthreads,
                       n, b, best_time, gflops);
        }
    }

    if (opt.verify) {
        /* exact check of every entry: sum_k (i+k)(k-j) = (i-j)*S1 - n*i*j + S2,
           evaluated in integers so -ffast-math cannot reassociate it */
        long long S1 = (long long)n * (n - 1) / 2;
        long long S2 = (long long)(n - 1) * n * (2LL * n - 1) / 6;
        double maxerr = 0.0, global_err;
        for (int i = 0; i < mloc; ++i) {
            for (int j = 0; j < nloc; ++j) {
                long long gi = i0 + i, gj = j0 + j;
                double ref = (double)((gi - gj) * S1 - (long long)n * gi * gj + S2);
                double err = fabs(ref - C[(size_t)i * nloc + j]) / (fabs(ref) + 1.0);
                if (err > maxerr) maxerr = err;
            }
        }
        MPI_Allreduce(&maxerr, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        failed = global_err >= 1e-9;
        if (wrank == 0)
            printf("Verify: max rel err %.3e (%s)\n", global_err, failed ? "FAIL" : "OK");
    }

    free(times);
    free(steps);
    for (int i = 0; i < 2; ++i) {
        free(bufA[i]);
        free(bufB[i]);
    }
    free(A);
    free(B);
    free(C);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
