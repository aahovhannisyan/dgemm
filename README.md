# Optimized DGEMM: Vectorized Micro-Kernels and Memory-Aware Parallelization

High-performance Double-precision General Matrix Multiplication (DGEMM) in C for N×N doubles, using:

- AVX2 + FMA vectorization (x86-64) or NEON `vfmaq_f64` (Apple Silicon / AArch64)
- A custom 6×8 register-blocked micro-kernel
- Thread-level parallelism with static 2D partitioning using POSIX threads
- Block sizes tuned for cache reuse

The kernel lives in `src/gemm_local.{c,h}` as a reusable routine,
`dgemm_local(m, n, k, A, lda, B, ldb, C, ldc, nthreads)`, which computes `C += A·B` for any
row-major shape (including submatrix views with `ld > width`). `dgemm.c` is a thin benchmark driver.

## Build
The program is compiled using `gcc` compiler on Linux. Make sure to have `gcc` and `make` installed and then run:
```
make
```

On macOS (Apple Silicon) `make` uses the system clang and the NEON kernel automatically.
Useful overrides: `make THREADS=10 KC=256 NC=64`, and `make VERIFY=1` to check every entry of C
against its closed form. `make test` runs `dgemm_local` against a naive reference on odd shapes,
submatrix views and a range of thread counts.

## Distributed (MPI SUMMA)
`src/summa.c` multiplies N×N matrices across MPI ranks with SUMMA on a 2D process grid. Each rank
owns a block of A, B and C; for each k-panel the owning process column broadcasts its slice of A
along the process row, the owning process row broadcasts its slice of B along the process column,
and every rank updates its C block with `dgemm_local`. The next panel's broadcasts
(`MPI_Ibcast`) are posted before the current panel is computed, so communication overlaps compute.
Matrices are generated in place from global indices, so no scatter is needed.

```
brew install open-mpi            # or the cluster's MPI module
make summa
mpirun -np 4 ./bin/summa -n 8192 -b 512 -t 4 --verify
```
Options: `-n` matrix order, `-b` panel width, `-t` threads per rank (default: `THREADS` split across
the ranks on a node), `-g PRxPC` process grid, `-r`/`-w` timed/warm-up repetitions, `--no-overlap`
for blocking broadcasts, `--verify` to check every entry of C, and `--csv` (with `-p` per-node peak
GFLOP/s) for a machine-readable line. `make test-summa` runs a correctness sweep over grid shapes,
uneven block sizes and panel widths.

To run on a Graviton (hpc7g) Slurm cluster on AWS, with ScaLAPACK, COSMA, SLATE and a Python
stack installed for comparison, see [infra/README.md](infra/README.md).

## Compare with numpy
```
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python bench_numpy.py
```
Note: on macOS numpy 2.x uses Accelerate, not OpenBLAS, so results are not comparable to the Linux/OpenBLAS numbers.

## Run
```
./bin/dgemm [N] [threads]
```
N defaults to 4096 and threads to the `THREADS` build setting. Output example (Apple M5 Pro):
```
Time: 0.225 s (610.8 GFLOP/s, 16 threads)
```

The program initializes A and B with simple deterministic patterns, computes C = A × B, and prints wall-clock time.
