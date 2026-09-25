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
