# Optimized DGEMM: Vectorized Micro-Kernels and Memory-Aware Parallelization

High-performance Double-precision General Matrix Multiplication (DGEMM) in C for N×N doubles, using:

- AVX2 + FMA vectorization (x86-64) or NEON `vfmaq_f64` (Apple Silicon / AArch64)
- A custom 6×8 register-blocked micro-kernel
- Thread-level parallelism with static partitioning using POSIX threads
- Block sizes tuned for cache reuse

## Build
The program is compiled using `gcc` compiler on Linux. Make sure to have `gcc` and `make` installed and then run:
```
make
```

On macOS (Apple Silicon) `make` uses the system clang and the NEON kernel automatically.
Useful overrides: `make THREADS=10 KC=256 NC=64`, and `make VERIFY=1` to check sampled results
against a naive dot product.

## Compare with numpy
```
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python bench_numpy.py
```
Note: on macOS numpy 2.x uses Accelerate, not OpenBLAS, so results are not comparable to the Linux/OpenBLAS numbers.

## Run
```
./bin/dgemm
```
 Output example:
 ```
 Time: 0.850 s
 ```

The program initializes A and B with simple deterministic patterns, computes C = A × B, and prints wall-clock time.
