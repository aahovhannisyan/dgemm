# Optimized DGEMM: Vectorized Micro-Kernels and Memory-Aware Parallelization

High-performance Double-precision General Matrix Multiplication (DGEMM) in C for N×N doubles, using:

- AVX2 + FMA vectorization
- A custom 6×8 register-blocked micro-kernel
- Thread-level parallelism with static partitioning using POSIX threads
- Block sizes tuned for cache reuse

## Build
The program is compiled using `gcc` compiler on Linux. Make sure to have `gcc` and `make` installed and then run:
```
make
```

## Run
```
./bin/dgemm
```
 Output example:
 ```
 Time: 0.850 s
 ```

The program initializes A and B with simple deterministic patterns, computes C = A × B, and prints wall-clock time.
