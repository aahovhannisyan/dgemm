import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")  # spurious FP-flag warnings from Accelerate

N = 4096
i = np.arange(N, dtype=np.float64)
A = i[:, None] + i[None, :]
B = i[:, None] - i[None, :]

A @ B  # warm-up
times = []
for _ in range(5):
    t0 = time.perf_counter()
    C = A @ B
    times.append(time.perf_counter() - t0)

best = min(times)
print(f"numpy {np.__version__}: best {best:.3f} s ({2 * N**3 / best * 1e-9:.1f} GFLOP/s), "
      f"median {sorted(times)[len(times)//2]:.3f} s")
