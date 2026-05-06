import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Data (Block Size 1024)
# --------------------------------------------------
N = np.array([
    16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152, 4194304, 8388608
])

base       = np.array([2.00, 1.79, 1.57, 1.47, 1.09, 1.65, 3.37, 4.56, 5.51, 5.90])
pre        = np.array([2.60, 2.45, 2.07, 2.55, 1.67, 2.46, 4.43, 6.10, 23.57, 22.87])
normal     = np.array([3.44, 3.89, 2.57, 3.22, 3.46, 5.11, 7.48, 10.09, 13.06, 9.92])

# --------------------------------------------------
# Plot 1: Performance vs N
# --------------------------------------------------
plt.figure()
plt.plot(N, base, label="Base")
plt.plot(N, pre, label="Pre")
plt.plot(N, normal, label="Normal")

plt.xscale("log")
plt.xlabel("N (log scale)")
plt.ylabel("Time (ms)")
plt.title("FFT Performance (Block Size 1024)")
plt.legend()
plt.grid()

import numpy as np
import matplotlib.pyplot as plt

# Block sizes
block_sizes = np.array([16, 32, 64, 128, 256, 512, 1024])

# Base values (ONLY last 3 rows per table)

base_16   = np.array([5.38, 5.65, 7.99])
base_32   = np.array([4.61, 4.24, 8.89])
base_64   = np.array([5.22, 4.97, 8.72])
base_128  = np.array([5.10, 5.29, 5.85])
base_256  = np.array([5.33, 5.60, 8.55])
base_512  = np.array([5.16, 6.94, 8.86])
base_1024 = np.array([4.56, 5.51, 5.90])

# Compute averages
base_avg = np.array([
    np.mean(base_16),
    np.mean(base_32),
    np.mean(base_64),
    np.mean(base_128),
    np.mean(base_256),
    np.mean(base_512),
    np.mean(base_1024)
])

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure()
plt.plot(block_sizes, base_avg, marker='o')

plt.xlabel("Block Size (Threads per Block)")
plt.ylabel("Average Time (ms)")
plt.title("Base Twiddle Performance (Large Sizes Only)")
plt.grid()

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
block_sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024])

base  = np.array([11.8954, 16.614, 13.6438, 10.878, 17.0505, 15.8292, 15.1934, 16.4784])
pre   = np.array([16.0453, 21.4325, 20.0116, 13.351, 23.7248, 23.4296, 22.9816, 23.9682])
norm  = np.array([43.3279, 40.3484, 34.3764, 37.6913, 56.9967, 45.9558, 33.8768, 44.1928])

# Plot
plt.figure()

plt.plot(block_sizes, base, marker='o', label="Base")
plt.plot(block_sizes, pre, marker='o', label="Pre")
plt.plot(block_sizes, norm, marker='o', label="Normal")

plt.xlabel("Block Size (Threads per Block)")
plt.ylabel("Relative Runtime (x cuFFT)")
plt.title("Performance vs Block Size (N = 4194304)")
plt.xscale("log", base=2)

plt.legend()
plt.grid()

plt.show()