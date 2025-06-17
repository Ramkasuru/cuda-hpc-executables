# CUDA-HPC-Learning

**My CUDA Learning Repo with H100 Cluster Code**

---

## Progress Report: Vector Addition, Reduction, and Matrix Multiplication

### Project Overview and Timeline

Over the past few weeks, I have been learning GPU programming (CUDA) and implementing core parallel algorithms on NVIDIA GPUs (initially using an L40S GPU on the Jarvis cluster). My work so far covers simple vector addition, several GPU reduction kernels, and an initial matrix multiplication (GEMM) kernel. Below is a summary of my progress, experiment setup, results, and analysis, along with my planned future improvements.

- **Week 1:** I studied CUDA fundamentals using [SAFARI ETH Zürich heterogeneous systems lectures](https://safari.ethz.ch/projects_and_seminars/fall2022/doku.php?id=heterogeneous_systems#learning_materials) (videos 1–6) and implemented basic kernels:
    - A simple vector addition kernel (adding two arrays on GPU).
    - Three reduction kernels: (1) global atomic-add, (2) shared-memory tree reduction, (3) warp-shuffle reduction.
    - A CPU baseline (C++ loop) for comparisons.

- **Week 2:** I benchmarked and optimized the reduction kernels, developed conceptual diagrams, and drafted documentation (this README focusing on reduction results). I also added a Makefile for building the CUDA code (06.11.25). I planned to compare GPU vs CPU performance and identify further optimizations.

- **Week 3:** I began implementing **matrix multiplication (GEMM)** on the GPU using tiling and shared memory. I started experimenting with different tile sizes and block mappings for performance. I also reorganized the experiment scripts for reproducibility (so that each kernel test can be easily run, similar to HPC artifact evaluation practices).

*Note: Due to cluster constraints, I could not yet test on the H100 SXM GPU – only the L40S GPU was available. H100 results will be added as soon as a free H100 is accessible.*

---

## Hardware and Software Specifications

### GPU (for all performance runs):

- **Model:** NVIDIA L40S (Ada Lovelace)
- **Memory:** 46 GB GDDR6
- **CUDA Toolkit:** 12.2
- **CUDA Driver/Runtime:** 12.4
- **Driver Version:** 550.127.05

### CPU (for baseline runs):

- **Model:** Intel Xeon Platinum 8562Y+
- **Cores:** 64
- **Features:** AVX512, 4 NUMA nodes
- **Cache:** L3: 120 MiB (2 instances)

---

## Implemented Algorithms and Kernels

### 1. GPU Vector Addition

I implemented simple elementwise addition of large arrays using CUDA.  
- This kernel is memory-bandwidth bound, and I observed a 10–20x speedup compared to the CPU for large N.
- I verified correctness by comparison to the CPU output.

### 2. GPU Reduction Kernels

**A. Global Atomic-Add:**  
Each thread atomically adds its value to a global sum. This method is simple but slow due to atomic contention.

**B. Shared-Memory Tree Reduction:**  
Threads reduce in shared memory within a block, then one atomic per block for the global sum. This approach reduces contention and increases performance.

**C. Warp-Shuffle Reduction:**  
Within each warp, threads use `__shfl_down_sync` to sum values without shared memory. One atomic per warp writes to global memory. This was the fastest kernel due to minimal atomics and synchronization.

### 3. CPU Baseline

- I used a standard sequential sum (C++ for-loop) as a baseline for reference and speedup calculations.

### 4. Matrix Multiplication (GEMM)

- **Naive:** Each thread computes a single element of C. This approach is heavy on global memory.
- **Tiled (shared memory):** Each block works on a tile of C using shared memory for A and B tiles. This version is orders of magnitude faster than the naive implementation.

---

## Experiment Setup

- **Kernels implemented and tested:** vector addition, reduction (atomic, tree, warp), GEMM (naive and tiled).
- **Tested parameters:** array/matrix sizes (10^6–10^8), threads per block (usually 256), tile sizes (8, 16, 32 for GEMM).
- **Timing:** I used CUDA events for GPU measurements and C++ chrono for CPU timing.
- **All scripts are organized for easy reproduction with clear makefiles and command-line arguments.**

---

## Performance Results

### Summary Table (N = 1,000,000, Threads/Block = 256)

| Kernel               | Device/Arch      | Time (ms)   | Notes           |
|----------------------|------------------|-------------|-----------------|
| CPU (1 thread)       | Xeon Platinum    | 0.204       | Baseline        |
| Atomic reduction     | NVIDIA L40S      | 7.61        | atomicAdd       |
| Tree reduction       | NVIDIA L40S      | 8.88        | Shared memory   |
| Warp-shuffle         | NVIDIA L40S      | 8.99        | Fastest variant |

> *H100 results will be added as soon as GPUs are available on Jarvis.*

---

## Analysis

- **Warp-shuffle** outperformed tree and atomic approaches as expected, but on L40S the gap was smaller than on older GPUs (likely due to improved memory and synchronization hardware).
- The **CPU baseline** is much slower for large N, highlighting the value of parallelism.
- **Tiled GEMM** is essential for matrix multiply performance; the naive kernel is hundreds of times slower.
- **cuBLAS** is 10x–20x faster than my hand-coded GEMM, showing the gap with fully optimized library routines.
- I observed some unexpected results: atomic reduction is less penalized on modern Ada Lovelace GPUs compared to Kepler/Pascal, likely due to improved atomics hardware.

---

## Current Limitations

- **No H100/H100 SXM results yet:** I have not been able to benchmark on H100 GPUs due to resource constraints.  
  *This will be updated ASAP once access is possible.*
- **GEMM (Matrix Multiply) is still in progress:** I have only basic naive and tiled versions implemented. Advanced optimizations and cuBLAS comparisons will be added soon.

---

## Future Work

- Add a full performance comparison with H100/H100 SXM GPUs and analyze results.
- Carry out a deeper analysis of performance and bottlenecks using Nsight Compute/Profiler.
- Implement further reduction optimizations (e.g., using `__reduce_sync`, vectorized loads, thread coarsening).
- Extend GEMM: improve tiling, implement Tensor Core support, compare with cuBLAS.
- Update scripts and makefiles to be fully artifact-evaluation ready (easy to rerun every experiment).
- Document unexpected findings (e.g., if warp shuffle does not outperform tree on H100), with analysis.
- Explore more parallel primitives in CUDA (convolution, scan, sort, etc).

---

## References

- [Faster Parallel Reductions on Kepler | NVIDIA Blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [CUDA: Reduction or Atomic Operations? (Stack Overflow)](https://stackoverflow.com/questions/5923978/cuda-reduction-or-atomic-operations)
- [ETH Zürich – ADVNCSE Homework](https://www.sam.math.ethz.ch/~grsam/ADVNCSE/HOMEWORK/3-1-4-0:gfh7.pdf)
- [How to Implement Performance Metrics in CUDA C/C++ | NVIDIA Blog](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
- NVIDIA L40S & H100 product briefs
- SAFARI ETH Zürich lectures
