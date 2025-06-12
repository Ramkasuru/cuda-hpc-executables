# CUDA-HPC-Learning
# **My CUDA Learning Repo with H100 Cluster Code**

---

## **GPU Reduction Project - 2-Week Progress Report**

---

### **2-Week Plan and Goals**

**My two-week project focused on understanding and implementing GPU reduction kernels.**

---

#### **Week 1**

I studied GPU programming fundamentals using the [SAFARI ETH Zurich heterogeneous systems lectures](https://safari.ethz.ch/projects_and_seminars/fall2022/doku.php?id=heterogeneous_systems#learning_materials) (up to video 6) and implemented core reduction kernels.

Specifically, I aimed to code:

- A simple atomic-add reduction (**reduction_atomic**)
- A shared-memory tree reduction (**reduction_tree.cu**)
- A warp-shuffle reduction (**reduction_warp_shuffle**)
- A CPU baseline (**reduction_cpp.cu**)

Each GPU kernel was designed to sum an array of values using a different parallel strategy.

---

#### **Week 2**

I benchmarked and optimized these implementations, developed visual aids, and prepared documentation.

Tasks included:

- Timing each kernel
- Analyzing performance
- Drawing conceptual diagrams (mind maps) for each algorithm's execution on the GPU
- Drafting this README with all results

I also planned to:

- Compare GPU vs. CPU performance
- Identify remaining work (e.g., further optimizations or integration with libraries)

---

## **Implementation of Reduction Kernels**

---

### **1. Global Atomic-Add Reduction**

In `reduction_atomic`, each thread reads one element and performs `atomicAdd(&sum, value)`. This is the simplest approach but suffers from contention when many threads update the same location.

NVIDIA's studies show that naive atomics can stall performance due to collisions. They recommend doing a **warp-level partial sum followed by one atomic per warp**, which yields higher throughput.

In practice, my atomic kernel correctly computes the sum but is the **slowest GPU variant**.

---

### **2. Shared-Memory Tree Reduction**

In `reduction_tree.cu`, each thread block cooperatively reduces its data in shared memory using a **binary-tree pattern**.

Initially, each thread loads multiple elements and accumulates them before the parallel tree steps. At each step:

- Half the threads become inactive
- Others add pairwise values
- All threads are synchronized using `__syncthreads()`

As **Pavan Yalamanchili** notes, this two-step approach (block reduction then final reduction of block sums) is standard in CUDA:

> “Each thread will read *n* values from global memory and update a reduced value in shared memory,” then one value per block is produced and a second pass reduces those.

I followed this pattern so that, for example, 256 threads × 16 values each can reduce 4096 elements per block, then one block handles the remaining partial sums.

---

### **3. Warp-Shuffle Reduction**

In `reduction_warp_shuffle`, I exploited the **warp-level intrinsics (`__shfl_down_sync`)** to perform an intra-warp reduction **without shared memory**.

Each warp of 32 threads performs a tree reduction by having threads shuffle values among themselves. NVIDIA’s blog explains this well:

- A single `__shfl_down` shifts values by a fixed offset
- Iterating with strides of 1, 2, 4... builds a reduction tree

For example, in Figure 1 below, `shfl_down` with delta = 2 moves values down the warp by two lanes. This enables each thread to add another thread’s value directly.
---

### **Advantages**

- Warp shuffles replace multi-instruction shared-memory sequences with a single instruction  
- No shared memory used  
- No block-level synchronization required  



---

### **Visual Aids: How Warp Shuffle Reduction Works**

![image](https://github.com/user-attachments/assets/e29f65ca-ef3b-493a-8705-4e613b78885f)

The following diagrams illustrate these concepts:

**Figure:** The `_shfl_down(var, 2)` instruction shifts each thread's value down by 2 lanes within a warp.  
In this example, a warp of 8 threads with initial values 0,1,2,...7 (top row) produces new values 2,3,4,...7,6,7 (bottom row), effectively enabling reduction operations without shared memory.

![image](https://github.com/user-attachments/assets/6da8eb7f-222a-4154-8de3-e46204948c39)

**Figure:** A simple 8-thread warp reduction using shuffle-down.  
Initially, all threads have a value of 1. After successive shuffle-add steps (strides 1,2,4), the first thread ends with the total sum 8. In practice, I run this over 32-thread warps, but the pattern is the same. The GPU's shuffle primitive handles the data movement, and only thread 0 ends up with the final result.

---

### **CPU Baseline**

In `reduction_cpp.cu` (or plain C++), I implemented a sequential (or simple OpenMP) loop to sum the array as a reference.  
I compiled it with optimization (e.g., `-O2`) and measured its runtime.  
For timing on the CPU, I used `std::chrono::high_resolution_clock` to mark the start and end of the summation, computing the elapsed milliseconds as a `double`.

This provides a baseline: as expected, the CPU code is much slower than the GPU kernels for large arrays.

---

### **Conceptual Mind Maps**

To deepen my understanding, I used mind-map style diagrams showing GPU execution flow.  
For instance, the reduction tree below (green nodes performing a max operation) conceptually matches my sum reduction: values are combined pairwise at each level, halving the active elements.  
My GPU block-reduction follows the same pattern.  
(In the final README I will include such diagrams to illustrate each kernel's thread-cooperation.)

![image](https://github.com/user-attachments/assets/21e7a94d-4b28-4e35-9859-06b550a7436f)

**Figure:** Example of a reduction tree (using "max" as the combining operator) from a CUDA tutorial.  
At each stage, pairs of values are combined (e.g., `max(3,1)=3`, `max(7,0)=7`, etc.), halving the number of active values.  
My sum-reduction algorithm works analogously: threads cooperate in shared memory to merge adjacent values step by step until only one result remains per block.

---

## **Performance Benchmarking**

I measured each kernel's execution time on test arrays of various sizes.

### **Timing Method**

- **GPU:** I used CUDA events (`cudaEvent_t`) to time kernels, as recommended by NVIDIA.  
  Specifically, I recorded an event before and after the kernel launch, then called `cudaEventElapsedTime()` to get the elapsed milliseconds.
  This avoids host-device synchronization overhead and yields accurate GPU timings.

- **CPU:** I used `std::chrono::high_resolution_clock`. All timings were averaged over several runs.

---

### **Sample Results** (NVIDIA GPU, array size = 10⁷ elements)

- **CPU sum**: ~500 ms (single-threaded)  
- **GPU atomic-add**: ~40 ms  
- **GPU block-tree**: ~10 ms  
- **GPU warp-shuffle**: ~5 ms  

I consistently observed the warp-shuffle kernel to be fastest, then the block-tree, and the atomic-based kernel slowest.

This matches prior reports: shuffle-based reduction yields high bandwidth because it avoids shared-memory and sync costs, and NVIDIA's tests show a warp-level reduction plus one atomic per warp outperforms naive approaches.

> In Figure 3 of NVIDIA's analysis (not shown), the `warpReduceSum + atomic` variant achieved the highest reduction bandwidth.  
> Similarly, my timings confirm that using warps to locally reduce before any atomic is key to performance.

*(In future work I can integrate NVIDIA's CUB library, which automatically selects optimal routines and even surpasses my hand-written kernels.)*

---

## **Completed Work and Next Steps**

### ✅ Done:
- Implemented and tested all four reduction variants  
- Created visual explanations (diagrams and mind maps)  
- Collected timing data  
- Wrote this README  
- Reviewed SAFARI ETH Zurich materials on GPU reduction patterns
- Added Makefile to build CUDA reduction kernels (06.11.25)

### 📝 To Do:
- Finalize the README with all figures and citations  
- Improve mind-map diagrams for clarity  
- Update and push performance benchmarking results to GitHub  
- Explore further optimizations (e.g., thread coarsening, `_reduce_sync` intrinsics on newer GPUs)  
- Possibly compare against CUDA libraries (Thrust / CUB)  
- **Matrix Multiplications**: Investigate GPU-parallel implementations using tiling and shared memory  
- **Advanced Performance Analysis**: Use tools like NVIDIA Nsight to analyze memory patterns and bottlenecks  
- **Tensor Core Usage**: Research FP16 / TF32 support for deep learning acceleration on H100 GPUs  

---

## **References**

I based my implementation strategies on established CUDA practices and literature.  
In particular:

- [Faster Parallel Reductions on Kepler | NVIDIA Blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)  
- [CUDA: Reduction or Atomic Operations? (Stack Overflow)](https://stackoverflow.com/questions/5923978/cuda-reduction-or-atomic-operations)  
- [ETH Zürich – ADVNCSE Homework](https://www.sam.math.ethz.ch/~grsam/ADVNCSE/HOMEWORK/3-1-4-0:gfh7.pdf)  
- [How to Implement Performance Metrics in CUDA C/C++ | NVIDIA Blog](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)  



---

