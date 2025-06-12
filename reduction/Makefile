# Makefile for CUDA Reduction Kernels

NVCC = nvcc
CXX = g++
CUDA_FLAGS = -O3 -arch=sm_90
CPP_FLAGS = -O3

TARGETS = reduction_atomic reduction_tree reduction_warp_shuffle reduction_cpu

all: $(TARGETS)

reduction_atomic: reduction/reduction_atomic.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

reduction_tree: reduction/reduction_tree.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

reduction_warp_shuffle: reduction/reduction_warp_shuffle.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

reduction_cpu: reduction/reduction_cpu.cpp
	$(CXX) $(CPP_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
