#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void reduction_tree(int* input, int* output, int N) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    shared_data[tid] = (i < N) ? input[i] : 0;

    if (i + blockDim.x < N) {
        shared_data[tid] += input[i + blockDim.x];
    }

    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result from block to global output atomically
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size_of_array>" << std::endl;
        return EXIT_FAILURE;
    }

    int N = std::atoi(argv[1]);
    std::vector<int> h_input(N, 1); // Fill with 1s

    int* d_input = nullptr;
    int* d_output = nullptr;
    int h_result = 0;

    checkCudaError(cudaMalloc((void**)&d_input, N * sizeof(int)), "Allocating device input array");
    checkCudaError(cudaMalloc((void**)&d_output, sizeof(int)), "Allocating device output variable");
    checkCudaError(cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice), "Copying input data to device");
    checkCudaError(cudaMemset(d_output, 0, sizeof(int)), "Initializing output variable on device");

    int blockSize = 256;
    int numBlocks = (N + blockSize * 2 - 1) / (blockSize * 2);

    reduction_tree<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing device after kernel execution");

    checkCudaError(cudaMemcpy(&h_result, d_output, sizeof(int), cudaMemcpyDeviceToHost), "Copying result back to host");

    std::cout << "Reduced Sum (tree reduction): " << h_result << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
