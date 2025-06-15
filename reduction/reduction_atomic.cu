#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>  // for atoi

__global__ void reduce_atomic(const float* input, float* result, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        atomicAdd(result, input[i]);
    }
}

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <array_size> <block_size>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int blockSize = std::atoi(argv[2]);

    if (N <= 0 || blockSize <= 0) {
        std::cerr << "Error: array_size and block_size must be positive integers." << std::endl;
        return 1;
    }

    size_t size = N * sizeof(float);
    float* h_input = new float[N];

    // Initialize input array with 1.0f for testing
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float* d_input;
    float* d_result;

    check_cuda_error(cudaMalloc(&d_input, size));
    check_cuda_error(cudaMalloc(&d_result, sizeof(float)));

    // Initialize result to zero on device
    float h_result = 0.0f;
    check_cuda_error(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));

    // Calculate number of blocks needed
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    reduce_atomic<<<numBlocks, blockSize>>>(d_input, d_result, N);
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaDeviceSynchronize());

    // Copy result back to host
    check_cuda_error(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Result from atomic reduction: " << h_result << std::endl;

    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
