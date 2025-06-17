#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>  // for std::stoi

// CUDA kernel: atomic reduction
__global__ void reduction_atomic(const int* input, int* result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(result, input[tid]);
    }
}

// Error checking function
// This function checks for CUDA errors and prints an error message if one occurs.
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_elements> <threads_per_block>\n";
        return EXIT_FAILURE;
    }

    int N = std::stoi(argv[1]);
    int threads = std::stoi(argv[2]);
    int blocks = (N + threads - 1) / threads;

    // Host memory
    std::vector<int> h_input(N, 1); // All ones for easy checking
    int h_result = 0;

    // Device memory
    int *d_input = nullptr, *d_result = nullptr;
    checkCuda(cudaMalloc(&d_input, N * sizeof(int)), "cudaMalloc d_input");
    checkCuda(cudaMalloc(&d_result, sizeof(int)), "cudaMalloc d_result");

    // Copy input to device, zero the result on device
    checkCuda(cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy input");
    checkCuda(cudaMemset(d_result, 0, sizeof(int)), "cudaMemset result");

    // kernel launch
    reduction_atomic<<<blocks, threads>>>(d_input, d_result, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "device synchronize");

    // Copy result back
    checkCuda(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy result");

    std::cout << "Sum = " << h_result << std::endl;

    cudaFree(d_input);
    cudaFree(d_result);
    return EXIT_SUCCESS;
}
