#include <iostream>
#include <cuda_runtime.h>

// Kernel: runs on GPU, does vector addition in parallel
__global__ void vecadd_kernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Host function: prepares data, calls kernel, and cleans up
void vecadd(float* A, float* B, float* C, int N) {
    float *A_d, *B_d, *C_d;

    // Allocate device memory
    cudaMalloc((void**)&A_d, N * sizeof(float));
    cudaMalloc((void**)&B_d, N * sizeof(float));
    cudaMalloc((void**)&C_d, N * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Launch kernel on GPU
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);

    // Copy result vector from device to host
    cudaMemcpy(C, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int N = 1024;
    float A[N], B[N], C[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        A[i] = float(i);
        B[i] = float(2 * i);
    }

    // Call the vector addition function
    vecadd(A, B, C, N);

    // Check a few results
    for (int i = 0; i < 5; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    return 0;
}
