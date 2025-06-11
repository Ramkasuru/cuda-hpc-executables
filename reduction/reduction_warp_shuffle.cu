#include <iostream>
#include <cuda_runtime.h>

#define N 1024

// Warp-level reduction using shuffle down intrinsics
__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_warp_shuffle(const float *input, float *result, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (idx < n) {
        sum = input[idx];
    }

    // Reduce within warp
    sum = warpReduceSum(sum);

    // Shared memory for warp sums
    __shared__ float warpSums[32]; // Max 32 warps per block

    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the warp sums
    if (warpId == 0) {
        sum = (lane < blockDim.x / warpSize) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) atomicAdd(result, sum);
    }
}

// Error checking wrapper
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    size_t size = N * sizeof(float);
    float* h_input = new float[N];
    float h_result = 0.0f;

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_result;
    checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_result, sizeof(float)), "cudaMalloc d_result");

    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "cudaMemcpy input");
    checkCudaError(cudaMemset(d_result, 0, sizeof(float)), "cudaMemset d_result");

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    reduce_warp_shuffle<<<numBlocks, blockSize>>>(d_input, d_result, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Device sync");

    checkCudaError(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "Memcpy result");

    std::cout << "Result from warp shuffle reduction: " << h_result << std::endl;

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
