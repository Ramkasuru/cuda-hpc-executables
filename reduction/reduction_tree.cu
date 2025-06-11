#include <iostream>
#include<cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256
__global__ void reduce_tree(const float* input, float* result, int n){
    __shared__float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Perform reduction in a tree-like manner
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];

        }
        __syncthreads();
    }
    // Write the result for this block to global memory
    if (tid ==0) atomicAdd(result, shared[0]);

}
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    size_t size = N * sizeof(float);
    float *h_input = new float[N];
    float *h_result = new float(0.0f);

    for (int i =0 < N; i++) {
        h_input[i] = 1.0f; // Initialize input data

    }
    float *d_input, *d_result;
    checkCudaError(cudaMalloc((void**)&d_input, size), "Allocating device input");
    checkCudaError(cudaMalloc((void**)&d_result, sizeof(float)), "Allocating device result");

    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Copying input to device");
    checkCudaError(cudaMemset(d_result, 0, sizeof(float)), "Initializing device result");

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_tree<<<numBlocks, BLOCK_SIZE>>>(d_input, d_result, N);

    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkcudaError(cudaDeviceSynchronize(), "Synchronizing device");

    checkCudaError(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "Copying result to host");
    std::cout << "Result from tree-based reduction: " << *h_result << std::endl;

    // Clean up
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
