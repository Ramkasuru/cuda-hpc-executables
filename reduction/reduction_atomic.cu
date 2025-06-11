#include <iostream>
#include <cuda_runtime.h>
#define N 1024

__global__ void reduce_atomic( const float* input, float* result) {
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

int main(){
    size_t size = N * sizeof(float);
    float * h_input = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Initialize input array with 1.0

    }
    // Allocating memory on the device
    float *d_input, *d_result;
    check_cuda_error(cudaMalloc(&d_input, size));
    check_cuda_error(cudaMalloc(&d_result, sizeof(float)));

    // Initialize result to zero on the device
    float h_result = 0.0f;

    // Copy data to device
    check_cuda_error(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_atomic<<<numBlocks, blockSize>>>(d_input, d_result);
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaDeviceSynchronize());

    // Copy result back
    check_cuda_error(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Result from atomic reduction: " << h_result << std::endl;

    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
