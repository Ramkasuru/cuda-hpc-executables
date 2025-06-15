#include <iostream>
#include <cuda_runtime.h>
#define N 1024

__global__ void reduce_atomic( const float* input, float* result) {
    int i = threadIdx.x + blockIdx.x * blaockDim.x;
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
    floast * h_input = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Initialize input array with 1.0
        
    }
}
