#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void reduce_atomic(int *input, int *output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(output, input[tid]);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <num_elements> <threads_per_block>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *h_input = (int*)malloc(N * sizeof(int));
    int h_output = 0;

    // Initialize input with 1s for easy checking
    for (int i = 0; i < N; ++i)
        h_input[i] = 1;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce_atomic<<<blocks, threadsPerBlock>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum: %d\n", h_output);
    printf("Kernel time (ms): %f\n", ms);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
