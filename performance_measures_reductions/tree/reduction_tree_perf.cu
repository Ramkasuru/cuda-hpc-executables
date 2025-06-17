#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void reduce_tree(int *input, int *output, int N) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) sdata[tid] = input[idx];
    else sdata[tid] = 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
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
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_tree<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, N);

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

