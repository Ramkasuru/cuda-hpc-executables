#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int reduction_cpu(int *input, int N) {
    int sum = 0;
    for (int i = 0; i < N; ++i) sum += input[i];
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_elements>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int *h_input = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int sum = reduction_cpu(h_input, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                (end.tv_nsec - start.tv_nsec) / 1e6;

    printf("Sum: %d\n", sum);
    printf("CPU time (ms): %f\n", ms);

    free(h_input);
    return 0;
}
