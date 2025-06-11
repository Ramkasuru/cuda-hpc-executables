#include <iostream>
#include <chrono>
#include <vector>
#define N (1<<5)
int main(){
    std:: vector<float> data(N);

    for (int i = 0; i < N; i++)
      data[i] = 1.0f;
    auto start = std::chrono::high_resolution_clock::now();

    float sum = 0.0f;
    for (int i = 0; i < N; i++)
      sum += data[i];
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    return 0;

}
