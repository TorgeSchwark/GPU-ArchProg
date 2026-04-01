#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

#include <chrono>

using Clock = std::chrono::high_resolution_clock;

using namespace std;


#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ------------------------------------------------------------
// Bit-Reversal Kernel
// ------------------------------------------------------------
__global__
void bit_reverse_kernel(cuFloatComplex* x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int j = 0;
    int n = N;
    int bit = n >> 1;
    int temp = i;

    while (bit > 0) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
        bit >>= 1;
    }

    if (i < j) {
        cuFloatComplex tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

// ------------------------------------------------------------
// Single FFT Stage Kernel
// ------------------------------------------------------------
__global__
void fft_stage_kernel(cuFloatComplex* x, int N, int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = m >> 1;

    int total_butterflies = N >> 1;
    if (tid >= total_butterflies) return;

    int group = tid / half;
    int j     = tid % half;

    int k = group * m;

    float theta = -2.0f * M_PI * j / m; // precompute 1/m ???

    float s, c;
    __sincosf(theta, &s, &c);

    cuFloatComplex w = make_cuFloatComplex(c, s);

    cuFloatComplex u = x[k + j];
    cuFloatComplex t = cuCmulf(w, x[k + j + half]);

    x[k + j]        = cuCaddf(u, t);
    x[k + j + half] = cuCsubf(u, t);
}

// ------------------------------------------------------------
// Host Wrapper
// ------------------------------------------------------------
double parallel_fft(std::vector<std::complex<float>>& data)
{
    int N = data.size();

    cuFloatComplex* d_x;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(cuFloatComplex)));

    // Copy to device
    std::vector<cuFloatComplex> temp(N);
    for (int i = 0; i < N; ++i)
        temp[i] = make_cuFloatComplex(data[i].real(), data[i].imag());

    CUDA_CHECK(cudaMemcpy(d_x, temp.data(),
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));

    int threads = 512;
    int blocks  = (N + threads - 1) / threads;


    // ---- Bit reversal ----
    bit_reverse_kernel<<<blocks, threads>>>(d_x, N);

    auto start = Clock::now();


    // ---- FFT Stages ----
    for (int m = 2; m <= N; m <<= 1)
    {
        int butterflies = N >> 1;
        int stageBlocks = (butterflies + threads - 1) / threads;

        fft_stage_kernel<<<stageBlocks, threads>>>(d_x, N, m);
    }
    cudaDeviceSynchronize();
    auto end = Clock::now();

    // Copy back
    CUDA_CHECK(cudaMemcpy(temp.data(), d_x,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i)
        data[i] = std::complex<float>(cuCrealf(temp[i]), cuCimagf(temp[i]));

    cudaFree(d_x);
    cudaDeviceSynchronize();

    return std::chrono::duration<double, std::milli>(end - start).count();

}

// ------------------------------------------------------------
// Test Main (optional)
// ------------------------------------------------------------
#ifdef TEST_PARALLEL_FFT
int main()
{
    const int N = 1024;

    std::vector<std::complex<double>> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = {1.0, 0.0};

    parallel_fft(data);

    for (int i = 0; i < 10; ++i)
        std::cout << data[i] << std::endl;

    return 0;
}
#endif