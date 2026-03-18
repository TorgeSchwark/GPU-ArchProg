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
// Fast Complex Multiply
// ------------------------------------------------------------
__device__ __forceinline__
cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(
        cuCrealf(a)*cuCrealf(b) - cuCimagf(a)*cuCimagf(b),
        cuCrealf(a)*cuCimagf(b) + cuCimagf(a)*cuCrealf(b));
}

// ------------------------------------------------------------
// Bit reversal
// ------------------------------------------------------------
__device__ __forceinline__
int reverse_bits(int x, int logN)
{
    return __brev(x) >> (32 - logN);
}



__global__
void bit_reverse_kernel_base(cuFloatComplex* x, int N, int logN)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int j = reverse_bits(i, logN);

    if (i < j)
    {
        cuFloatComplex tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

__device__ __forceinline__
cuFloatComplex cpow_int(cuFloatComplex base, int exp)
{
    cuFloatComplex result = make_cuFloatComplex(1.0,0.0);

    while(exp)
    {
        if(exp & 1)
            result = cmul(result, base);

        base = cmul(base, base);
        exp >>= 1;
    }

    return result;
}

__global__
void fft_stage_kernel_base(
    cuFloatComplex* __restrict__ x,
    cuFloatComplex base_twiddle,
    int N,
    int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int half = m >> 1;

    int butterflies = N >> 1;

    if (tid >= butterflies) return;

    int group = tid / half;
    int j     = tid % half;

    int base = group * m;

    int i1 = base + j;
    int i2 = i1 + half;

    cuFloatComplex u = x[i1];
    cuFloatComplex v = x[i2];

    // compute twiddle
    cuFloatComplex w = cpow_int(base_twiddle, j);

    cuFloatComplex t = cmul(w, v);

    x[i1] = cuCaddf(u, t);
    x[i2] = cuCsubf(u, t);
}

float parallel_fft_base_twiddle(std::vector<std::complex<float>>& data)
{
    int N = data.size();

    cuFloatComplex* d_x;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(cuFloatComplex)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        reinterpret_cast<cuFloatComplex*>(data.data()),
        N * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice));

    int threads = 512;
    int blocks  = (N + threads - 1) / threads;

    auto start = Clock::now();

    int logN = __builtin_ctz(N);

    bit_reverse_kernel_base<<<blocks, threads>>>(d_x, N, logN);
    CUDA_CHECK(cudaDeviceSynchronize());

    int stage = 0;

    for (int m = 2; m <= N; m <<= 1)
    {
        int half = m >> 1;

        int butterflies = N >> 1;

        int stageBlocks =
            (butterflies + threads - 1) / threads;

        // compute base twiddle
        double angle = -2.0 * M_PI / m;

        cuFloatComplex base_twiddle =
            make_cuFloatComplex(cos(angle), sin(angle));

        fft_stage_kernel_base<<<stageBlocks, threads>>>(
            d_x,
            base_twiddle,
            N,
            m);


        stage++;
    }
    cudaDeviceSynchronize();

    auto end = Clock::now();


    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<cuFloatComplex*>(data.data()),
        d_x,
        N * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost));

    cudaFree(d_x);

    return std::chrono::duration<double, std::milli>(end - start).count();

}
// ------------------------------------------------------------
// Test
// ------------------------------------------------------------
#ifdef TEST_PARALLEL_FFT

int main()
{
    const int N = 1024;

    std::vector<std::complex<double>> data(N);

    for (int i = 0; i < N; ++i)
        data[i] = {1.0, 0.0};

    parallel_fft_fast(data);

    for (int i = 0; i < 10; ++i)
        std::cout << data[i] << std::endl;

    return 0;
}

#endif