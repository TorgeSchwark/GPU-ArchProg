#include "stockham_FFT.h"

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
// ------------------------------------------------------------
// Stockham FFT Kernel
// ------------------------------------------------------------
__global__
void fft_stage_kernel_stockham(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    cuFloatComplex base_twiddle,
    int N,
    int m,
    int log_m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int half = m >> 1;

    int j = tid & (m - 1);        // position in group
    int k = tid >> log_m; // group index

    int j_mod = j & (half - 1);

    // indices in input
    int i1 = k * m + j_mod;
    int i2 = i1 + half;

    cuFloatComplex u = in[i1];
    cuFloatComplex v = in[i2];

    // twiddle
    cuFloatComplex w = cpow_int(base_twiddle, j);

    cuFloatComplex t = cmul(w, v);


    // output index (linear, coalesced)
    int out_idx = tid;

    if (j < half)
        out[out_idx] = cuCaddf(u, t);
    else
        out[out_idx] = cuCsubf(u, t);
}

// ------------------------------------------------------------
// Stockham FFT Wrapper
// ------------------------------------------------------------
float parallel_fft_stockham(std::vector<std::complex<float>>& data)
{
    int N = data.size();

    cuFloatComplex* d_in;
    cuFloatComplex* d_out;

    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(cuFloatComplex)));

    CUDA_CHECK(cudaMemcpy(
        d_in,
        reinterpret_cast<cuFloatComplex*>(data.data()),
        N * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice));

    int threads = 512;
    int blocks  = (N + threads - 1) / threads;

    auto start = Clock::now();

    // stages
    for (int m = 2; m <= N; m <<= 1)
    {
        int log_m = __builtin_ctz(m);
        double angle = -2.0 * M_PI / m;

        cuFloatComplex base_twiddle =
            make_cuFloatComplex(cos(angle), sin(angle));
        fft_stage_kernel_stockham<<<blocks, threads>>>(
            d_in,
            d_out,
            base_twiddle,
            N,
            m,
            log_m);

        // swap buffers
        cuFloatComplex* tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = Clock::now();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<cuFloatComplex*>(data.data()),
        d_in,
        N * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);

    return std::chrono::duration<double, std::milli>(end - start).count();
}