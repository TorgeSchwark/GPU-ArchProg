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

// ------------------------------------------------------------
// Precompute staged twiddles
// ------------------------------------------------------------
void precompute_twiddles_staged(
    int N,
    std::vector<cuFloatComplex>& twiddles,
    std::vector<int>& offsets)
{
    int stages = __builtin_ctz(N);

    offsets.resize(stages);
    twiddles.clear();

    int offset = 0;

    for (int s = 1; s <= stages; ++s)
    {
        int m = 1 << s;
        int half = m >> 1;

        offsets[s-1] = offset;

        double theta = -2.0 * M_PI / m;
        double c = cos(theta);
        double s_ = sin(theta);

        double wr = 1.0;
        double wi = 0.0;

        for (int k = 0; k < half; ++k)
        {
            twiddles.push_back(make_cuFloatComplex(wr, wi));

            double new_wr = wr * c - wi * s_;
            double new_wi = wr * s_ + wi * c;

            wr = new_wr;
            wi = new_wi;
        }

        offset += half;
    }
}


__global__
void bit_reverse_kernel_fast(cuFloatComplex* x, int N, int logN)
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

// ------------------------------------------------------------
// FFT Stage Kernel
// ------------------------------------------------------------
__global__
void fft_stage_kernel_fast(
    cuFloatComplex* __restrict__ x,
    const cuFloatComplex* __restrict__ stage_twiddles,
    int N,
    int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int half = m >> 1;

    int groups = N / m;
    int butterflies = groups * half;

    if (tid >= butterflies) return;

    int group = tid / half;
    int j     = tid % half;

    int base = group * m;

    int i1 = base + j;
    int i2 = i1 + half;

    cuFloatComplex u = x[i1];
    cuFloatComplex v = x[i2];

    cuFloatComplex w = __ldg(&stage_twiddles[j]);

    cuFloatComplex t = cmul(w, v);

    x[i1] = cuCaddf(u, t);
    x[i2] = cuCsubf(u, t);
}

// ------------------------------------------------------------
// FFT Host Wrapper
// ------------------------------------------------------------
double parallel_fft_fast(std::vector<std::complex<float>>& data, int threads)
{

    int N = data.size();

    cuFloatComplex* d_x;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(cuFloatComplex)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        reinterpret_cast<cuFloatComplex*>(data.data()),
        N * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice));

    int blocks  = (N + threads - 1) / threads;

    // Twiddle generation track time of this halfe twiddle values dont need to be recomputed(Last call)
    std::vector<cuFloatComplex> twiddles;
    std::vector<int> offsets;
    
    precompute_twiddles_staged(N, twiddles, offsets);

    cuFloatComplex* d_twiddles;

    CUDA_CHECK(cudaMalloc(
        &d_twiddles,
        twiddles.size() * sizeof(cuFloatComplex)));

    CUDA_CHECK(cudaMemcpy(
        d_twiddles,
        twiddles.data(),
        twiddles.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Bit reversal messure time!
    // --------------------------------------------------------

    int logN = __builtin_ctz(N);

    bit_reverse_kernel_fast<<<blocks, threads>>>(d_x, N, logN);

    // --------------------------------------------------------
    // FFT stages
    // --------------------------------------------------------
    int stage = 0;  
    auto start = Clock::now();

    for (int m = 2; m <= N; m <<= 1)
    {
        int groups = N / m;
        int butterflies = groups * (m >> 1);

        int stageBlocks = (butterflies + threads - 1) / threads;

        cuFloatComplex* stage_ptr =
            d_twiddles + offsets[stage];

        int half = m >> 1;

        int sharedMem = 0;

        if (half <= 1024)
            sharedMem = half * sizeof(cuFloatComplex);

        fft_stage_kernel_fast<<<stageBlocks, threads, sharedMem>>>(
            d_x,
            stage_ptr,
            N,
            m);

        stage++;
    }

    cudaDeviceSynchronize();

    auto end = Clock::now();
    
    // --------------------------------------------------------
    // Copy back
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<cuFloatComplex*>(data.data()),
        d_x,
        N * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_twiddles);

    cudaDeviceSynchronize();
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