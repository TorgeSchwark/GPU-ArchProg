#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <fftw3.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "sequential_FFT.h"
#include "parallel_FFT.h"
#include "imporved_parallel_FFT.h"
#include "second_parallel_FFT.h"
#include "compare_FFT.h"
#include "stockham_FFT.h"

using Clock = std::chrono::high_resolution_clock;

// ------------------------------------------------------------
// Random Data Generator
// ------------------------------------------------------------
std::vector<std::complex<float>>
generate_random_data(size_t N)
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    std::vector<std::complex<float>> data(N);
    for (size_t i = 0; i < N; ++i)
        data[i] = {dist(rng), dist(rng)};

    return data;
}

// ------------------------------------------------------------
// Sequential Benchmark
// ------------------------------------------------------------
double benchmark_seq_fft(const std::vector<std::complex<float>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto start = Clock::now();
        fft(temp);
        auto end = Clock::now();

        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total / runs;
}



// ------------------------------------------------------------
// CUDA Benchmark
// ------------------------------------------------------------
double benchmark_cuda_fft(const std::vector<std::complex<float>>& data, int runs, int block_size)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft(temp, block_size);

        total += time;
    }

    return total / runs;
}

double benchmark_cufft(const std::vector<std::complex<float>>& data, int runs)
{
    int N = data.size();

    cufftComplex* d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex) * N);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    double total = 0.0;

    for (int r = 0; r < runs; ++r)
    {
        // Copy input to GPU
        cudaMemcpy(d_data, data.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

        auto start = Clock::now();

        cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

        cudaDeviceSynchronize();

        auto end = Clock::now();

        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    cufftDestroy(plan);
    cudaFree(d_data);

    return total / runs;
}

// ------------------------------------------------------------
// FFTW Benchmark
// ------------------------------------------------------------
double benchmark_fftw(const std::vector<std::complex<float>>& data, int runs)
{
    int N = data.size();

    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

    fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    double total = 0.0;

    for (int r = 0; r < runs; ++r)
    {
        for (int i = 0; i < N; ++i) {
            in[i][0] = data[i].real();
            in[i][1] = data[i].imag();
        }

        auto start = Clock::now();
        fftwf_execute(plan);
        auto end = Clock::now();

        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return total / runs;
}

double benchmark_base_twiddle_fft(const std::vector<std::complex<float>>& data, int runs, int block_size)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft_base_twiddle(temp, block_size);

        total += time;
    }

    return total / runs;
}

double benchmark_stockham_fft(const std::vector<std::complex<float>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft_stockham(temp);

        total += time;
    }

    return total / runs;
}


double benchmark_precomputed(const std::vector<std::complex<float>>& data, int runs, int block_size)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft_fast(temp, block_size);

        total += time;
    }

    return total / runs;
}
// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
#include <iomanip>

int main()
{
    const int runs = 20;
    const size_t N = 4194304;

    auto data = generate_random_data(N);

    std::cout << std::left
              << std::setw(12) << "Block"
              << std::setw(16) << "cuFFT (ms)"
              << std::setw(16) << "Seq x"
              << std::setw(16) << "Base x"
              << std::setw(16) << "Pre x"
              << std::setw(16) << "Normal x"
              << "\n";

    std::cout << std::string(80, '-') << "\n";

    // Reference once (outside loop)
    auto ref_data = data;
    fft(ref_data);

    // cuFFT baseline
    double cufft_time = benchmark_cufft(data, runs);

    for (int block_size = 8; block_size <= 1024; block_size <<= 1)
    {
        double base_sum = 0.0;
        double seq_sum  = 0.0;
        double pre_sum  = 0.0;
        double norm_sum = 0.0;

        for (int i = 0; i < runs; ++i)
        {
            base_sum += benchmark_base_twiddle_fft(data, 1, block_size);
            seq_sum  += benchmark_seq_fft(data, 1);
            pre_sum  += benchmark_precomputed(data, 1, block_size);
            norm_sum += benchmark_cuda_fft(data, 1, block_size);
        }

        double base_avg = base_sum / runs;
        double seq_avg  = seq_sum / runs;
        double pre_avg  = pre_sum / runs;
        double norm_avg = norm_sum / runs;

        std::cout << std::left
                  << std::setw(12) << block_size
                  << std::setw(16) << cufft_time
                  << std::setw(16) << seq_avg / cufft_time
                  << std::setw(16) << base_avg / cufft_time
                  << std::setw(16) << pre_avg / cufft_time
                  << std::setw(16) << norm_avg / cufft_time
                  << "\n";
    }

    return 0;
}