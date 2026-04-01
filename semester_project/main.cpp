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
double benchmark_cuda_fft(const std::vector<std::complex<float>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft(temp);

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

double benchmark_base_twiddle_fft(const std::vector<std::complex<float>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft_base_twiddle(temp);

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


double benchmark_precomputed(const std::vector<std::complex<float>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto time = parallel_fft_fast(temp);

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
    const int runs = 10;

    std::cout << std::left
              << std::setw(16)  << "N"
              << std::setw(16) << "seque (ms)"
              << std::setw(16) << "CUDA (ms)"
              << std::setw(16) << "FFTW (ms)"
              << std::setw(16) << "CUDA TWIDLE (ms)"
              << std::setw(16) << "CUDA Library (ms)"
              << std::setw(16) << "Seq x"
              << std::setw(16) << "CUDA x"
              << std::setw(16) << "CUDA TWIDLE x"
              << std::setw(16) << "CUDA PRECOMPUTED"
              << std::setw(16) << "CUDA stockham"
              << "\n";

    std::cout << std::string(68, '-') << "\n";

    for (int p = 8; p <= 26; ++p)   // 2^8 bis 2^14
    {
        size_t N = 1ULL << p;
        auto data = generate_random_data(N);

        // ----------------------------
        // Korrektheit prüfen
        // ----------------------------
        auto seq_data  = data;
        auto cuda_data = data;

        fft(seq_data);
        parallel_fft(cuda_data);

        int n = N;
        fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
        fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);

        fftwf_plan plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        for (int i = 0; i < n; ++i) {
            in[i][0] = data[i].real();
            in[i][1] = data[i].imag();
        }

        fftwf_execute(plan);

        std::vector<std::complex<float>> ref(n);
        for (int i = 0; i < n; ++i)
            ref[i] = {out[i][0], out[i][1]};

        fftwf_destroy_plan(plan);
        fftwf_free(in);
        fftwf_free(out);

        if (!compare_fft(seq_data, ref, 10000000.0f))
        {
            std::cout << "Sequential FFT incorrect for N = " << N << "\n";
            return 1;
        }

        if (!compare_fft(cuda_data, ref, 10000000.0f))
        {
            std::cout << "CUDA FFT incorrect for N = " << N << "\n";
            return 1;
        }

        // ----------------------------
        // Benchmark
        // ----------------------------
        double cuda_time = benchmark_cuda_fft(data, runs);
        double fftw_time = benchmark_fftw(data, runs);
        double twiddle_time = benchmark_base_twiddle_fft(data, runs);
        double seq_time = benchmark_seq_fft(data, runs);
        double precomputed = benchmark_precomputed(data, runs);
        double cuda_library = benchmark_cufft(data, runs);
        double stockham = benchmark_stockham_fft(data, runs);



        std::cout << std::left
                  << std::setw(16)  << N
                  << std::setw(16) << std::fixed << std::setprecision(4) << seq_time
                  << std::setw(16) << cuda_time
                  << std::setw(16) << fftw_time
                  << std::setw(16) << twiddle_time
                  << std::setw(16) << cuda_library
                  << std::setw(16) << seq_time / cuda_library
                  << std::setw(16) << cuda_time / cuda_library
                  << std::setw(16) << twiddle_time / cuda_library
                  << std::setw(16) << precomputed / cuda_library
                  << std::setw(16) << stockham / cuda_library

                  << "\n";
    }

    return 0;
}