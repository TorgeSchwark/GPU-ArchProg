#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <fftw3.h>

#include "sequential_FFT.h"
#include "parallel_FFT.h"
#include "compare_FFT.h"

using Clock = std::chrono::high_resolution_clock;

// ------------------------------------------------------------
// Random Data Generator
// ------------------------------------------------------------
std::vector<std::complex<double>>
generate_random_data(size_t N)
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::complex<double>> data(N);
    for (size_t i = 0; i < N; ++i)
        data[i] = {dist(rng), dist(rng)};

    return data;
}

// ------------------------------------------------------------
// Sequential Benchmark
// ------------------------------------------------------------
double benchmark_seq_fft(const std::vector<std::complex<double>>& data, int runs)
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
double benchmark_cuda_fft(const std::vector<std::complex<double>>& data, int runs)
{
    double total = 0.0;

    for (int i = 0; i < runs; ++i)
    {
        auto temp = data;

        auto start = Clock::now();
        parallel_fft(temp);
        auto end = Clock::now();

        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total / runs;
}

// ------------------------------------------------------------
// FFTW Benchmark
// ------------------------------------------------------------
double benchmark_fftw(const std::vector<std::complex<double>>& data, int runs)
{
    int N = data.size();

    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    double total = 0.0;

    for (int r = 0; r < runs; ++r)
    {
        for (int i = 0; i < N; ++i) {
            in[i][0] = data[i].real();
            in[i][1] = data[i].imag();
        }

        auto start = Clock::now();
        fftw_execute(plan);
        auto end = Clock::now();

        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

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
              << std::setw(8)  << "N"
              << std::setw(12) << "Seq (ms)"
              << std::setw(12) << "CUDA (ms)"
              << std::setw(12) << "FFTW (ms)"
              << std::setw(12) << "Seq x"
              << std::setw(12) << "CUDA x"
              << "\n";

    std::cout << std::string(68, '-') << "\n";

    for (int p = 8; p <= 24; ++p)   // 2^8 bis 2^14
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
        fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

        fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        for (int i = 0; i < n; ++i) {
            in[i][0] = data[i].real();
            in[i][1] = data[i].imag();
        }

        fftw_execute(plan);

        std::vector<std::complex<double>> ref(n);
        for (int i = 0; i < n; ++i)
            ref[i] = {out[i][0], out[i][1]};

        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        if (!compare_fft(seq_data, ref, 1e-5))
        {
            std::cout << "Sequential FFT incorrect for N = " << N << "\n";
            return 1;
        }

        if (!compare_fft(cuda_data, ref, 1e-5))
        {
            std::cout << "CUDA FFT incorrect for N = " << N << "\n";
            return 1;
        }

        // ----------------------------
        // Benchmark
        // ----------------------------
        double seq_time  = benchmark_seq_fft(data, runs);
        double cuda_time = benchmark_cuda_fft(data, runs);
        double fftw_time = benchmark_fftw(data, runs);

        std::cout << std::left
                  << std::setw(8)  << N
                  << std::setw(12) << std::fixed << std::setprecision(4) << seq_time
                  << std::setw(12) << cuda_time
                  << std::setw(12) << fftw_time
                  << std::setw(12) << seq_time / fftw_time
                  << std::setw(12) << cuda_time / fftw_time
                  << "\n";
    }

    return 0;
}