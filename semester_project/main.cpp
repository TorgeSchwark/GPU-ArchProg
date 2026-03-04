#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>

#include "sequential_FFT.h"
#include "compare_FFT.h"

std::vector<std::complex<double>>
generate_random_data(size_t N)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<std::complex<double>> data(N);

    for (size_t i = 0; i < N; ++i)
        data[i] = {dist(rng), dist(rng)};

    return data;
}

bool run_tests(size_t num_tests)
{
    bool all_ok = true;

    for (size_t t = 0; t < num_tests; ++t)
    {
        // Zufällige Zweierpotenz zwischen 2^3 und 2^12
        size_t power = 3 + (t % 10);   // 8 bis 4096
        size_t N = 1ULL << power;

        std::cout << "Test " << t+1
                  << " | N = " << N << std::endl;

        auto data = generate_random_data(N);

        auto my_data = data;
        fft(my_data);

        auto ref_data = fftw_reference(data);

        bool ok = compare_fft(my_data, ref_data, 1e-8);

        if (!ok) {
            std::cout << "FAILED at N = " << N << "\n\n";
            all_ok = false;
            break;
        }
        else {
            std::cout << "OK\n\n";
        }
    }

    return all_ok;
}

int main()
{
    size_t num_tests = 20;

    bool success = run_tests(num_tests);

    if (success)
        std::cout << "ALL TESTS PASSED ✅" << std::endl;
    else
        std::cout << "TEST FAILURE ❌" << std::endl;

    return 0;
}