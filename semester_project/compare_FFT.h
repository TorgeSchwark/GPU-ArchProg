#ifndef COMPARE_FFT
#define COMPARE_FFT

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

bool compare_fft(
    const std::vector<std::complex<float>>& a,
    const std::vector<std::complex<float>>& b,
    float tolerance
);

#endif