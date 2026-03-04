#ifndef COMPARE_FFT
#define COMPARE_FFT

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

bool compare_fft(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b,
    double tolerance
);

#endif