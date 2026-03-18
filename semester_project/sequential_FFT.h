#ifndef SEQUENTIAL_FFT
#define SEQUENTIAL_FFT

#include <iostream>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

void fft(vector<complex<float>>& x);

std::vector<std::complex<float>>
fftw_reference(const std::vector<std::complex<float>>& input);

#endif