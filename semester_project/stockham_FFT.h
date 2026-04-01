#ifndef STOCKHAM_FFT
#define STOCKHAM_FFT

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>

// CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;

// ------------------------------------------------------------
// Interface
// ------------------------------------------------------------
float parallel_fft_stockham(std::vector<std::complex<float>>& data);

#endif