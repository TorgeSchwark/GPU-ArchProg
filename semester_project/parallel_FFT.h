#ifndef PARALLEL_FFT_H
#define PARALLEL_FFT_H

#include <vector>
#include <complex>

// ------------------------------------------------------------
// CUDA Parallel FFT (Host Wrapper)
// Implemented in parallel_FFT.cu
// ------------------------------------------------------------
double parallel_fft(std::vector<std::complex<float>>& data);

#endif // PARALLEL_FFT_H