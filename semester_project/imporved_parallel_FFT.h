#ifndef IMPROVED_PARALLEL_FFT_H
#define IMPROVED_PARALLEL_FFT_H

#include <vector>
#include <complex>


// ------------------------------------------------------------
// CUDA Parallel FFT (Host Wrapper)
// Implemented in parallel_FFT.cu
// ------------------------------------------------------------
double parallel_fft_fast(std::vector<std::complex<float>>& data, int threads);

#endif // PARALLEL_FFT_H