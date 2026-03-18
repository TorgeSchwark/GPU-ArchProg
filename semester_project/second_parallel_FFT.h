#ifndef SECOND_PARALLEL_FFT_H
#define SECOND_PARALLEL_FFT_H

#include <vector>
#include <complex>

// ------------------------------------------------------------
// CUDA Parallel FFT (Host Wrapper)
// Implemented in parallel_FFT.cu
// ------------------------------------------------------------
float parallel_fft_base_twiddle(std::vector<std::complex<float>>& data);

#endif // PARALLEL_FFT_H