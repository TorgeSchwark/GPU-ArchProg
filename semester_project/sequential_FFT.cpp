#include "sequential_FFT.h"


void fft(vector<complex<float>>& x) {
    const size_t N = x.size();

    // ---- Bit-Reversal ----
    size_t j = 0;
    for (size_t i = 1; i < N; i++) {
        size_t bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if (i < j)
            swap(x[i], x[j]);
    }

    // ---- Iterative Stages ----
    for (size_t m = 2; m <= N; m <<= 1) {
        float theta = -2.0 * M_PI / m;
        complex<float> wm(cos(theta), sin(theta));

        for (size_t k = 0; k < N; k += m) {
            complex<float> w(1.0, 0.0);
            for (size_t j = 0; j < m / 2; j++) {
                complex<float> t = w * x[k + j + m/2];
                complex<float> u = x[k + j];

                x[k + j]         = u + t;
                x[k + j + m/2]   = u - t;

                w *= wm;
            }
        }
    }
}


std::vector<std::complex<float>>
fftw_reference(const std::vector<std::complex<float>>& input)
{
    int N = input.size();

    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

    for (int i = 0; i < N; ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }

    fftwf_plan plan = fftwf_plan_dft_1d(
        N, in, out,
        FFTW_FORWARD,
        FFTW_ESTIMATE
    );

    fftwf_execute(plan);

    std::vector<std::complex<float>> result(N);

    for (int i = 0; i < N; ++i)
        result[i] = {out[i][0], out[i][1]};

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return result;
}

