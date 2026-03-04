#include "sequential_FFT.h"


void fft(vector<complex<double>>& x) {
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
        double theta = -2.0 * M_PI / m;
        complex<double> wm(cos(theta), sin(theta));

        for (size_t k = 0; k < N; k += m) {
            complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < m / 2; j++) {
                complex<double> t = w * x[k + j + m/2];
                complex<double> u = x[k + j];

                x[k + j]         = u + t;
                x[k + j + m/2]   = u - t;

                w *= wm;
            }
        }
    }
}

