#include "compare_FFT.h"

bool compare_fft(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b,
    double tolerance
) {
    if (a.size() != b.size())
        return false;

    bool ok = true;

    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);

        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << "\n";
            std::cout << "My FFT:     " << a[i] << "\n";
            std::cout << "FFTW FFT:   " << b[i] << "\n";
            std::cout << "Difference: " << diff << "\n\n";
            ok = false;
        }
    }

    return ok;
}