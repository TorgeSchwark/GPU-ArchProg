#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include "sequential_FFT.h"
#include "compare_FFT.h"

using namespace std;

// Semester Project implementing the Cooley-Tukey-Algorithm to solve FFT
int main() {

    std::vector<std::complex<double>> data = {1,0,0,0,0,0,0,0};

    auto my_data = data;
    fft(my_data);

    auto ref_data = fftw_reference(data);

    bool ok = compare_fft(my_data, ref_data, 1e-9);

    if (ok)
        std::cout << "FFT OK ✅" << std::endl;
    else
        std::cout << "FFT WRONG ❌" << std::endl;

    return 0;
}