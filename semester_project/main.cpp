#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include "sequential_FFT.h"

using namespace std;

// Semester Project implementing the Cooley-Tukey-Algorithm to solve FFT
int main() {
    vector<complex<double>> data = {1,0,0,0,0,0,0,0};
    fft(data);

    for (auto& v : data)
        cout << v << endl;

    return 0;
}