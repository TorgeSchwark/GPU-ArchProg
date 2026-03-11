cmake_minimum_required(VERSION 3.18)

project(FFT_Project LANGUAGES CXX CUDA)

# ----------------------------------------------------
# Standards
# ----------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: sinnvolle Default-Build-Type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------
# Executable
# ----------------------------------------------------
add_executable(fft
    main.cpp
    sequential_FFT.cpp
    compare_FFT.cpp
    parallel_FFT.cu
    improved_parallel_FFT.cu
)

# ----------------------------------------------------
# CUDA Properties
# ----------------------------------------------------
set_target_properties(fft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Architektur anpassen!
# z.B. für RTX 30xx → 86
set_property(TARGET fft PROPERTY CUDA_ARCHITECTURES 75)

# ----------------------------------------------------
# FFTW über pkg-config
# ----------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

target_include_directories(fft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(fft PRIVATE ${FFTW_LIBRARIES})

# ----------------------------------------------------
# Compiler-Warnungen
# ----------------------------------------------------
if (MSVC)
    target_compile_options(fft PRIVATE /W4)
else()
    target_compile_options(fft PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ----------------------------------------------------
# Optimierung für Release
# ----------------------------------------------------
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        target_compile_options(fft PRIVATE /O2)
    else()
        target_compile_options(fft PRIVATE -O3 -march=native)
    endif()
endif()