cmake_minimum_required(VERSION 3.16)

project(FFT_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fft
    main.cpp
    sequential_FFT.cpp
    compare_FFT.cpp
)

# ---- FFTW über pkg-config ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

target_include_directories(fft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(fft PRIVATE ${FFTW_LIBRARIES})

# Optional: Warnungen
if (MSVC)
    target_compile_options(fft PRIVATE /W4)
else()
    target_compile_options(fft PRIVATE -Wall -Wextra -Wpedantic)
endif()