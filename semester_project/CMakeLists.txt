cmake_minimum_required(VERSION 3.16)

# Projektname und Sprache
project(FFT_Project LANGUAGES CXX)

# C++ Standard setzen
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executable definieren
add_executable(fft
    main.cpp
    sequential_FFT.cpp
)

# Optional: Warnungen aktivieren
if (MSVC)
    target_compile_options(fft PRIVATE /W4)
else()
    target_compile_options(fft PRIVATE -Wall -Wextra -Wpedantic)
endif()