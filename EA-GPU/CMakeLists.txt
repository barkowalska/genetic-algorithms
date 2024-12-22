cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Ustawienie standardu CUDA
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dodanie katalogów nagłówkowych
include_directories(${CMAKE_SOURCE_DIR}/include)

# Znalezienie OpenMP
find_package(OpenMP REQUIRED)

# Preferowanie flagi -pthread zamiast -lpthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Dodanie głównego pliku wykonywalnego i jego plików źródłowych
add_executable(main
    test/test-main.cu
)

# Ustawienie właściwości dla kompilacji CUDA
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Ustawienie katalogów wyjściowych dla trybów Debug i Release
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Linkowanie OpenMP do docelowego pliku wykonywalnego
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

# Linkowanie wątków do docelowego pliku wykonywalnego
target_link_libraries(main PRIVATE Threads::Threads)

# Przekazanie flag OpenMP do kompilatora CUDA
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

# Opcje debugowania i optymalizacji
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -lineinfo") # Debug: szczegółowe debugowanie
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")  # Release: pełna optymalizacja

# Konfiguracja build typów
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Ustawienie opcji kompilacji dla Debug i Release
target_compile_options(main PRIVATE
    $<$<CONFIG:Debug>:-G -g -Xcompiler=-fopenmp>           # Debug: szczegółowe debugowanie
    $<$<CONFIG:Release>:-O3 -Xcompiler=-fopenmp>           # Release: optymalizacja
)

# Domyślny typ kompilacji to Release, jeśli nie został określony
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
