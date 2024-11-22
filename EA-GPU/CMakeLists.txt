cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Ustawienie standardu CUDA
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dodanie katalogów nagłówkowych
include_directories(${CMAKE_SOURCE_DIR}/include)

# Włączenie testowania
enable_testing()

# Dołączenie GoogleTest za pomocą FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# Dla Windows: Zapobieganie nadpisywaniu ustawień kompilatora/linkera projektu nadrzędnego
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

# Linkowanie OpenMP do docelowego pliku wykonywalnego
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

# Linkowanie wątków do docelowego pliku wykonywalnego
target_link_libraries(main PRIVATE Threads::Threads)

# Przekazanie flag OpenMP do kompilatora CUDA
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

# Enable debugging symbols for CUDA
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Add debug flags to CUDA target
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g -Xcompiler=-fopenmp>
)

# Default to Debug build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
