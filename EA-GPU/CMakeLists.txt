cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Specify the executable and its source files
add_executable(MyCUDAExecutable
    test/test-main.cu
)

# Set properties for CUDA compilation
set_target_properties(MyCUDAExecutable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
