# unitTests/CMakeLists.txt

# Collect all test source files
file(GLOB UNIT_TEST_SOURCES *.cu)

# Add executable for unit tests
add_executable(UnitTests ${UNIT_TEST_SOURCES})

# Include directories
target_include_directories(UnitTests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Google Test and CUDA libraries
target_link_libraries(UnitTests
    PRIVATE
    gtest
    gtest_main
    cuda
    cudart
)

# Set properties for CUDA compilation
set_target_properties(UnitTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# Add the test to CTest
add_test(NAME UnitTests COMMAND UnitTests)
