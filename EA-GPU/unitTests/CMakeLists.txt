# unitTests/CMakeLists.txt

# Collect all test source files
set(TEST_SOURCES
#   test_tournament_selection.cu
    test_boltzmannTournament_selection.cu
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# For each test source file, create an executable and add a test
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the test name without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Add executable for the test
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Link Google Test and CUDA libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        gtest
        gtest_main
        cuda
        cudart
    )

    # Set properties for CUDA compilation
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
    )

    # Add the test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
