cmake_minimum_required(VERSION 3.10)

# Set the project name and specify C++ as the language
project(MyProject LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Collect all source files in the src directory
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

# Create a static library from the source files
add_library(MyStaticLibrary STATIC ${SRC_FILES})

# Optionally, specify include directories
target_include_directories(MyStaticLibrary PUBLIC include)

# Add the executable (adjust path as necessary)
add_executable(test test/main.cpp)

# Link the static library to the executable
target_link_libraries(test PRIVATE MyStaticLibrary)

# Find OpenMP if not in Debug mode
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  find_package(OpenMP REQUIRED)
endif()

# Link OpenMP to the static library if available
if(OpenMP_FOUND)
  target_link_libraries(MyStaticLibrary PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link OpenMP to the executable if available
if(OpenMP_FOUND)
  target_link_libraries(test PRIVATE MyStaticLibrary OpenMP::OpenMP_CXX)
else()
  target_link_libraries(test PRIVATE MyStaticLibrary)
endif()

# Set compiler options for MyStaticLibrary
target_compile_options(MyStaticLibrary PRIVATE
  $<$<CONFIG:Debug>:-g -O0>
  $<$<CONFIG:Release>:-O3>
)

# Set compiler options for test executable
target_compile_options(test PRIVATE
  $<$<CONFIG:Debug>:-g -O0>
  $<$<CONFIG:Release>:-O3>
)

# Set output directories based on build type
set_target_properties(MyStaticLibrary PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}"
)

set_target_properties(test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}"
)
