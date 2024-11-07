cmake_minimum_required(VERSION 3.10)

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$<0:>")

# Set the output directory for shared libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/$<0:>")

# Set the output directory for static libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/$<0:>")

# Set the project name and specify C++ as the language
project(MyProject LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
