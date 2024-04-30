cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)

set(CMAKE_OSX_ARCHITECTURES "arm64")

# Using mac 2.3.0 cpu version in lib. Change as needed
find_package(Torch REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch/)
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

set(RPATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
list(APPEND CMAKE_BUILD_RPATH ${RPATH})
message(STATUS "libomp found: ${RPATH}")

find_package(Threads REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

add_executable(main main.cpp)

target_include_directories(main PRIVATE ./lib)
target_link_libraries(main "${TORCH_LIBRARIES}" "${CMAKE_THREAD_LIBS_INIT}")
target_compile_features(main PUBLIC cxx_std_17)  # Ensures C++17

if(TORCH_CUDA_VERSION)
    message(STATUS "CUDA Version: ${TORCH_CUDA_VERSION}")
    add_definitions(-DUSE_CUDA)
endif()