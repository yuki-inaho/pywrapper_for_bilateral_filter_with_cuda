
cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME pyBilateralFilter)
set(PROJ_NAME_CU pyBilateralFilterCUDA)
project(${PROJ_NAME})
find_package(CUDA REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_61,code=sm_61 -ccbin g++
)

find_package(PythonLibs 3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
MESSAGE("PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS})
MESSAGE("PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES})
add_subdirectory(extern/pybind11)

include_directories(
    include
    extern/pybind11/include
)


link_directories(
    /usr/local/cuda/lib64
)

cuda_add_library(
    ${PROJ_NAME_CU} SHARED src/bilateral_filter.cu
)

pybind11_add_module(
    ${PROJ_NAME}
    ${PYTHON_INCLUDE_DIRS}
    src/bindings.cpp
    src/ndarray_converter.cpp
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_CFLAGS})

target_link_libraries(${PROJ_NAME} ${OpenCV_LDFLAGS})
target_link_libraries(${PROJ_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJ_NAME} PUBLIC Python3::NumPy)

target_link_libraries(${PROJ_NAME_CU} ${OpenCV_LDFLAGS})
target_link_libraries(${PROJ_NAME} PRIVATE ${PROJ_NAME_CU})