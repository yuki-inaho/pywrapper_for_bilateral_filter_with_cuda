#include <string>
#include <pybind11/stl.h>
#include "ndarray_converter.h"
#include "pybind11/pybind11.h"

#include "bilateral_filter.cuh"

namespace py = pybind11;

PYBIND11_MODULE(pyBilateralFilter, m) {
    NDArrayConverter::init_numpy();
    m.def("bilateral_filter", &BilateralFilterGPU, "bilateral filtering with cuda");
}