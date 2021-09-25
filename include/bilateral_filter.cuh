#pragma once

/*
https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/3_Imaging/bilateralFilter/bilateral_kernel.cu
*/
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

cv::Mat BilateralFilterGPU(const cv::Mat &input_image, const int &radius = 9, const float &sigma_pos = 10.0, const float& sigma_color = 30.0);
