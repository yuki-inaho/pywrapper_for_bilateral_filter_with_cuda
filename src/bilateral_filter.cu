/*
Mainly 
https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/3_Imaging/bilateralFilter/bilateral_kernel.cu
*/
#include "bilateral_filter.cuh"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

__constant__ float cGaussian[64];
texture<unsigned char, 2, cudaReadModeElementType> grayTexture;

extern "C" void setPositionalGaussianDictionary(int radius, float std)
{
    float fGaussian[64];
    for (int i = 0; i < 2 * radius + 1; i++)
    {
        float x = i - radius;
        fGaussian[i] = expf(-(x * x) / (2 * std * std));
    }
    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * radius + 1));
}

__device__ inline float gaussian(float x, float sigma)
{
    return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

__global__ void op_bilateral_filter(unsigned char *input, unsigned char *output,
                                    int width, int height,
                                    int radius, float sigma_pos, float sigma_color)
{
	int x = __mul24(blockIdx.x, 16) + threadIdx.x;
	int y = __mul24(blockIdx.y, 16) + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    float total = 0.0f;
    float sum = 0.0f;    
    unsigned char center = tex2D(grayTexture, x, y);
    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            unsigned char curPix = tex2D(grayTexture, x + dx, y + dy);
            float weight = cGaussian[dy + radius] * cGaussian[dy + radius] * gaussian(curPix - center, sigma_color);
            total += weight * curPix;
            sum += weight;
        }
    }

    output[y * width + x] = uchar(total/sum);
}

cv::Mat BilateralFilterGPU(const cv::Mat &input_image, const int &radius, const float &sigma_pos, const float& sigma_color)
{
	int gray_size = input_image.step * input_image.rows;
    cv::Mat output_image = Mat::zeros(cv::Size(input_image.cols, input_image.rows), CV_8UC1);

    size_t pitch;
    unsigned char *d_input = NULL;
    unsigned char *d_output;

    setPositionalGaussianDictionary(radius, sigma_pos);

    cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input_image.step, input_image.rows);
    cudaMemcpy2D(d_input, pitch, input_image.ptr(), sizeof(unsigned char) * input_image.step, sizeof(unsigned char) * input_image.step, input_image.rows, cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, grayTexture, d_input, input_image.step, input_image.rows, pitch);
    cudaMalloc<unsigned char>(&d_output, gray_size);

    dim3 block(16, 16);
    dim3 grid((input_image.cols + block.x - 1) / block.x, (input_image.rows + block.y - 1) / block.y);

    op_bilateral_filter<<<grid, block>>>(d_input, d_output, input_image.cols, input_image.rows, radius, sigma_pos, sigma_color);
    cudaMemcpy(output_image.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output_image;
}