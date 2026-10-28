#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <array>
#include <cstddef>

Image convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue);
void convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int threads = 1);
void convolutionRows(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int firstRow, int endRow);

class ConvolutionGPU {
public:
    ConvolutionGPU(const Image& image, const std::array<std::array<float, 3>, 3>& filter, bool absoluteValue);
    ~ConvolutionGPU();
    ConvolutionGPU(const ConvolutionGPU&) = delete;
    ConvolutionGPU& operator=(const ConvolutionGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    cl_mem filterBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
};
