#pragma once

#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>

void bernsenRows(const Image& image, Image& result, int radius, int minContrast, int firstRow, int endRow);
void bernsenThreshold(const Image& image, Image& result, int radius, int minContrast, int threads = 1);

class BernsenGPU {
public:
    BernsenGPU(const Image& image, int radius, int minContrast);
    ~BernsenGPU();
    BernsenGPU(const BernsenGPU&) = delete;
    BernsenGPU& operator=(const BernsenGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
};
