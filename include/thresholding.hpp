#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>

Image thresholding(const Image& image, int threshold = 128);
void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max);
void ThresholdParallel(Image& image, int threshold, int threadCount);

class ThresholdGPU {
public:
    ThresholdGPU(const Image& image, int threshold);
    ~ThresholdGPU();
    ThresholdGPU(const ThresholdGPU&) = delete;
    ThresholdGPU& operator=(const ThresholdGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    std::size_t bufferSize;
};
