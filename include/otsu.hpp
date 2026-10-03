#pragma once
#include <array>
#include <cstddef>
#include "image.hpp"
#include "opencl_utils.hpp"

using Histogram = std::array<std::size_t, 256>;
Histogram calculateHistogram(const Image& image, int threads = 1);
int calculateOtsuThreshold(const Histogram& histogram);
Image otsuSegmentation(const Image& image, int threads = 1);
Image otsuSegmentationGPU(const Image& image);

class OtsuGPU {
public:
    explicit OtsuGPU(const Image& image);
    ~OtsuGPU();
    OtsuGPU(const OtsuGPU&) = delete;
    OtsuGPU& operator=(const OtsuGPU&) = delete;
    Histogram run(const Image& image, Image& output, cl_event* histogramEvent = nullptr, cl_event* thresholdEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel histogramKernel = nullptr;
    cl_kernel thresholdKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem histogramBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t pixelCount;
};
