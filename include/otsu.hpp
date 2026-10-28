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
    void run(const Image& image, Image& output, cl_event* histogramEvent = nullptr,
             cl_event* otsuThresholdEvent = nullptr, cl_event* thresholdEvent = nullptr);
    // run split into parts, so the computation can be measured without copying the image to and from the GPU
    void upload(const Image& image);
    void compute(cl_event* histogramEvent = nullptr, cl_event* otsuThresholdEvent = nullptr, cl_event* thresholdEvent = nullptr);
    void finish();
    void download(Image& output);

private:
    void release();
    OpenCL cl;
    cl_kernel histogramKernel = nullptr;
    cl_kernel thresholdKernel = nullptr;
    cl_kernel otsuThresholdKernel = nullptr;
    cl_kernel clearHistogramKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem histogramBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    cl_mem thresholdBuffer = nullptr;
    std::size_t pixelCount;
};
