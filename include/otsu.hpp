#pragma once
#include <array>
#include <cstddef>
#include <vector>
#include "image.hpp"
#include "opencl_utils.hpp"

using Histogram = std::array<std::size_t, 256>;
Histogram calculateHistogram(const Image& image, int threads = 1);
int calculateOtsuThreshold(const Histogram& histogram);
Image otsuSegmentation(const Image& image, int threads = 1);
Image otsuSegmentationGPU(const Image& image);
std::vector<int> calculateMultiOtsuThresholds(const Histogram& histogram, int classes);
std::array<unsigned char, 256> multiOtsuLookupTable(const std::vector<int>& thresholds);
Image multiOtsuSegmentation(const Image& image, int classes, int threads = 1);

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

class MultiOtsuGPU {
public:
    MultiOtsuGPU(const Image& image, int classes, bool zeroCopy = true);
    ~MultiOtsuGPU();
    MultiOtsuGPU(const MultiOtsuGPU&) = delete;
    MultiOtsuGPU& operator=(const MultiOtsuGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* histogramEvent = nullptr,
             std::vector<cl_event>* thresholdsEvents = nullptr, cl_event* lookupEvent = nullptr);
    void upload(const Image& image, Image& output);
    void compute(cl_event* histogramEvent = nullptr, std::vector<cl_event>* thresholdsEvents = nullptr,
                 cl_event* lookupEvent = nullptr);
    void finish();
    void download(Image& output);
    const std::vector<int>& thresholds() const { return thresholdValues; }

private:
    void release();
    void setImageArguments();
    void bindImages(const Image& image, Image& output);
    OpenCL cl;
    int classes;
    bool zeroCopy;
    cl_kernel clearHistogramKernel = nullptr;
    cl_kernel histogramKernel = nullptr;
    cl_kernel prefixKernel = nullptr;
    cl_kernel layerKernel = nullptr;
    cl_kernel lookupTableKernel = nullptr;
    cl_kernel lookupKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem histogramBuffer = nullptr;
    cl_mem prefixCountBuffer = nullptr;
    cl_mem prefixSumBuffer = nullptr;
    cl_mem layersBuffer = nullptr;
    cl_mem splitBuffer = nullptr;
    cl_mem thresholdsBuffer = nullptr;
    cl_mem lookupBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t pixelCount;
    unsigned char* boundInput = nullptr;
    unsigned char* boundOutput = nullptr;
    std::vector<int> thresholdValues;
};

Image multiOtsuSegmentationGPU(const Image& image, int classes, bool zeroCopy = true);
