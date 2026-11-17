#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include "otsu.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using SmoothedHistogram = std::array<std::uint64_t, 256>;

std::vector<int> peaks_valleys_detectionThresholds(const Histogram& histogram, int maxClasses, int* iterations = nullptr);
int peaks_valleys_detectionRange(const SmoothedHistogram& histogram, int left_peak_intensity, int right_peak_intensity);
Image peaks_valleys_detection(const Image& image, int maxClasses = 3);
void peaks_valleys_detectionParallel(const Image& image, Image& output, int threads, int maxClasses = 3);

class Peaks_valleys_detectionGPU {
public:
    Peaks_valleys_detectionGPU(const Image& image, int maxClasses = 3, bool zeroCopy = true);
    ~Peaks_valleys_detectionGPU();
    Peaks_valleys_detectionGPU(const Peaks_valleys_detectionGPU&) = delete;
    Peaks_valleys_detectionGPU& operator=(const Peaks_valleys_detectionGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* histogramEvent = nullptr,
             cl_event* thresholdsEvent = nullptr, cl_event* lookupEvent = nullptr);
    void upload(const Image& image, Image& output);
    void compute(cl_event* histogramEvent = nullptr, cl_event* thresholdsEvent = nullptr, cl_event* lookupEvent = nullptr);
    void finish();
    void download(Image& output);
    const std::vector<int>& thresholds() const { return thresholdValues; }
    int iterations() const { return iterationCount; }

private:
    void release();
    void setImageArguments();
    void bindImages(const Image& image, Image& output);
    OpenCL cl;
    int maxClasses;
    bool zeroCopy;
    cl_kernel clearHistogramKernel = nullptr;
    cl_kernel histogramKernel = nullptr;
    cl_kernel thresholdsKernel = nullptr;
    cl_kernel lookupKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem histogramBuffer = nullptr;
    cl_mem resultBuffer = nullptr;
    cl_mem lookupBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t pixelCount;
    unsigned char* boundInput = nullptr;
    unsigned char* boundOutput = nullptr;
    std::vector<int> result;
    std::vector<int> thresholdValues;
    int iterationCount = 0;
};
