#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>

void bernsenRows(const Image& image, Image& result, int radius, int minContrast, int firstRow, int endRow);
void bernsenThreshold(const Image& image, Image& result, int radius, int minContrast, int threads = 1);
void niblackThreshold(const Image& image, Image& result, int radius, double k, int threads = 1);
void sauvolaThreshold(const Image& image, Image& result, int radius, double k, int threads = 1);

enum class LocalThresholdMethod { Niblack, Sauvola };

class LocalThresholdGPU {
public:
    LocalThresholdGPU(const Image& image, LocalThresholdMethod method, int radius, double k);
    ~LocalThresholdGPU();
    LocalThresholdGPU(const LocalThresholdGPU&) = delete;
    LocalThresholdGPU& operator=(const LocalThresholdGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
    int width;
    int height;
    std::size_t globalWorkSize[2];
    std::size_t localWorkSize[2];
};

class BernsenGPU {
public:
    BernsenGPU(const Image& image, int radius, int minContrast);
    ~BernsenGPU();
    BernsenGPU(const BernsenGPU&) = delete;
    BernsenGPU& operator=(const BernsenGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* rowEvent = nullptr, cl_event* columnEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel rowKernel = nullptr;
    cl_kernel columnKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem rowMinMaxBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
    bool rowLocal = false;
    bool columnLocal = false;
};
