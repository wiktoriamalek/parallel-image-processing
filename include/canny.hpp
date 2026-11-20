#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>
#include <vector>

Image gauss_function(const Image& image);
Image canny(const Image& image, float lowThreshold, float highThreshold);
void canny(const Image& image, Image& result, float lowThreshold, float highThreshold, int threads);

class CannyGPU {
public:
    CannyGPU(const Image& image, float lowThreshold, float highThreshold);
    ~CannyGPU();
    CannyGPU(const CannyGPU&) = delete;
    CannyGPU& operator=(const CannyGPU&) = delete;
    void run(const Image& image, Image& output, double* kernelUs = nullptr);
    int iterations() const { return iterationCount; }

private:
    void release();
    void launch(cl_kernel kernel, const std::size_t* global, const std::size_t* local, double* kernelUs);
    void addKernelTimes(double* kernelUs);
    OpenCL cl;
    cl_kernel gaussKernel = nullptr;
    cl_kernel gradientKernel = nullptr;
    cl_kernel suppressKernel = nullptr;
    cl_kernel trackKernel = nullptr;
    cl_kernel finishKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem blurredBuffer = nullptr;
    cl_mem magnitudeBuffer = nullptr;
    cl_mem sectorBuffer = nullptr;
    cl_mem stateBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    cl_mem changedBuffer = nullptr;
    cl_mem activeBufferA = nullptr;
    cl_mem activeBufferB = nullptr;
    int tilesX;
    int tilesY;
    std::vector<cl_event> kernelEvents;
    std::size_t bufferSize;
    int width;
    int height;
    std::size_t globalWorkSize[2];
    std::size_t tileGlobalWorkSize[2];
    int iterationCount = 0;
};
