#include "image.hpp"
#pragma once
#include "opencl_utils.hpp"

Image regionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance);
Image regionGrowingCPU(const Image& image, int seedX, int seedY, int tolerance);
class RegionGrowingGPU {
public:
    RegionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance);
    ~RegionGrowingGPU();
    RegionGrowingGPU(const RegionGrowingGPU&) = delete;
    RegionGrowingGPU& operator=(const RegionGrowingGPU&) = delete;
    void run(const Image& image, Image& output);

private:
    void release();
    OpenCL cl;
    cl_kernel initKernel = nullptr;
    cl_kernel stepKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem maskBufferA = nullptr;
    cl_mem maskBufferB = nullptr;
    cl_mem changedBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
    int inputWidth;
    int inputHeight;
    int seedIndex;
};