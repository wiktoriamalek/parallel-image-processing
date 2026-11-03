#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <vector>

Image regionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance, bool localTiles = true);
Image regionGrowingCPU(const Image& image, int seedX, int seedY, int tolerance);
Image regionGrowingCPUParallel(const Image& image, int seedX, int seedY, int tolerance, int threads);
class RegionGrowingGPU {
public:
    RegionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance, bool localTiles = true);
    ~RegionGrowingGPU();
    RegionGrowingGPU(const RegionGrowingGPU&) = delete;
    RegionGrowingGPU& operator=(const RegionGrowingGPU&) = delete;
    void run(const Image& image, Image& output, double* kernelUs = nullptr);
    int iterations() const { return iterationCount; }

private:
    void release();
    void runPixelSteps(double* kernelUs);
    void runTiles(double* kernelUs);
    void launch(cl_kernel kernel, cl_uint dimensions, const std::size_t* global, const std::size_t* local, double* kernelUs);
    void addKernelTimes(double* kernelUs);
    OpenCL cl;
    bool localTiles;
    cl_kernel initKernel = nullptr;
    cl_kernel stepKernel = nullptr;
    cl_kernel activeInitKernel = nullptr;
    cl_kernel tileKernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem maskBufferA = nullptr;
    cl_mem maskBufferB = nullptr;
    cl_mem activeBufferA = nullptr;
    cl_mem activeBufferB = nullptr;
    cl_mem changedBuffer = nullptr;
    cl_mem resultMask = nullptr;
    std::vector<cl_event> kernelEvents;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
    std::size_t tileGlobalWorkSize[2];
    std::size_t tileLocalWorkSize[2];
    std::size_t tileCount;
    int inputWidth;
    int inputHeight;
    int tilesX;
    int tilesY;
    int seedIndex;
    int iterationCount = 0;
};
