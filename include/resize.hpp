#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"

Image resizeNearestCPU(const Image &image, int newWidth, int newHeight, int threads = 1);
Image resizeBilinearCPU(const Image &image, int newWidth, int newHeight, int threads = 1);
void resizeNearestRange(const Image &image, Image& output, int startY, int endY);
void resizeBilinearRange(const Image &image, Image& output, int startY, int endY);
Image resizeGPU(const Image& image, int newWidth, int newHeight, bool bilinear);

class ResizeGPU {
public:
    ResizeGPU(const Image& image, int newWidth, int newHeight, bool bilinear);
    ~ResizeGPU();
    ResizeGPU(const ResizeGPU&) = delete;
    ResizeGPU& operator=(const ResizeGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t inputBufferSize;
    std::size_t outputBufferSize;
    std::size_t globalWorkSize[2];
    int inputWidth;
    int inputHeight;
};
