#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>

Image negative(const Image& image);
void negativeRange(const Image& image, Image& output, std::size_t begin, std::size_t end);
void negativeParallel(const Image& image, Image& output, int threads);

class NegativeGPU {
public:
    explicit NegativeGPU(const Image& image);
    ~NegativeGPU();
    NegativeGPU(const NegativeGPU&) = delete;
    NegativeGPU& operator=(const NegativeGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
};
