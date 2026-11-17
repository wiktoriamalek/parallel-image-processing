#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>

Image contrast(const Image& image);
void contrastRange(const Image& image, Image& output, std::size_t begin, std::size_t end);
void contrastParallel(const Image& image, Image& output, int threads);

class ContrastGPU {
public:
    explicit ContrastGPU(const Image& image);
    ~ContrastGPU();
    ContrastGPU(const ContrastGPU&) = delete;
    ContrastGPU& operator=(const ContrastGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
};
