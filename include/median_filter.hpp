#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <array>
#include <cstddef>

Image medianFilter(const Image& image);
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow, bool sortingNetwork=false); // it doesnt need to return a result, it can modify it through a reference
void medianFilterParallel(const Image& image, Image& result, int threads, bool sortingNetwork=false);
unsigned char medianOf9(std::array<unsigned char, 9>& p);

class MedianFilterGPU {
public:
    explicit MedianFilterGPU(const Image& image, bool sortingNetwork = false);
    ~MedianFilterGPU();
    MedianFilterGPU(const MedianFilterGPU&) = delete;
    MedianFilterGPU& operator=(const MedianFilterGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
};
