#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>
#include <vector>

constexpr int gaborMaxRadius = 32;
constexpr int gaborMaxOrientations = 16;
constexpr int gaborGroupSize = 16;
constexpr int gaborPixelsPerItem = 8;
constexpr int gaborColumnBlock = 1024;

struct GaborBank {
    int radius = 0;
    int orientations = 0;
    std::vector<float> weights;
    int size() const { return 2 * radius + 1; }
    int taps() const { return size() * size(); }
};

int gaborRadius(float sigma, float gamma);
GaborBank gaborBank(float sigma, float wavelength, float gamma, int orientations);
std::size_t gaborTileBytes(int radius);

Image gaborFilter(const Image& image, const GaborBank& bank);
void gaborFilter(const Image& image, const GaborBank& bank, Image& result, int threads = 1);
void gaborRows(const Image& image, const GaborBank& bank, Image& result, int firstRow, int endRow);

class GaborGPU {
public:
    GaborGPU(const Image& image, const GaborBank& bank);
    ~GaborGPU();
    GaborGPU(const GaborGPU&) = delete;
    GaborGPU& operator=(const GaborGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    cl_mem weightsBuffer = nullptr;
    std::size_t bufferSize;
    std::size_t globalWorkSize[2];
};
