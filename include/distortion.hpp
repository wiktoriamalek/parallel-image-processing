#pragma once
#include "image_io.hpp"
#include "opencl_utils.hpp"

Image correctRadialDistortionCPU(const Image& image, float k1, float k2, int threads = 1);
void correctRadialDistortionCPU(const Image& image, float k1, float k2, Image& output, int threads = 1);
Image distortRadial(const Image& image, float k1, float k2);
void correctRadialDistortionRange(const Image& image, Image& output, float k1, float k2, int startY, int endY);
Image correctRadialDistortionGPU(const Image& image, float k1, float k2);

class RadialDistortionGPU {
public:
    RadialDistortionGPU(const Image& image, float k1, float k2);
    ~RadialDistortionGPU();
    RadialDistortionGPU(const RadialDistortionGPU&) = delete;
    RadialDistortionGPU& operator=(const RadialDistortionGPU&) = delete;
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
