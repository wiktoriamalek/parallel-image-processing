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

struct BrownConradyCoefficients {
    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
};

Image distortBrownConrady(const Image& image, const BrownConradyCoefficients& coefficients, int threads = 1);
void correctBrownConradyRange(const Image& image, Image& output, const BrownConradyCoefficients& coefficients, int startY, int endY);
void correctBrownConradyCPU(const Image& image, const BrownConradyCoefficients& coefficients, Image& output, int threads = 1);
Image correctBrownConradyCPU(const Image& image, const BrownConradyCoefficients& coefficients, int threads = 1);

class BrownConradyGPU {
public:
    BrownConradyGPU(const Image& image, const BrownConradyCoefficients& coefficients);
    ~BrownConradyGPU();
    BrownConradyGPU(const BrownConradyGPU&) = delete;
    BrownConradyGPU& operator=(const BrownConradyGPU&) = delete;
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
