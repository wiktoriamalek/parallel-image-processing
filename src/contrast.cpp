#include "contrast.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Image contrast(const Image& image, float factor)
{
    Image contrast_image = image;
    contrastRange(image, contrast_image, 0, image.pixels.size(), factor);
    return contrast_image;
}

void contrastRange(const Image& image, Image& output, std::size_t begin, std::size_t end, float factor)
{
    for (std::size_t i = begin; i < end; ++i) {
        float scaled = (image.pixels[i] - 128) * factor;
        float value = scaled + 128.0f;
        output.pixels[i] = static_cast<unsigned char>(std::lround(std::clamp(value, 0.0f, 255.0f)));
    }
}

void contrastParallel(const Image& image, Image& output, int threads, float factor)
{
    if (output.pixels.size() != image.pixels.size()) {
        throw std::runtime_error("Output image size does not match the input");
    }
    parallelRanges(image.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        contrastRange(image, output, begin, end, factor);
    });
}

ContrastGPU::ContrastGPU(const Image& image, float factor)
    : cl("kernels/contrast.cl"), bufferSize(image.pixels.size()) {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "contrast_gpu", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(factor), &factor), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void ContrastGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

ContrastGPU::~ContrastGPU() {
    release();
}

void ContrastGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 1, nullptr, &bufferSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
