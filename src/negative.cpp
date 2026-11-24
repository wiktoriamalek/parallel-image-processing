#include "negative.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>

Image negative(const Image& image)
{
    Image negative_image = image;
    negativeRange(image, negative_image, 0, image.pixels.size());
    return negative_image;
}

void negativeRange(const Image& image, Image& output, std::size_t begin, std::size_t end)
{
    const unsigned char* input = image.pixels.data();
    unsigned char* result = output.pixels.data();
    for (std::size_t i = begin; i < end; ++i) {
        result[i] = 255 - input[i];
    }
}

void negativeParallel(const Image& image, Image& output, int threads)
{
    if (output.pixels.size() != image.pixels.size()) {
        throw std::runtime_error("Output image size does not match the input");
    }
    parallelRanges(image.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        negativeRange(image, output, begin, end);
    });
}

NegativeGPU::NegativeGPU(const Image& image)
    : cl("kernels/negative.cl"), bufferSize(image.pixels.size()) {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "negative_gpu", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        const int pixelCount = static_cast<int>(bufferSize);
        checkCL(clSetKernelArg(kernel, 2, sizeof(pixelCount), &pixelCount), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void NegativeGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

NegativeGPU::~NegativeGPU() {
    release();
}

void NegativeGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    enqueueLine(cl.queue, kernel, bufferSize, kernelEvent);
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
