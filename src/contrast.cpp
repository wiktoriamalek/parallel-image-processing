#include "contrast.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>

Image contrast(const Image& image)
{
    Image contrast_image = image;
    contrastRange(image, contrast_image, 0, image.pixels.size());
    return contrast_image;
}

void contrastRange(const Image& image, Image& output, std::size_t begin, std::size_t end)
{
    for (std::size_t i = begin; i < end; ++i) {
        if ((image.pixels[i]-128)*2>255){output.pixels[i]=255;}
        else {output.pixels[i] = (image.pixels[i]-128)*2;}
    }
}

void contrastParallel(const Image& image, Image& output, int threads)
{
    if (output.pixels.size() != image.pixels.size()) {
        throw std::runtime_error("Output image size does not match the input");
    }
    parallelRanges(image.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        contrastRange(image, output, begin, end);
    });
}

ContrastGPU::ContrastGPU(const Image& image)
    : cl("kernels/contrast.cl"), bufferSize(image.pixels.size()) {
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
