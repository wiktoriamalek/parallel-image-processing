#include "thresholding.hpp"
#include <cstddef>
#include <stdexcept>
#include "parallel_utils.hpp"

Image thresholding(const Image& image, int threshold)
{
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("Image is empty");
    }

    std::size_t expectedCount = static_cast<std::size_t>(image.width) * image.height;
    if (image.pixels.size() != expectedCount) {
        throw std::runtime_error("Image size mismatch");
    }

    Image result = image; // we are modifying the copy
    ThresholdRange(result, threshold, 0, result.pixels.size());
    return result;
}

void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max)
{
    for (std::size_t i = min; i < max; ++i) {
        if (image.pixels[i] < threshold) {
            image.pixels[i] = 0;
        }
        else {
            image.pixels[i] = 255;
        }
    }
}

void ThresholdParallel(Image& image, int threshold, int numThreads) {
    parallelRanges(image.pixels.size(), numThreads, [&](std::size_t begin, std::size_t end, int) {
        ThresholdRange(image, threshold, begin, end);
    });
}

ThresholdGPU::ThresholdGPU(const Image& image, int threshold)
    : cl("kernels/threshold.cl"), bufferSize(image.pixels.size()) {
    try {
        cl_int error;
        buffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error); // nullptr because now the buffor doesnt have pixels, its just for memory reservation
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "threshold_kernel", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(buffer), &buffer), "clSetKernelArg"); // which kernel do I set an argument for, argument number, argument size, argument content address
        checkCL(clSetKernelArg(kernel, 1, sizeof(threshold), &threshold), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void ThresholdGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (buffer) clReleaseMemObject(buffer);
}

ThresholdGPU::~ThresholdGPU() {
    release();
}

void ThresholdGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image size does not match the GPU buffer");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, buffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 1, nullptr, &bufferSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, buffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
