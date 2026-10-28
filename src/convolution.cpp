#include "convolution.hpp"
#include "parallel_utils.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>

Image convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue) {
    Image result = image;
    convolution(image, kernel, absoluteValue, result);
    return result;
}

void convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int threads) {
    parallelRanges(std::max(0, image.height - 2), threads, [&](std::size_t begin, std::size_t end, int) {
        convolutionRows(image, kernel, absoluteValue, result, static_cast<int>(begin) + 1, static_cast<int>(end) + 1);
    });
}

void convolutionRows(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int firstRow, int endRow) {

    // dont start from edges
    // y = 1...height-2
    // x = 1 ... width -2

    for (int y=firstRow; y<endRow; y++) {
        for (int x=1; x<image.width-1; x++) {
            float sum = 0.0f;
            // even though images are 2D, pixels are stored as one big 1D array
            // 0 1 2 3 4
            // 5 6 7 8 9 1
            // 0 11 12 13 14 - pixels, 3 rows 5 numbers each
            // index = y * width + x
            // index = 1 * 5 + 2 = 7, so pixel(2, 1) = 7
            // kx = -1, ky = 0 => 1 pixel left, the same row
            for (int ky=-1; ky<=1; ky++) {
                for (int kx=-1; kx<=1; kx++) {
                    int neighourX = x + kx;
                    int neighourY = y + ky;
                    std::size_t neighbourIndex = neighourY * image.width + neighourX;
                    // neighbour value * kernel weight
                    float neighbourValue = image.pixels[neighbourIndex];
                    sum += neighbourValue * kernel[ky+1][kx+1];
                }
            }
            std::size_t resultIndex = y * image.width + x;
            if (absoluteValue) {
                sum = std::abs(sum);
            }

            sum = std::clamp(sum, 0.0f, 255.0f);
            result.pixels[resultIndex] = static_cast<unsigned char>(sum);
        }
    }

}

ConvolutionGPU::ConvolutionGPU(const Image& image, const std::array<std::array<float, 3>, 3>& filter, bool absoluteValue)
    : cl("kernels/convolution.cl"),
      bufferSize(image.pixels.size()),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    std::array<float, 9> gpuFilter;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) gpuFilter[y * 3 + x] = filter[y][x];
    }
    int absolute = absoluteValue;
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        filterBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(gpuFilter), gpuFilter.data(), &error); // create a memory on GPU and copy 9 elements there
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "convolution3x3", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(filterBuffer), &filterBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(absolute), &absolute), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void ConvolutionGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
    if (filterBuffer) clReleaseMemObject(filterBuffer);
}

ConvolutionGPU::~ConvolutionGPU() {
    release();
}

void ConvolutionGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(image.width) != globalWorkSize[0] || static_cast<std::size_t>(image.height) != globalWorkSize[1] ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}
