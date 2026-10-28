#include "bernsen.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <stdexcept>

void bernsenRows(const Image &image, Image &result, int radius, int minContrast, int firstRow, int endRow) {
    for (int y=firstRow; y<endRow; y++) {
        for (int x=0; x<image.width; x++) {
            int left = std::max(x- radius, 0);
            int right = std::min(image.width-1, x+radius);
            int top = std::max(y-radius, 0);
            int bottom = std::min(image.height-1,y+radius);
            int Jmin  = 255; // because we're going to be searching by std::min and update the value
            int Jmax  = 0;
            for (int ny=top; ny<=bottom; ny++) {
                for (int nx=left; nx<=right; nx++) {
                    std::size_t index = ny*image.width + nx;
                    int value = image.pixels[index];
                    Jmin = std::min(Jmin, value);
                    Jmax = std::max(Jmax, value);
                }
            }
            int contrast = Jmax - Jmin;
            if (contrast < minContrast) {
                result.pixels[y*image.width + x] = 0;
            }
            else {
                int threshold = (Jmax + Jmin)/2;
                int pixel = image.pixels[y*image.width + x];
                //result.pixels[y*image.width + x] = pixel < threshold ? 0 : 255; // Andrea Latina speaking....
                if (pixel < threshold) {
                    result.pixels[y*image.width + x] = 0;
                }

                else {
                    result.pixels[y*image.width + x] = 255;
                }
            }
        }
    }
}

void bernsenThreshold(const Image &image, Image &result, int radius, int minContrast, int threads) {
    // we need to divide the rows for threads and for each range call bernsenRows

    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        bernsenRows(image, result, radius, minContrast, static_cast<int>(begin), static_cast<int>(end));
    });
}

BernsenGPU::BernsenGPU(const Image& image, int radius, int minContrast)
    : cl("kernels/bernsen.cl"),
      bufferSize(image.pixels.size()),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "bernsen", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(radius), &radius), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(minContrast), &minContrast), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void BernsenGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

BernsenGPU::~BernsenGPU() {
    release();
}

void BernsenGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(image.width) != globalWorkSize[0] || static_cast<std::size_t>(image.height) != globalWorkSize[1] ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}
