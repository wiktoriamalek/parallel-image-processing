#include "region_growing.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>
#include <queue>
#include <cstdlib>
#include <algorithm>
#include <utility>

Image regionGrowingCPU(const Image& image, int seedX, int seedY, int tolerance)
{
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("regionGrowingCPU: invalid arguments");
    }

    Image mask;
    mask.width = image.width;
    mask.height = image.height;
    mask.pixels.resize((std::size_t)mask.width * mask.height);
    int seedIndex = seedY * image.width + seedX;
    int seedValue = image.pixels[seedIndex];
    mask.pixels[seedIndex] = 255;
    std::queue<int> pending;
    pending.push(seedIndex);
    int offsetX[4] = {-1 ,1, 0, 0};
    int offsetY[4] = { 0, 0, -1, 1};

    while (!pending.empty())
    {   // queue is a list of pixels that we yet need to check
        int currentIndex = pending.front(); // pixel A
        pending.pop(); // delete
        int currentX = currentIndex % image.width; // column
        int currentY = currentIndex / image.width; // row
        // if pixel B matched the spot, we want to add it to the queue and check the neighbours again
        for (int direction = 0; direction < 4; direction++)
        {
            int neighbourX = currentX + offsetX[direction];
            int neighbourY = currentY + offsetY[direction];
            if(neighbourX < 0 || neighbourX>=image.width || neighbourY < 0 || neighbourY>=image.height){continue;}
            int neighbourIndex = neighbourY * image.width + neighbourX;
            if (mask.pixels[neighbourIndex] != 0){continue;}
            int neighbourValue = image.pixels[neighbourIndex];
            int difference = std::abs(neighbourValue - seedValue);
            if (difference <= tolerance)
            {
                mask.pixels[neighbourIndex] = 255;
                pending.push(neighbourIndex);
            }
        }
    }

return mask;
}


RegionGrowingGPU::RegionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance)
    : cl("kernels/region_growing.cl"),
      bufferSize(image.pixels.size() * sizeof(unsigned char)),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)},
      inputWidth(image.width),
      inputHeight(image.height)
{
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("regionGrowingCPU: invalid arguments");
    }
    seedIndex = seedY * image.width + seedX;
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        maskBufferA = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        maskBufferB = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        changedBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");

        initKernel = clCreateKernel(cl.program, "region_init", &error);
        checkCL(error, "clCreateKernel");
        stepKernel = clCreateKernel(cl.program, "region_step", &error);
        checkCL(error, "clCreateKernel");

        checkCL(clSetKernelArg(initKernel, 0, sizeof(cl_mem), &maskBufferA), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 1, sizeof(int), &inputWidth), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 2, sizeof(int), &inputHeight), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 3, sizeof(int), &seedX), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 4, sizeof(int), &seedY), "clSetKernelArg");

        checkCL(clSetKernelArg(stepKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 1, sizeof(cl_mem), &maskBufferA), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 2, sizeof(cl_mem), &maskBufferB), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 3, sizeof(cl_mem), &changedBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 4, sizeof(int), &inputWidth), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 5, sizeof(int), &inputHeight), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 7, sizeof(int), &tolerance), "clSetKernelArg");

    } catch (...) {
        release();
        throw;
    }
}

void RegionGrowingGPU::release() {
    if (initKernel) clReleaseKernel(initKernel);
    if (stepKernel) clReleaseKernel(stepKernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (maskBufferA) clReleaseMemObject(maskBufferA);
    if (maskBufferB) clReleaseMemObject(maskBufferB);
    if (changedBuffer) clReleaseMemObject(changedBuffer);

}

RegionGrowingGPU::~RegionGrowingGPU() {
    release();
}

void RegionGrowingGPU::run(const Image& image, Image& output) {
    if (static_cast<std::size_t>(output.width) != globalWorkSize[0] || static_cast<std::size_t>(output.height) != globalWorkSize[1] ||
        image.width != inputWidth || image.height != inputHeight ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    int seedValue = image.pixels[seedIndex];
    checkCL(clSetKernelArg(stepKernel, 6, sizeof(int), &seedValue), "clSetKernelArg");

    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, initKernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");

    cl_mem previousMask = maskBufferA;
    cl_mem nextMask = maskBufferB;
    cl_int changed = 1;

    while (changed!=0)
    {
        changed = 0;
        checkCL(clEnqueueWriteBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        checkCL(clSetKernelArg(stepKernel, 1, sizeof(cl_mem), &previousMask), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 2, sizeof(cl_mem), &nextMask), "clSetKernelArg");
        checkCL(clEnqueueNDRangeKernel(cl.queue, stepKernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        checkCL(clEnqueueReadBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        std::swap(previousMask, nextMask);
    }

    checkCL(clEnqueueReadBuffer(cl.queue, previousMask, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}

Image regionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance) {
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("regionGrowingCPU: invalid arguments");
    }
    Image output;
    output.width = image.width;
    output.height = image.height;
    output.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    RegionGrowingGPU gpu(image, seedX, seedY, tolerance);
    gpu.run(image, output);
    return output;
}