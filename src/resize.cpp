#include "resize.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

Image resizeNearestCPU(const Image &image, int newWidth, int newHeight, int threads)
{
    if (newWidth <= 0 || newHeight <= 0 || image.width <= 0 || image.height <= 0)
    {
       throw std::invalid_argument("Invalid image size");
    }
    Image resizedImage;
    resizedImage.width = newWidth;
    resizedImage.height = newHeight;
    resizedImage.pixels.resize(static_cast<std::size_t>(newWidth) * newHeight);

    parallelRanges(resizedImage.height, threads, [&](std::size_t begin, std::size_t end, int) {
        resizeNearestRange(image, resizedImage, begin, end);});
    return resizedImage;
}

void resizeBilinearRange(const Image &image,Image& output, int startY, int endY)
{
    if (output.width <= 0 || output.height <= 0 || image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("Invalid image size");
    }
    float scaleX = (float)image.width/output.width;
    float scaleY = (float)image.height/output.height;

    for (int y = startY; y < endY; y++)
    {
        for (int x = 0; x < output.width; x++)
        {
            float sourcePixelX = (x+0.5f)*scaleX-0.5f;
            float sourcePixelY = (y+0.5f)*scaleY-0.5f;
            sourcePixelX = std::clamp(sourcePixelX, 0.0f, static_cast<float>(image.width - 1));
            sourcePixelY = std::clamp(sourcePixelY, 0.0f, static_cast<float>(image.height - 1));
            int destinationIndex = y * output.width + x;
            int x0 = std::floor(sourcePixelX);
            int y0 = std::floor(sourcePixelY);
            int x1 = std::min(x0+1, image.width-1);
            int y1 = std::min(y0+1, image.height-1);
            float fx = sourcePixelX - x0;
            float fy = sourcePixelY - y0;
            float p00 = image.pixels[y0*image.width+x0];
            float p10 = image.pixels[y0*image.width+x1];
            float p01 = image.pixels[y1*image.width+x0];
            float p11 = image.pixels[y1*image.width+x1];
            float top = (1-fx)*p00+fx*p10;
            float bottom = (1-fx)*p01+fx*p11;
            float value = static_cast<unsigned char>(std::lround((1-fy)*top+fy*bottom));
            output.pixels[destinationIndex] = value;
        }
    }
}


Image resizeBilinearCPU(const Image &image, int newWidth, int newHeight, int threads)
{
    if (newWidth <= 0 || newHeight <= 0 || image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("Invalid image size");
    }
    Image resizedImage;
    resizedImage.width = newWidth;
    resizedImage.height = newHeight;
    resizedImage.pixels.resize(static_cast<std::size_t>(newWidth) * newHeight);
    parallelRanges(resizedImage.height, threads, [&](std::size_t begin, std::size_t end, int) {
        resizeBilinearRange(image, resizedImage, begin, end);});

    return resizedImage;
}

void resizeNearestRange(const Image &image, Image& output, int startY, int endY)
{
    float scaleX = (float)image.width/output.width;
    float scaleY = (float)image.height/output.height;

    for (int y = startY; y < endY; y++)
    {
        for (int x = 0; x < output.width; x++)
        {
            float sourcePixelX = (x+0.5f)*scaleX-0.5f;
            float sourcePixelY = (y+0.5f)*scaleY-0.5f;
            int sourceX = std::lround(sourcePixelX);
            int sourceY = std::lround(sourcePixelY);
            int destinationIndex = y * output.width + x;

            sourceX = std::clamp(sourceX, 0, image.width - 1);
            sourceY = std::clamp(sourceY, 0, image.height - 1);
            int sourceIndex = sourceY * image.width + sourceX;
            output.pixels[destinationIndex] = image.pixels[sourceIndex];
        }
    }
}

ResizeGPU::ResizeGPU(const Image& image, int newWidth, int newHeight, bool bilinear)
    : cl("kernels/resize.cl"),
      inputBufferSize(image.pixels.size() * sizeof(unsigned char)),
      outputBufferSize(static_cast<std::size_t>(newWidth) * newHeight * sizeof(unsigned char)),
      globalWorkSize{static_cast<std::size_t>(newWidth), static_cast<std::size_t>(newHeight)},
      inputWidth(image.width),
      inputHeight(image.height)
{
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, inputBufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, outputBufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, bilinear ?"resize_bilinear":"resize_nearest", &error); // value ? true:false
        checkCL(error, "clCreateKernel");

        checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(int), &inputWidth), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(int), &inputHeight), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(int), &newWidth), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(int), &newHeight), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void ResizeGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

ResizeGPU::~ResizeGPU() {
    release();
}

void ResizeGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(output.width) != globalWorkSize[0] || static_cast<std::size_t>(output.height) != globalWorkSize[1] ||
        image.width != inputWidth || image.height != inputHeight ||
        image.pixels.size() != inputBufferSize || output.pixels.size() != outputBufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, inputBufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, outputBufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}

Image resizeGPU(const Image& image, int newWidth, int newHeight, bool bilinear) {
    if (newWidth <= 0 || newHeight <= 0 || image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("Invalid image size");
    }
    Image output;
    output.width = newWidth;
    output.height = newHeight;
    output.pixels.resize((std::size_t)newWidth * newHeight);
    ResizeGPU gpu(image, newWidth, newHeight, bilinear);
    gpu.run(image, output);
    return output;
}