#include "distortion.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

Image distortRadial(const Image& image, float k1, float k2) {
    Image output = image;
    float centerX = (image.width - 1)/2.0f;
    float centerY = (image.height - 1)/2.0f;
    float scale = std::max(image.width, image.height) / 2.0f;

    for (int pixelX = 0; pixelX < image.width; pixelX++) {
        for (int pixelY = 0; pixelY < image.height; pixelY++) {
            float normalizedX = (pixelX - centerX) / scale;
            float normalizedY = (pixelY - centerY) / scale;
            float distortedRadius = std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY); // rd
            float lowerRadius = 0.0f;
            float upperRadius = distortedRadius;
            for (int iteration = 0; iteration < 25; iteration++) {
                float middleRadius = (lowerRadius + upperRadius)/2.0f;
                float ru = middleRadius;
                float r2 = middleRadius*middleRadius; // ru^2
                float calculatedRadius = ru * (1 + k1 * r2 + k2 * r2 * r2);
                if (calculatedRadius > distortedRadius) {
                    upperRadius = middleRadius;
                }
                else if (calculatedRadius < distortedRadius) {
                    lowerRadius = middleRadius;
                }

            }
            float undistortedRadius = (lowerRadius + upperRadius)/2.0f;
            float radiusRatio;
            if (distortedRadius == 0.0f) {
                radiusRatio = 1.0f;
            }
            else {
                radiusRatio = undistortedRadius/distortedRadius;
            }

            float sourceNormalizedX = normalizedX * radiusRatio;
            float sourceNormalizedY = normalizedY * radiusRatio;

            float sourceRelativeX = sourceNormalizedX * scale; // original picture, undistorted, relative to the center of the image
            float sourceRelativeY = sourceNormalizedY * scale;

            float sourcePixelX = sourceRelativeX + centerX;
            float sourcePixelY = sourceRelativeY + centerY;

            int sourceX = std::lround(sourcePixelX);
            int sourceY = std::lround(sourcePixelY);

            int destinationIndex = pixelY * image.width + pixelX;
            if (sourceX < 0 || image.width <= sourceX || sourceY < 0 || image.height <= sourceY) {
                output.pixels[destinationIndex] = 0;
            }

            else {
                int sourceIndex = sourceY * image.width + sourceX;
                output.pixels[destinationIndex] = image.pixels[sourceIndex];
            }
        }
    }
    return output;
}

void correctRadialDistortionCPU(const Image& image, float k1, float k2, Image& output, int threads) {
    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        correctRadialDistortionRange(image, output, k1, k2, begin, end);
    });
}

Image correctRadialDistortionCPU(const Image& image, float k1, float k2, int threads) {
    Image output = image;
    correctRadialDistortionCPU(image, k1, k2, output, threads);
    return output;
}

void correctRadialDistortionRange(const Image &image, Image &output, float k1, float k2, int startY, int endY) {
    float centerX = (image.width - 1)/2.0f;
    float centerY = (image.height - 1)/2.0f;
    float scale = std::max(image.width, image.height) / 2.0f;

    for (int pixelY = startY; pixelY < endY; pixelY++) {
        for (int pixelX = 0; pixelX < image.width; pixelX++) {
            float relativeX = pixelX - centerX;
            float relativeY = pixelY - centerY;

            float normalizedX = relativeX / scale;
            float normalizedY = relativeY / scale;

            float r2 = normalizedX*normalizedX + normalizedY*normalizedY; // squared distance from the center
            float factor = 1.0f + k1 * r2 + k2 * r2 * r2;

            float distortedX = normalizedX * factor; // radial distortion
            float distortedY = normalizedY * factor; // radial distortion

            float distortedRelativeX = distortedX * scale ;
            float distortedRelativeY = distortedY * scale ;

            float distortedPixelX = distortedRelativeX + centerX;
            float distortedPixelY = distortedRelativeY + centerY;

            // int sourceX = std::lround(distortedPixelX);
            // int sourceY = std::lround(distortedPixelY);

            int x0 = std::floor(distortedPixelX);
            int y0 = std::floor(distortedPixelY);
            int x1 = x0 +1;
            int y1 = y0 +1;
            float fx = distortedPixelX - x0;
            float fy = distortedPixelY - y0;
            int destinationIndex = pixelY * image.width + pixelX;

            if (distortedPixelX < 0 || image.width - 1 < distortedPixelX || distortedPixelY < 0 || image.height -1 < distortedPixelY) {
                output.pixels[destinationIndex] = 0;
            }

            else {
                x1 = std::min(image.width - 1, x1);
                y1 = std::min(image.height - 1, y1);
                // int sourceIndex = y1 * image.width + x1;
                // output.pixels[destinationIndex] = image.pixels[sourceIndex];
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
}

RadialDistortionGPU::RadialDistortionGPU(const Image& image, float k1, float k2)
    : cl("kernels/distortion_correction.cl"),
      bufferSize(image.pixels.size() * sizeof(unsigned char)),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    // Setup
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "distortion_correction", &error);
        checkCL(error, "clCreateKernel");

        checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(int), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(int), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(float), &k1), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(float), &k2), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void RadialDistortionGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

RadialDistortionGPU::~RadialDistortionGPU() {
    release();
}

void RadialDistortionGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(image.width) != globalWorkSize[0] || static_cast<std::size_t>(image.height) != globalWorkSize[1] ||
        output.width != image.width || output.height != image.height ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}

Image correctRadialDistortionGPU(const Image& image, float k1, float k2) {
    Image output = image;
    RadialDistortionGPU gpu(image, k1, k2);
    gpu.run(image, output);
    return output;
}
