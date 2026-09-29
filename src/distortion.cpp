#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <stdexcept>
#include <thread>
#include <chrono>
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <functional>
#include "distortion.hpp"

double median(std::vector<long long> values) {
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n%2 == 1) {
        return values[n/2];
    }
    return (values[n/2 - 1] + values[n/2])/2.0;
}


void checkCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

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

Image correctRadialDistortionCPU2(const Image& image, float k1, float k2) {

    Image output = image;
    int middleY = image.height/2;
    std::vector<long long> cpu2Times;

    for (int repeat=0; repeat<100; repeat++) {
        auto start = std::chrono::steady_clock::now();
        std::thread thread1(correctRadialDistortionRange, std::cref(image), std::ref(output), k1, k2, 0, middleY);
        std::thread thread2(correctRadialDistortionRange, std::cref(image), std::ref(output), k1, k2, middleY, image.height);

        thread1.join();
        thread2.join();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        cpu2Times.push_back(duration.count());
    }
    double medianCPU2 = median(cpu2Times);
    std::cout <<"Median time taken on 2-threads on CPU: " << medianCPU2 <<" us" << std::endl;

    return output;
}

Image correctRadialDistortionCPU1(const Image& image, float k1, float k2) {
    std::vector<long long> cpu1Times;
    Image output = image;
    for (int repeat=0; repeat<100; repeat++) {
        auto start = std::chrono::steady_clock::now();
        correctRadialDistortionRange(image, output, k1, k2, 0, image.height);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        cpu1Times.push_back(duration.count());
    }
    double medianCPU1 = median(cpu1Times);
    std::cout <<"Median time taken on single-thread CPU: " << medianCPU1 <<" us" << std::endl;

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

            int sourceX = std::lround(distortedPixelX);
            int sourceY = std::lround(distortedPixelY);

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
}

Image correctRadialDistortionGPU(const Image& image, float k1, float k2) {
        Image gpuResult = image;

        // GPU
        std::vector<long long> gpuKernelOnly;
        std::vector<long long> gpuAll;

        // Setup
        cl_uint platformCount = 0;
        cl_int error;
        checkCL(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs");
        std::vector<cl_platform_id> platforms(platformCount);
        checkCL(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs");

        for (auto platform : platforms) {
            cl_uint count = 0;
            checkCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count), "clGetDeviceIDs");
            std::vector<cl_device_id> devices(count);
            checkCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr), "clGetDeviceIDs");
            for (auto device : devices) {
                char name[256] = {};
                checkCL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo");
                cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
                checkCL(error, "clCreateContext");
                cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
                checkCL(error, "clCreateCommandQueue");
                std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);

                cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
                checkCL(error, "clCreateBuffer");
                cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
                checkCL(error, "clCreateBuffer");

                std::ifstream kernelFile("kernels/distortion_correction.cl");
                if (!kernelFile) throw std::runtime_error("Could not open kernels/distortion_correction.cl");
                std::stringstream kernelStream;
                kernelStream << kernelFile.rdbuf();
                std::string kernelSource = kernelStream.str();

                const char* source = kernelSource.c_str();
                cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
                checkCL(error, "clCreateProgramWithSource");
                checkCL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram"); // OpenCL compile this program to the device - mac
                cl_kernel kernel = clCreateKernel(program, "distortion_correction", &error);
                checkCL(error, "clCreateKernel");
                int width = image.width;
                int height = image.height;

                checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 2, sizeof(int), &height), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 3, sizeof(int), &width), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 4, sizeof(float), &k1), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 5, sizeof(float), &k2), "clSetKernelArg");

                std::size_t globalWorkSize[2] = {static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)};

            for (int repeat=0; repeat<100; repeat++) {
                cl_event kernelEvent;
                auto gpuTotalStart = std::chrono::steady_clock::now();
                checkCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
                checkCL(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernelEvent), "clEnqueueNDRangeKernel");
                checkCL(clFinish(queue), "clFinish");
                checkCL(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, gpuResult.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
                auto gpuTotalEnd = std::chrono::steady_clock::now();
                auto gpuTotalDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTotalEnd - gpuTotalStart);
                cl_ulong kernelStart, kernelEnd;
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, nullptr), "clGetEventProfilingInfo");
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, nullptr), "clGetEventProfilingInfo");
                checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
                gpuAll.push_back(gpuTotalDuration.count());
                gpuKernelOnly.push_back(kernelEnd - kernelStart);
            }
            checkCL(clReleaseKernel(kernel), "clReleaseKernel");
            checkCL(clReleaseProgram(program), "clReleaseProgram");
            checkCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
                checkCL(clReleaseMemObject(outputBuffer), "clReleaseMemObject");
            checkCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
            checkCL(clReleaseContext(context), "clReleaseContext");
        }
    }

        double medianGPUAll = median(gpuAll);
        double medianGPUKernel = median(gpuKernelOnly) / 1000.0;

        std::cout << "Median GPU total (from writing to reading after kernel operations): " << medianGPUAll << " us\n";
        std::cout << "Median GPU kernel only: "<< medianGPUKernel << " us\n";

    return gpuResult;
}
