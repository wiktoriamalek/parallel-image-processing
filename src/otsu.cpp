#include "otsu.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional> // so we can use std::ref
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <stdexcept>
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <array>

double median(std::vector<long long> values) {
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n%2 == 1) {
        return values[n/2];
    }
    return (values[n/2 - 1] + values[n/2])/2.0;
}

void calculateHistogramRange(const Image& image, Histogram& histogram, std::size_t begin, std::size_t end) {
    for (std::size_t i=begin; i<end; i++) {
        int pixel = image.pixels[i];
        ++histogram[pixel];
    }
}

Histogram calculateHistogram(const Image& image) {
    Histogram histogram{}; // resets all 256 counters to zero.

    for (unsigned char pixel : image.pixels) {
        ++histogram[pixel]; // if pixel is e.g 50, we increase histogram[50]
    }
    return histogram;
}

Histogram calculateHistogram2Threads(const Image& image) {
    Histogram histogram1{};
    Histogram histogram2{};
    std::size_t Npixels = image.pixels.size();
    std::size_t middle = Npixels / 2;

    std::thread t1(calculateHistogramRange, std::ref(image), std::ref(histogram1), 0, middle);
    std::thread t2(calculateHistogramRange, std::ref(image), std::ref(histogram2), middle, Npixels);
    t1.join();
    t2.join();

    Histogram histogram{};

    for (int i=0; i<256; i++) {
        histogram[i] = histogram1[i] + histogram2[i];
    }

    return histogram;

}

int calculateOtsuThreshold(const Histogram& histogram) {
    std::size_t totalPixels = 0;
    double totalSum = 0.0;

    std::size_t backgroundPixels = 0; // number of pixels with brightness <= current threshold t
    double backgroundSum = 0.0; // brightness sum of those pixels

    for (int i =0; i<256; i++) {
        totalPixels += histogram[i];
        totalSum += i * histogram[i];
    }

    double maxVariance = 0.0;
    int bestThreshold = 0;
    for (int t =0; t<256; t++) {
        backgroundPixels += histogram[t];
        backgroundSum += t * histogram[t];

        std::size_t foregroundPixels = totalPixels - backgroundPixels;
        double foregroundSum = totalSum - backgroundSum;

        if (foregroundPixels == 0 || backgroundPixels == 0) {
            continue;
        }

        double foregroundAverage = foregroundSum / foregroundPixels;
        double backgroundAverage = backgroundSum / backgroundPixels;

        double backgroundWeight = static_cast<double>(backgroundPixels) / totalPixels;
        double foregroundWeight = static_cast<double>(foregroundPixels) / totalPixels;
        double difference = backgroundAverage - foregroundAverage;
        double betweenClassVariance = backgroundWeight * foregroundWeight * (difference * difference);
        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            bestThreshold = t;
        }
    }

    // std::cout << "Total pixels: " << totalPixels << '\n';
    // std::cout << "Total sum: " << totalSum << '\n';
    // std::cout << "Best threshold: " << bestThreshold << '\n';
    // std::cout << "Maximum variance: " << maxVariance << '\n';

    return bestThreshold;
}

Image otsuSegmentation(const Image& image) {
    std::vector<long long> cpu1Times;
    Image result = image;

    for (int repeat=0; repeat<100; repeat++) {
        auto start = std::chrono::high_resolution_clock::now();
        result = image;

        Histogram histogram = calculateHistogram(image);
        int threshold = calculateOtsuThreshold(histogram);

        for (unsigned char& pixel : result.pixels) {
            pixel = pixel <= threshold ? 0:255;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        cpu1Times.push_back(duration.count());
    }
    double medianCPU1 = median(cpu1Times);
    std::cout <<"Median time taken on single-thread CPU: " << medianCPU1 <<"us" << std::endl;


    return result;
}
                        // not const, because we will be modifying it
void applyThresholdRange(Image& image, int threshold, std::size_t begin, std::size_t end) {
    for (std::size_t i=begin; i<end; i++) {
        if (image.pixels[i] <= threshold)
        {
            image.pixels[i] = 0;
        }
        else
        {
            image.pixels[i] = 255;
        }
    }
}

Image otsuSegmentation2Threads(const Image& image) {
    std::vector<long long> cpu2Times;
    Image result = image;

    for (int repeat=0; repeat<100; repeat++) {
        auto start2 = std::chrono::high_resolution_clock::now();
        result = image;

        Histogram histogram = calculateHistogram2Threads(image);
        int threshold = calculateOtsuThreshold(histogram);

        std::size_t Npixels = image.pixels.size();
        std::size_t middle = Npixels / 2;

        std::thread t1(applyThresholdRange, std::ref(result), threshold, 0, middle);
        std::thread t2(applyThresholdRange, std::ref(result), threshold, middle, Npixels);
        t1.join();
        t2.join();
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2);
        cpu2Times.push_back(duration2.count());
    }
    double medianCPU2 = median(cpu2Times);
    std::cout <<"Median time taken on 2-threads on CPU: " << medianCPU2 <<"us" << std::endl;
    return result;
}

void checkCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

Histogram calculateHistogramGPU(const Image& image) {
        Histogram result{};
        std::vector<long long> gpuKernelOnly;
        std::vector<long long> gpuAll;

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
                std::array<unsigned int, 256> gpuHistogram{};
                cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
                checkCL(error, "clCreateBuffer");
                cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(unsigned int), nullptr, &error);
                checkCL(error, "clCreateBuffer");
                                                                                // create a memory on GPU and copy 9 elements there
                std::ifstream kernelFile("kernels/otsu.cl");
                if (!kernelFile) throw std::runtime_error("Could not open kernels/otsu.cl");
                std::stringstream kernelStream;
                kernelStream << kernelFile.rdbuf();
                std::string kernelSource = kernelStream.str();

                const char* source = kernelSource.c_str();
                cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
                checkCL(error, "clCreateProgramWithSource");
                checkCL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram"); // OpenCL compile this program to the device - mac
                cl_kernel kernel = clCreateKernel(program, "histogramKernel", &error);
                checkCL(error, "clCreateKernel");
                int Npixels = image.pixels.size();
                checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 2, sizeof(int), &Npixels), "clSetKernelArg");

                std::size_t globalWorkSize = {static_cast<std::size_t>(image.pixels.size())};

            for (int repeat=0; repeat<100; repeat++) {
                cl_event kernelEvent;
                gpuHistogram.fill(0);
                auto gpuTotalStart = std::chrono::high_resolution_clock::now();
                checkCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
                checkCL(clEnqueueWriteBuffer(queue, histogramBuffer, CL_TRUE, 0, 256 * sizeof(unsigned int), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
                checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, &kernelEvent), "clEnqueueNDRangeKernel");
                checkCL(clFinish(queue), "clFinish");
                checkCL(clEnqueueReadBuffer(queue, histogramBuffer, CL_TRUE, 0, 256 * sizeof(unsigned int), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done

                auto gpuTotalEnd = std::chrono::high_resolution_clock::now();
                auto gpuTotalDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTotalEnd - gpuTotalStart);
                cl_ulong kernelStart, kernelEnd;
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, nullptr), "clGetEventProfilingInfo");
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, nullptr), "clGetEventProfilingInfo");
                checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
                gpuAll.push_back(gpuTotalDuration.count());
                gpuKernelOnly.push_back(kernelEnd - kernelStart);


            }
                std::copy(gpuHistogram.begin(), gpuHistogram.end(), result.begin());
            checkCL(clReleaseKernel(kernel), "clReleaseKernel");
            checkCL(clReleaseProgram(program), "clReleaseProgram");
            checkCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
                checkCL(clReleaseMemObject(histogramBuffer), "clReleaseMemObject");
                checkCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
            checkCL(clReleaseContext(context), "clReleaseContext");
        }

    }

        double medianGPUAll = median(gpuAll);
        double medianGPUKernel = median(gpuKernelOnly) / 1000.0;

        std::cout << "Median GPU total (from writing to reading after kernel operations): " << medianGPUAll << " us\n";
        std::cout << "Median GPU kernel only: "<< medianGPUKernel << " us\n";
return result;
}