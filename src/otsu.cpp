#include "otsu.hpp"
#include <thread>
#include <vector>
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

    return bestThreshold;
}

Image otsuSegmentation(const Image& image) {
    Image result = image;
    Histogram histogram = calculateHistogram(image);
    int threshold = calculateOtsuThreshold(histogram);

    for (unsigned char& pixel : result.pixels) {
        pixel = pixel <= threshold ? 0 : 255;
    }

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
    Image result = image;
    Histogram histogram = calculateHistogram2Threads(image);
    int threshold = calculateOtsuThreshold(histogram);

    std::size_t Npixels = image.pixels.size();
    std::size_t middle = Npixels / 2;

    std::thread t1(applyThresholdRange, std::ref(result), threshold, 0, middle);
    std::thread t2(applyThresholdRange, std::ref(result), threshold, middle, Npixels);
    t1.join();
    t2.join();
    return result;
}

void checkCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

Histogram calculateHistogramGPU(const Image& image) {
    Histogram result{};

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
            cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
            checkCL(error, "clCreateCommandQueue");
            std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);
            std::array<unsigned int, 256> gpuHistogram{};
            cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
            checkCL(error, "clCreateBuffer");
            cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(unsigned int), nullptr, &error);
            checkCL(error, "clCreateBuffer");
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

            checkCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkCL(clEnqueueWriteBuffer(queue, histogramBuffer, CL_TRUE, 0, 256 * sizeof(unsigned int), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
            checkCL(clFinish(queue), "clFinish");
            checkCL(clEnqueueReadBuffer(queue, histogramBuffer, CL_TRUE, 0, 256 * sizeof(unsigned int), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done

            std::copy(gpuHistogram.begin(), gpuHistogram.end(), result.begin());
            checkCL(clReleaseKernel(kernel), "clReleaseKernel");
            checkCL(clReleaseProgram(program), "clReleaseProgram");
            checkCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
            checkCL(clReleaseMemObject(histogramBuffer), "clReleaseMemObject");
            checkCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
            checkCL(clReleaseContext(context), "clReleaseContext");
        }

    }

    return result;
}

Image applyThresholdGPU(const Image& image, int threshold) {
    Image result = image;
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
            cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
            checkCL(error, "clCreateCommandQueue");
            std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);
            cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
            checkCL(error, "clCreateBuffer");
            cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
            checkCL(error, "clCreateBuffer");
            std::ifstream kernelFile("kernels/otsu.cl");
            if (!kernelFile) throw std::runtime_error("Could not open kernels/otsu.cl");
            std::stringstream kernelStream;
            kernelStream << kernelFile.rdbuf();
            std::string kernelSource = kernelStream.str();

            const char* source = kernelSource.c_str();
            cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
            checkCL(error, "clCreateProgramWithSource");
            checkCL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram"); // OpenCL compile this program to the device - mac
            cl_kernel kernel = clCreateKernel(program, "thresholdKernel", &error);
            checkCL(error, "clCreateKernel");
            int Npixels = image.pixels.size();
            checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(kernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
            checkCL(clSetKernelArg(kernel, 3, sizeof(int), &threshold), "clSetKernelArg");
            std::size_t globalWorkSize = {static_cast<std::size_t>(image.pixels.size())};

            checkCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
            checkCL(clFinish(queue), "clFinish");
            checkCL(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, result.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done

            checkCL(clReleaseKernel(kernel), "clReleaseKernel");
            checkCL(clReleaseProgram(program), "clReleaseProgram");
            checkCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
            checkCL(clReleaseMemObject(outputBuffer), "clReleaseMemObject");
            checkCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
            checkCL(clReleaseContext(context), "clReleaseContext");
        }
    }
    return result;
}

Image otsuSegmentationGPU(const Image& image) {
    Histogram histogram = calculateHistogramGPU(image);
    int threshold = calculateOtsuThreshold(histogram);
    return applyThresholdGPU(image, threshold);
}
