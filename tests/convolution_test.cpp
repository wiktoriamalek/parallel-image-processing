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

void checkCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

double median(std::vector<long long> values) {
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n%2 == 1) {
        return values[n/2];
    }
    return (values[n/2 - 1] + values[n/2])/2.0;
}

int main() {
    try {
        std::array<std::array<float, 3>, 3> identity = {
            {
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 0},
            }};

        std::array<std::array<float, 3>, 3> boxBlur = {
            {
                // box blur 3x3 - it takes for each pixel, 9 values from the surrounding pixels and calculates mean
                {1.0f/9, 1.0f/9, 1.0f/9},
                {1.0f/9, 1.0f/9, 1.0f/9},
                {1.0f/9, 1.0f/9, 1.0f/9},
            }};

        std::array<std::array<float, 3>, 3> edgeDetectionH = {
            {
                {-1, -1, -1},
                {0,   0,  0},
                {1,   1,  1},
            }};

        std::array<std::array<float, 3>, 3> edgeDetectionV = {
            {
                {-1,  0,  1},
                {-1,  0,  1},
                {-1,  0,  1},
            }};

        std::array<std::array<float, 3>, 3> sharpen = {
            {
                {0,  -1,  0},
                {-1,  5, -1},
                {0,  -1,  0},
            }};

        std::array<float, 9> gpuFilter;
        for (int y=0; y<3;y++) {
            for (int x=0; x<3;x++) {
                gpuFilter[y*3 + x] = edgeDetectionV[y][x];
            }
        }


        Image image = loadImage("images/cms.png");
        Image convoluted = image;
        std::vector<long long> cpu1Times;

        for (int repeat=0; repeat<100; repeat++) {
            auto start = std::chrono::high_resolution_clock::now();
            convolution(image, edgeDetectionV, true, convoluted);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
            cpu1Times.push_back(duration.count());
            // if (convoluted.pixels != image.pixels) {
            //     throw std::runtime_error("CPU identity convolution differs from input");
            // }
        }
        double medianCPU1 = median(cpu1Times);
        std::cout <<"Median time taken on single-thread CPU: " << medianCPU1 <<"us" << std::endl;

        // 2 - thread CPU
        std::vector<long long> cpu2Times;
        Image divided = image;
        int middle = image.height/2;
        for (int repeat=0; repeat<100; repeat++) {
            auto start2 = std::chrono::high_resolution_clock::now();
            std::thread t1([&]{convolutionRows(image, edgeDetectionV, true, divided, 1, middle);});
            std::thread t2([&]{convolutionRows(image, edgeDetectionV, true, divided, middle, image.height-1);});
            t1.join();
            t2.join();
            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2);
            cpu2Times.push_back(duration2.count());
            if (divided.pixels != convoluted.pixels) {
                throw std::runtime_error("CPU results differ");
            }
        }

        double medianCPU2 = median(cpu2Times);
        std::cout <<"Median time taken on 2-threads on CPU: " << medianCPU2 <<"us" << std::endl;

        //saveImage("results/cms_identity.png", identical);

        Image blurred = convolution(image, boxBlur, false);
        //saveImage("results/cms_blurred.png", blurred);

        Image sharpened = convolution(image, sharpen, false);
        //saveImage("results/cms_sharpened.png", sharpened);

        Image edgeDetectedH = convolution(image, edgeDetectionH, true);
        //saveImage("results/cms_edgeDetectionH.png", edgeDetectedH);

        Image edgeDetectedV = convolution(image, edgeDetectionV, true);
        //saveImage("results/cms_edgeDetectionV.png", edgeDetectedV);

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
                                                                                // create a memory on GPU and copy 9 elements there
                cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gpuFilter.size() * sizeof(float), gpuFilter.data(), &error);
                checkCL(error, "clCreateBuffer");

                std::ifstream kernelFile("kernels/convolution.cl");
                if (!kernelFile) throw std::runtime_error("Could not open kernels/convolution.cl");
                std::stringstream kernelStream;
                kernelStream << kernelFile.rdbuf();
                std::string kernelSource = kernelStream.str();

                const char* source = kernelSource.c_str();
                cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
                checkCL(error, "clCreateProgramWithSource");
                checkCL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram"); // OpenCL compile this program to the device - mac
                cl_kernel kernel = clCreateKernel(program, "convolution3x3", &error);
                checkCL(error, "clCreateKernel");
                int width = image.width;
                int height = image.height;
                int absoluteValue = 1;

                checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 2, sizeof(int), &width), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 3, sizeof(int), &height), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &filterBuffer), "clSetKernelArg");
                checkCL(clSetKernelArg(kernel, 5, sizeof(int), &absoluteValue), "clSetKernelArg");

                std::size_t globalWorkSize[2] = {
                    static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)
                };
                Image gpuResult = image;


            for (int repeat=0; repeat<100; repeat++) {
                cl_event kernelEvent;
                auto gpuTotalStart = std::chrono::high_resolution_clock::now();
                checkCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
                checkCL(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernelEvent), "clEnqueueNDRangeKernel");
                checkCL(clFinish(queue), "clFinish");
                checkCL(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, gpuResult.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
                auto gpuTotalEnd = std::chrono::high_resolution_clock::now();
                auto gpuTotalDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTotalEnd - gpuTotalStart);
                cl_ulong kernelStart, kernelEnd;
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, nullptr), "clGetEventProfilingInfo");
                checkCL(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, nullptr), "clGetEventProfilingInfo");
                checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
                gpuAll.push_back(gpuTotalDuration.count());
                gpuKernelOnly.push_back(kernelEnd - kernelStart);
                if (gpuResult.pixels != convoluted.pixels) {
                    throw std::runtime_error("GPU and CPU convolution results differ");
                }

            }
                std::cout << "GPU and CPU convolution are identical!\n";
            checkCL(clReleaseKernel(kernel), "clReleaseKernel");
            checkCL(clReleaseProgram(program), "clReleaseProgram");
            checkCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
                checkCL(clReleaseMemObject(outputBuffer), "clReleaseMemObject");
                checkCL(clReleaseMemObject(filterBuffer), "clReleaseMemObject");
            checkCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
            checkCL(clReleaseContext(context), "clReleaseContext");
        }

    }

        double medianGPUAll = median(gpuAll);
        double medianGPUKernel = median(gpuKernelOnly) / 1000.0;

        std::cout << "Median GPU total (from writing to reading after kernel operations): " << medianGPUAll << " us\n";
        std::cout << "Median GPU kernel only: "<< medianGPUKernel << " us\n";


        return 0;
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
