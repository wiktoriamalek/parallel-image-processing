#include <iostream>
#include "image_io.hpp"
#include <stdexcept>
#include <thread>
#include <functional>
#include <chrono>
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

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
        Image image = loadImage("images/clic_resolution_change/clic_1024.png");
        int resolution = 1024;

        // test of 1-thread on CPU
        std::vector<long long> cpu1Times;

        for (int repeat=0; repeat<100; repeat++) {
            Image singleCPUthread = image;
            std::size_t Npixels = singleCPUthread.pixels.size();
            auto start = std::chrono::high_resolution_clock::now();
            ThresholdRange(singleCPUthread, 128, 0, Npixels);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
            cpu1Times.push_back(duration.count());
        }

        double medianCPU1 = median(cpu1Times);
        std::cout <<"Median time taken on single-thread CPU: " << medianCPU1 <<"us" << std::endl;

        // multi-thread CPU
        std::vector<long long> cpu2Times;
        for (int repeat=0; repeat<100; repeat++) {
            Image divided = image;
            std::size_t Npixels = divided.pixels.size();
            std::size_t middle = Npixels / 2;
            auto start2 = std::chrono::high_resolution_clock::now();
            std::thread t1(ThresholdRange, std::ref(divided), 128, 0, middle);
            std::thread t2(ThresholdRange, std::ref(divided), 128, middle, Npixels);
            t1.join();
            t2.join();
            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2);
            cpu2Times.push_back(duration2.count());
        }

        double medianCPU2 = median(cpu2Times);
        std::cout <<"Median time taken on 2-threads on CPU: " << medianCPU2 <<"us" << std::endl;

        // GPU
        std::vector<long long> gpuKernelOnly;
        std::vector<long long> gpuAll;

        // Setup
        cl_uint platformCount = 0;
        cl_int error = clGetPlatformIDs(0, nullptr, &platformCount);
        std::vector<cl_platform_id> platforms(platformCount);
        error = clGetPlatformIDs(platformCount, platforms.data(), nullptr);

        for (auto platform : platforms) {
            cl_uint count = 0;
            error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
            std::vector<cl_device_id> devices(count);
            error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);
            for (auto device : devices) {
                char name[256] = {};
                error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
                cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
                std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);
                cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error); // nullptr because now the buffor doesnt have pixels, its just for memory reservation
                std::ifstream kernelFile("kernels/threshold.cl");
                std::stringstream kernelStream;
                kernelStream << kernelFile.rdbuf();
                std::string kernelSource = kernelStream.str();
                const char* source = kernelSource.c_str();
                cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
                error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // OpenCL compile this program to the device - mac
                cl_kernel kernel = clCreateKernel(program, "threshold_kernel", &error);
                error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer); // which kernel do I set an argument for, argument number, argument size, argument content address
                int threshold = 128;
                error = clSetKernelArg(kernel, 1, sizeof(int), &threshold);
                std::size_t globalWorkSize = image.pixels.size();
                Image gpuResult = image;

                for (int repeat = 0; repeat < 100; ++repeat) {
                    cl_event kernelEvent = nullptr;

                    auto gpuTotalStart = std::chrono::steady_clock::now();
                    auto gpuTotalEnd = std::chrono::steady_clock::now();

                    cl_ulong kernelStart = 0;
                    cl_ulong kernelEnd = 0;

                    auto totalDuration =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            gpuTotalEnd - gpuTotalStart
                        );

                    gpuAll.push_back(totalDuration.count());
                    gpuKernelOnly.push_back(kernelEnd - kernelStart);
                }
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseMemObject(buffer);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        }

        double medianGPUAll = median(gpuAll);
        double medianGPUKernel = median(gpuKernelOnly) / 1000.0;

        std::cout << "Median GPU total (from writing to reading after kernel operations): " << medianGPUAll << " us\n";
        std::cout << "Median GPU kernel only: "<< medianGPUKernel << " us\n";


        // Save everything to a CSV file
        std::ofstream file("results/threshold_test_16384.csv", std::ios::app);
        file << resolution << ","
             << image.pixels.size() << ","
             << medianCPU1 << ","
             << medianCPU2 << ","
             << medianGPUKernel << ","
             << medianGPUAll << "\n";


        return 0;
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}