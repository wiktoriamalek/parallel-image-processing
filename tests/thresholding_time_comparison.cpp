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

int main() {
    try {
        Image image = loadImage("images/resolution_change/clic_8192.png");
        Image divided = image;

        // test of 1-thread on CPU
        auto start = std::chrono::high_resolution_clock::now();
        Image singleCPUthread = thresholding(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        std::cout <<"Time taken on single-thread CPU: " << duration.count() <<"us" << std::endl;

        // multi-thread CPU
        auto start2 = std::chrono::high_resolution_clock::now();
        std::size_t Npixels = divided.pixels.size();
        std::size_t middle = Npixels / 2;
        std::thread t1(ThresholdRange, std::ref(divided), 128, 0, middle);
        std::thread t2(ThresholdRange, std::ref(divided), 128, middle, Npixels);
        t1.join();
        t2.join();
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2);
        std::cout <<"Time taken on 2-threads on CPU: " << duration2.count() <<"us" << std::endl;

        // GPU
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
                cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
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
                auto gpuTotalStart = std::chrono::high_resolution_clock::now();
                error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr);
                auto kernelStart = std::chrono::high_resolution_clock::now();
                error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
                error = clFinish(queue);
                auto kernelEnd = std::chrono::high_resolution_clock::now();
                error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize, gpuResult.pixels.data(), 0, nullptr, nullptr); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
                auto gpuTotalEnd = std::chrono::high_resolution_clock::now();
                clReleaseKernel(kernel);
                clReleaseProgram(program);
                clReleaseMemObject(buffer);
                clReleaseCommandQueue(queue);
                clReleaseContext(context);
                auto gpuTotalDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTotalEnd - gpuTotalStart);
                auto kernelDuration = std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - kernelStart);
                std::cout << "GPU total (from writing to reading after kernel operations): " << gpuTotalDuration.count() << " us\n";
                std::cout << "GPU kernel only: "<< kernelDuration.count() << " us\n";

            }
        }
        return 0;
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}