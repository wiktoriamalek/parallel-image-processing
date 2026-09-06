#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>
#include "image_io.hpp"

int main() {
    Image image = loadImage("images/clic_image.png");
    cl_uint platformCount = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &platformCount);
    if (error != CL_SUCCESS || platformCount == 0) {
        std::cerr << "Nie znaleziono platformy OpenCL. Kod: " << error << '\n';
        return 1;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    error = clGetPlatformIDs(platformCount, platforms.data(), nullptr); // what platforms are on this computer
    if (error != CL_SUCCESS) return 1;
    bool found = false;

    for (auto platform : platforms) {
        cl_uint count = 0;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count); // on a chosen platfrom we are searching for a device - GPU
        if (error != CL_SUCCESS) {
            std::cerr << "Zapytanie o GPU: kod " << error << '\n';
            continue;
        }
        std::vector<cl_device_id> devices(count);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);
        if (error != CL_SUCCESS) return 1;
        for (auto device : devices) {
            char name[256] = {};
            error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr); // information about a found device
            if (error != CL_SUCCESS) return 1;
            std::cout << "GPU wykryte przez OpenCL: " << name << '\n';
            found = true;
            cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);

            if (error != CL_SUCCESS) {
                std::cerr << "Context not found: " << error << '\n';
                return 1;
            }

            std::cout << "Context created successfully!\n";

            cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);

            if (error != CL_SUCCESS) {
                std::cerr << "Command queue failed: " << error << '\n';
                return 1;
            }
            std::cout << "Command queue created successfully!\n";

            std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);

            cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error); // nullptr because now the buffor doesnt have pixels, its just for memory reservation

            if (error != CL_SUCCESS) {
                std::cerr << "Buffer creation failed: " << error << '\n';
                return 1;
            }
            std::cout << "Buffer created successfully!\n";
            error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr);

            if (error != CL_SUCCESS) {
                std::cerr << "Buffer write failed: " << error << '\n';
                return 1;
            }
            std::cout << "Buffer write successfully!\n";
            clReleaseMemObject(buffer);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }
    return found ? 0 : 1;
}
