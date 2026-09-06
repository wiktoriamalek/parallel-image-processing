#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>

int main() {
    cl_uint platformCount = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &platformCount);
    if (error != CL_SUCCESS || platformCount == 0) {
        std::cerr << "Nie znaleziono platformy OpenCL. Kod: " << error << '\n';
        return 1;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    error = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (error != CL_SUCCESS) return 1;
    bool found = false;

    for (auto platform : platforms) {
        cl_uint count = 0;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        if (error != CL_SUCCESS) {
            std::cerr << "Zapytanie o GPU: kod " << error << '\n';
            continue;
        }
        std::vector<cl_device_id> devices(count);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);
        if (error != CL_SUCCESS) return 1;
        for (auto device : devices) {
            char name[256] = {};
            error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            if (error != CL_SUCCESS) return 1;
            std::cout << "GPU wykryte przez OpenCL: " << name << '\n';
            found = true;
        }
    }
    return found ? 0 : 1;
}
