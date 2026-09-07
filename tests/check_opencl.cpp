#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>
#include "image_io.hpp"
#include <fstream>
#include <sstream>
#include <string>

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

            // here we now add a kernel

            std::ifstream kernelFile("kernels/threshold.cl");
            if (!kernelFile) {
                std::cerr << "Kernel file not found!\n";
                return 1;
            }
            std::stringstream kernelStream;
            kernelStream << kernelFile.rdbuf();
            std::string kernelSource = kernelStream.str();

            const char* source = kernelSource.c_str();
                                                        // OpenCL environment, 1 source code fragment, kernel text address, OpenCL will set the text length, error code
            cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Program creation failed: " << error << '\n';
                return 1;
            }
            error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // OpenCL compile this program to the device - mac
            if (error != CL_SUCCESS) {
                std::cerr << "Program build failed because of syntax error in threshold.cl: " << error << '\n';
                return 1;
            }

            std::cout << "Program build successfully!\n";

            cl_kernel kernel = clCreateKernel(program, "threshold_kernel", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Kernel creation failed: " << error << '\n';
                return 1;
            }
            std::cout << "Kernel created successfully!\n";

            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer); // which kernel do I set an argument for, argument number, argument size, argument content address
            if (error != CL_SUCCESS) {
                std::cerr << "Kernel set argument failed: " << error << '\n';
                return 1;
            }
            std::cout << "Kernel set argument successfully!\n";

            int threshold = 128;

            error = clSetKernelArg(kernel, 1, sizeof(int), &threshold);

            if (error != CL_SUCCESS) {
                std::cerr << "Threshold argument failed: " << error << '\n';
                return 1;
            }

            std::size_t globalWorkSize = image.pixels.size();

            error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);

            if (error != CL_SUCCESS) {
                std::cerr << "Kernel launch failed: " << error << '\n';
                return 1;
            }

            std::cout << "Kernel launched successfully!\n";

            Image gpuResult = image;
            error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize, gpuResult.pixels.data(), 0, nullptr, nullptr); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done

            if (error != CL_SUCCESS) {
                std::cerr << "Buffer read failed: " << error << '\n';
                return 1;
            }
            std::cout << "Buffer read successfully!\n";
            Image cpuResult = thresholding(image);

            if (cpuResult.pixels == gpuResult.pixels) {
                std::cout << "GPU and CPU results are identical!" << std::endl;
            }
            else {
                std::cerr << "GPU and CPU results are not identical!" << std::endl;
            }
            saveImage("results/clic_threshold_gpu.png", gpuResult);







            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseMemObject(buffer);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }
    return found ? 0 : 1;
}
