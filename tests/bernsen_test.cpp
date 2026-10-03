#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "bernsen.hpp"
#include <iostream>
#include "opencl_utils.hpp"
#include <vector>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_512.png","results/bernsen_scaling.csv",{"--radius", "--contrast"});

    int radius = parseNumber<int>(options.parameter("--radius", "7"));
    int minContrast = parseNumber<int>(options.parameter("--contrast", "15"));

    Image image = loadImage(options.image);
    Image cpuResult = image;
    Image result = image;

    double cpu1 = measure([&] { // lambda can use through a reference variables mentioned!
        bernsenThreshold(image, cpuResult, radius, minContrast, 1);
    }, options.repeats);

    double cpuN = measure([&] {bernsenThreshold(image, result, radius, minContrast, options.threads);}, options.repeats);

    if (result.pixels != cpuResult.pixels) {
        throw std::runtime_error("CPU Bernsen results differ");
    }

    std::cout << "CPU 1 thread: " << cpu1 << " us = "<< cpu1/1e3<< "ms\n";
    std::cout << "CPU " << options.threads << " threads: " << cpuN << " us = "<< cpuN/1e3<< "ms\n";
    saveImage( "results/bernsen_4t_rep3_rad15_cont15_result.png", result);

    OpenCL cl("kernels/bernsen.cl");
    cl_int error;
    std::size_t size = image.pixels.size();
    cl_mem inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, size, nullptr, &error);
    checkCL(error, "clCreateBuffer");

    cl_mem outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, size, nullptr, &error);
    checkCL(error, "clCreateBuffer");

    cl_kernel kernel = clCreateKernel(cl.program, "bernsen", &error);
    checkCL(error, "clCreateKernel");

    checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 4, sizeof(radius), &radius), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 5, sizeof(minContrast), &minContrast), "clSetKernelArg");

    std::size_t workSize[2] = {
        static_cast<std::size_t>(image.width),
        static_cast<std::size_t>(image.height)
    };
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] {
            checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, size, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, workSize, nullptr, 0, nullptr, &event), "clEnqueueNDRangeKernel");
            checkCL(clFinish(cl.queue), "clFinish");
            checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, size, result.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
        }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU local thresholding results differ");
        });
        clReleaseKernel(kernel);
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);

        double gpuKernel = median(kernelTimes);

        std::cout << "GPU total: " << gpuTotal << " us = " << gpuTotal / 1e3 << " ms\n";
        std::cout << "GPU kernel: " << gpuKernel << " us = " << gpuKernel / 1e3 << " ms\n";
        return 0;
    }

    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
