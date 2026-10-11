#include "median_filter.hpp"
#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/nuclear_noisy.png", "results/median_filter_scaling.csv");
        Image image = loadImage(options.image);
        Image result = image; // borders are not filtered, so both results start as a copy of the image
        Image parallelResult = image;
        double cpu1 = measure([&] { medianFilterParallel(image, result, 1); }, options.repeats);
        double cpuN = measure([&] { medianFilterParallel(image, parallelResult, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != parallelResult.pixels) {
                throw std::runtime_error("CPU 1T and CPU many threads result differ!");
            }
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        OpenCL cl("kernels/median_filter.cl");
        std::size_t size = image.pixels.size();
        cl_int error;
        cl_mem inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, size, nullptr, &error);
        checkCL(error, "clCreateBuffer input");
        cl_mem outputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, size, nullptr, &error);
        checkCL(error, "clCreateBuffer output");
        cl_kernel kernel = clCreateKernel(cl.program, "median_filter3x3", &error);
        checkCL(error, "clCreateKernel");
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer); checkCL(error, "clSetKernelArg");
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer); checkCL(error, "clSetKernelArg");
        error = clSetKernelArg(kernel, 2, sizeof(cl_int), &image.width); checkCL(error, "clSetKernelArg");
        error = clSetKernelArg(kernel, 3, sizeof(cl_int), &image.height); checkCL(error, "clSetKernelArg");
        std::size_t workSize[2] = {
            static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)
        };
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] {
            error = clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, size, image.pixels.data(), 0, nullptr, nullptr); // 0 means we dont wait for other events
            checkCL(error, "clEnqueueWriteBuffer");            // image.pixels.data() is a data address on CPU
            error = clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, workSize, nullptr, 0, nullptr, &event); // this function enqueues the execution of a kernel on an OpenCL device
            checkCL(error, "clEnqueueNDRangeKernel");
            error = clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, size, gpuResult.pixels.data(), 0, nullptr, nullptr);
            checkCL(error, "clEnqueueReadBuffer");
        }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (result.pixels != gpuResult.pixels) {
                throw std::runtime_error("CPU and GPU results differ!");
            }
        });
        clReleaseKernel(kernel);
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        double gpuKernel = median(kernelTimes);

        //saveImage("results/nuclear_original_1t.png", result);
        std::filesystem::create_directories("results/median_filter");
        saveImage("results/median_filter/" + std::filesystem::path(options.image).stem().string() + "_median.png", result);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
