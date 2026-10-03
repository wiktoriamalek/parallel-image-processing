#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <iomanip>
#include <iostream>
#include <map>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png",
                                 "results/convolution_scaling.csv", {"--filter"});
        const std::map<std::string, std::array<std::array<float, 3>, 3>> filters = {
            {"identity", {{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}}},
            // box blur 3x3 - it takes for each pixel, 9 values from the surrounding pixels and calculates mean
            {"box_blur", {{{1.0f/9, 1.0f/9, 1.0f/9}, {1.0f/9, 1.0f/9, 1.0f/9}, {1.0f/9, 1.0f/9, 1.0f/9}}}},
            {"sharpen", {{{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}}},
            {"edge_detection_h", {{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}}},
            {"edge_detection_v", {{{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}}}
        };
        std::string filterName = options.parameter("--filter", "sharpen");
        auto selected = filters.find(filterName);
        if (selected == filters.end()) throw std::invalid_argument("Unknown filter: " + filterName);
        const auto& filter = selected->second;
        int absoluteValue = filterName == "edge_detection_h" || filterName == "edge_detection_v";
        std::array<float, 9> gpuFilter;
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) gpuFilter[y * 3 + x] = filter[y][x];
        }

        Image image = loadImage(options.image);
        Image cpuResult = image;
        Image result = image;
        double cpu1 = measure([&] { convolution(image, filter, absoluteValue, cpuResult); }, options.repeats);
        // multi-thread CPU
        double cpuN = measure([&] { convolution(image, filter, absoluteValue, result, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU convolution results differ");
        });

        // GPU
        // Setup
        OpenCL cl("kernels/convolution.cl");
        cl_int error;
        std::size_t size = image.pixels.size();
        cl_mem inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, size, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        cl_mem outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, size, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        cl_mem filterBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(gpuFilter), gpuFilter.data(), &error); // create a memory on GPU and copy 9 elements there
        checkCL(error, "clCreateBuffer");
        cl_kernel kernel = clCreateKernel(cl.program, "convolution3x3", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(filterBuffer), &filterBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(absoluteValue), &absoluteValue), "clSetKernelArg");
        std::size_t workSize[2] = {static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)};
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
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU convolution results differ");
        });
        clReleaseKernel(kernel);
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(filterBuffer);
        double gpuKernel = median(kernelTimes);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << filterName << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,filter,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
