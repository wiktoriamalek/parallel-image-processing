#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "bernsen.hpp"
#include <filesystem>
#include <iomanip>
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
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        if (result.pixels != cpuResult.pixels) {
            throw std::runtime_error("CPU Bernsen results differ");
        }

        std::cout << "CPU 1 thread: " << cpu1 << " us = "<< cpu1/1e3<< "ms\n";
        std::cout << "CPU " << options.threads << " threads: " << cpuN << " us = "<< cpuN/1e3<< "ms\n";
        std::filesystem::create_directories("results/bernsen");
        saveImage("results/bernsen/" + std::filesystem::path(options.image).stem().string() + "_radius" +
                  std::to_string(radius) + "_contrast" + std::to_string(minContrast) + ".png", result);

        BernsenGPU gpu(image, radius, minContrast);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU local thresholding results differ");
        });

        double gpuKernel = median(kernelTimes);

        std::cout << "GPU total: " << gpuTotal << " us = " << gpuTotal / 1e3 << " ms\n";
        std::cout << "GPU kernel: " << gpuKernel << " us = " << gpuKernel / 1e3 << " ms\n";

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << radius << ',' << minContrast << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,radius,contrast,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        return 0;
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
