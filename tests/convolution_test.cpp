#include "image_io.hpp"
#include "convolution.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png",
                                 "results/convolution_scaling.csv", {"--filter"});
        std::string filterName = options.parameter("--filter", "sharpen");
        const ConvolutionFilter& selected = convolutionFilter(filterName);
        const auto& filter = selected.kernel;
        bool absoluteValue = selected.absoluteValue;

        Image image = loadImage(options.image);
        Image cpuResult = image;
        Image result = image;
        double cpu1 = measure([&] { convolution(image, filter, absoluteValue, cpuResult); }, options.repeats);
        // multi-thread CPU
        double cpuN = measure([&] { convolution(image, filter, absoluteValue, result, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU convolution results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        // GPU
        // Setup
        ConvolutionGPU gpu(image, filter, absoluteValue);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU convolution results differ");
        });
        double gpuKernel = median(kernelTimes);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << filterName << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,filter,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
