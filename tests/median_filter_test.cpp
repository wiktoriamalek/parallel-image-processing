#include "median_filter.hpp"
#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/nuclear_noisy.png", "results/median_filter_scaling.csv", {"--sort"});
        std::string sort = options.parameter("--sort", "standard");
        if (sort != "standard" && sort != "network") throw std::invalid_argument("--sort must be standard or network");
        int sortingNetwork = sort == "network";
        Image image = loadImage(options.image);
        Image result = image; // borders are not filtered, so both results start as a copy of the image
        Image parallelResult = image;
        Image referenceResult = image; // always std::nth_element, so the sorting network is checked against it
        medianFilterParallel(image, referenceResult, 1, false);

        double cpu1 = measure([&] { medianFilterParallel(image, result, 1, sortingNetwork); }, options.repeats);
        if (result.pixels != referenceResult.pixels) {
            throw std::runtime_error("Standard and network results differ!");
        }
        double cpuN = measure([&] { medianFilterParallel(image, parallelResult, options.threads, sortingNetwork); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != parallelResult.pixels) {
                throw std::runtime_error("CPU 1T and CPU many threads result differ!");
            }
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        MedianFilterGPU gpu(image, sortingNetwork);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (result.pixels != gpuResult.pixels) {
                throw std::runtime_error("CPU and GPU results differ!");
            }
        });
        double gpuKernel = median(kernelTimes);

        //saveImage("results/nuclear_original_1t.png", result);
        std::filesystem::create_directories("results/median_filter");
        saveImage("results/median_filter/" + std::filesystem::path(options.image).stem().string() + "_median.png", result);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << sort << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,sort,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << sort << " sort, median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
