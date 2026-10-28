#include "image_io.hpp"
#include "thresholding.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/clic_resolution_change/clic_1024.png",
                                 "results/threshold_scaling.csv", {"--threshold"});
        int threshold = parseNumber<int>(options.parameter("--threshold", "128"));
        if (threshold < 0 || threshold > 255) throw std::invalid_argument("Threshold must be between 0 and 255");
        Image image = loadImage(options.image);
        Image expected = thresholding(image, threshold);
        Image result = image;
        auto reset = [&] { result = image; };
        auto verify = [&](int) {
            if (result.pixels != expected.pixels) throw std::runtime_error("CPU threshold results differ");
        };
        // test of 1-thread on CPU
        double cpu1 = measure([&] { ThresholdParallel(result, threshold, 1); }, options.repeats, reset, verify);
        // multi-thread CPU
        double cpuN = measure([&] { ThresholdParallel(result, threshold, options.threads); }, options.repeats, reset, verify);

        double overheadN = measureThreadOverhead(options.threads, options.repeats);
        // GPU
        // Setup
        ThresholdGPU gpu(image, threshold);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != expected.pixels) throw std::runtime_error("GPU and CPU threshold results differ");
        });
        double gpuKernel = median(kernelTimes);

        // Save everything to a CSV file
        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ','
            << image.width << ',' << image.height << ',' << threshold << ','
            << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,width,height,threshold,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
