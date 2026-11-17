#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "negative.hpp"
#include "opencl_utils.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
Image randomImage(int width, int height, std::mt19937& random) {
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);
    for (auto& pixel : image.pixels) pixel = static_cast<unsigned char>(random());
    image.pixels.front() = 0;
    image.pixels.back() = 255;
    return image;
}

void checkNegative(const Image& image, const Image& result, const std::string& name) {
    if (result.width != image.width || result.height != image.height || result.pixels.size() != image.pixels.size()) {
        throw std::runtime_error(name + ": wrong output size");
    }
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        if (result.pixels[i] != 255 - image.pixels[i]) {
            throw std::runtime_error(name + ": pixel " + std::to_string(i) + " is " + std::to_string(result.pixels[i]) +
                                     " instead of " + std::to_string(255 - image.pixels[i]));
        }
    }
}

void checkRandomImages(int threads) {
    std::mt19937 random(2024);
    for (auto [width, height] : {std::pair{1, 1}, {2, 1}, {1, 7}, {3, 5}, {257, 129}, {1000, 662}}) {
        Image image = randomImage(width, height, random);
        std::string size = std::to_string(width) + "x" + std::to_string(height);
        checkNegative(image, negative(image), "negative " + size);
        Image parallel = image;
        negativeParallel(image, parallel, threads);
        checkNegative(image, parallel, "negativeParallel " + size);
        Image gpuResult = image;
        NegativeGPU gpu(image);
        gpu.run(image, gpuResult);
        checkNegative(image, gpuResult, "NegativeGPU " + size);
    }
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png", "results/negative_scaling.csv");
        checkRandomImages(options.threads);

        Image image = loadImage(options.image);
        Image cpuResult = image;
        Image result = image;
        double cpu1 = measure([&] { negativeParallel(image, cpuResult, 1); }, options.repeats);
        checkNegative(image, cpuResult, "CPU 1 thread");
        double cpuN = measure([&] { negativeParallel(image, result, options.threads); }, options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU negative results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        NegativeGPU gpu(image);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU negative results differ");
        });
        double gpuKernel = median(kernelTimes);

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
