#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "contrast.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cmath>
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

Image allLevels() {
    Image image;
    image.width = 16;
    image.height = 16;
    image.pixels.resize(256);
    for (int level = 0; level < 256; ++level) image.pixels[level] = static_cast<unsigned char>(level);
    return image;
}

void checkContrast(const Image& image, const Image& result, float factor, const std::string& name) {
    if (result.width != image.width || result.height != image.height || result.pixels.size() != image.pixels.size()) {
        throw std::runtime_error(name + ": wrong output size");
    }
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        double expected = std::clamp((image.pixels[i] - 128) * static_cast<double>(factor) + 128.0, 0.0, 255.0);
        if (std::abs(result.pixels[i] - expected) > 0.5 + 1e-4) {
            throw std::runtime_error(name + ", factor " + std::to_string(factor) + ": pixel " + std::to_string(i) + " = " +
                                     std::to_string(image.pixels[i]) + " gives " + std::to_string(result.pixels[i]) +
                                     " instead of " + std::to_string(expected));
        }
    }
}

void checkSame(const Image& expected, const Image& result, const std::string& name) {
    if (result.pixels != expected.pixels) throw std::runtime_error(name + " differs from contrast() on CPU");
}

void checkRandomImages(int threads) {
    std::mt19937 random(2025);
    for (float factor : {2.0f, 0.5f, 1.0f, 0.0f, 1.3f, 3.7f, -1.0f}) {
        std::vector<Image> images = {allLevels()};
        for (auto [width, height] : {std::pair{1, 1}, {2, 1}, {1, 7}, {3, 5}, {257, 129}, {1000, 662}}) {
            images.push_back(randomImage(width, height, random));
        }
        for (const Image& image : images) {
            std::string name = std::to_string(image.width) + "x" + std::to_string(image.height);
            Image serial = contrast(image, factor);
            checkContrast(image, serial, factor, "contrast " + name);
            Image parallel = image;
            contrastParallel(image, parallel, threads, factor);
            checkSame(serial, parallel, "contrastParallel " + name);
            Image gpuResult = image;
            ContrastGPU gpu(image, factor);
            gpu.run(image, gpuResult);
            checkSame(serial, gpuResult, "ContrastGPU " + name);
        }
    }
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png", "results/contrast_scaling.csv",
                                 {"--factor"});
        float factor = parseNumber<float>(options.parameter("--factor", "2.0"));
        checkRandomImages(options.threads);

        Image image = loadImage(options.image);
        Image cpuResult = image;
        Image result = image;
        double cpu1 = measure([&] { contrastParallel(image, cpuResult, 1, factor); }, options.repeats);
        checkContrast(image, cpuResult, factor, "CPU 1 thread");
        double cpuN = measure([&] { contrastParallel(image, result, options.threads, factor); }, options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU contrast results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        ContrastGPU gpu(image, factor);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != cpuResult.pixels) throw std::runtime_error("GPU and CPU contrast results differ");
        });
        double gpuKernel = median(kernelTimes);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << factor << ','
            << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,factor,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
