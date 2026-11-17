#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "bernsen.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include "opencl_utils.hpp"
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
Image direct(const Image& image, int radius, int minContrast) {
    Image result = image;
    bernsenRows(image, result, radius, minContrast, 0, image.height);
    return result;
}

Image randomImage(int width, int height, std::mt19937& generator) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    std::uniform_int_distribution<int> value(0, 255);
    const int flat = value(generator);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool inFlatArea = (x / 5 + y / 7) % 3 == 0;
            image.pixels[static_cast<std::size_t>(y) * width + x] = static_cast<unsigned char>(inFlatArea ? flat : value(generator));
        }
    }
    return image;
}

void checkSame(const Image& image, int radius, int minContrast, int threads) {
    std::ostringstream name;
    name << image.width << 'x' << image.height << ", radius " << radius << ", contrast " << minContrast;
    const Image expected = direct(image, radius, minContrast);
    Image result = image;
    bernsenThreshold(image, result, radius, minContrast, 1);
    if (result.pixels != expected.pixels) throw std::runtime_error(name.str() + ": CPU 1 thread differs from the direct version");
    bernsenThreshold(image, result, radius, minContrast, threads);
    if (result.pixels != expected.pixels) throw std::runtime_error(name.str() + ": CPU with threads differs from the direct version");
    BernsenGPU gpu(image, radius, minContrast);
    Image gpuResult = image;
    gpu.run(image, gpuResult);
    if (gpuResult.pixels != expected.pixels) throw std::runtime_error(name.str() + ": GPU differs from the direct version");
}

void checkRandomImages(int threads) {
    std::mt19937 generator(2031);
    for (auto [width, height] : {std::pair{1, 1}, {1, 7}, {7, 1}, {3, 5}, {64, 64}, {300, 129}, {129, 300}}) {
        const Image image = randomImage(width, height, generator);
        for (int radius : {0, 1, 2, 7, 20, 50, 200}) {
            for (int minContrast : {0, 15, 255}) checkSame(image, radius, minContrast, threads);
        }
    }
    for (int level : {0, 100, 255}) {
        checkSame(Image{40, 30, std::vector<unsigned char>(40 * 30, static_cast<unsigned char>(level))}, 3, 15, threads);
        checkSame(Image{40, 30, std::vector<unsigned char>(40 * 30, static_cast<unsigned char>(level))}, 3, 0, threads);
    }
    std::cout << "CPU 1 thread, CPU " << threads << " threads and GPU match the direct version on random and flat images "
              << "(radius 0-200, contrast 0-255)\n";
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_512.png","results/bernsen_scaling.csv",{"--radius", "--contrast"});

        int radius = parseNumber<int>(options.parameter("--radius", "7"));
        int minContrast = parseNumber<int>(options.parameter("--contrast", "15"));
        checkRandomImages(options.threads);

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
        cl_event rowEvent = nullptr;
        cl_event columnEvent = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &rowEvent, &columnEvent); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(rowEvent) + kernelTimeUs(columnEvent);
            checkCL(clReleaseEvent(rowEvent), "clReleaseEvent");
            checkCL(clReleaseEvent(columnEvent), "clReleaseEvent");
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
