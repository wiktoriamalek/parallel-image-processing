#include "bernsen.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Method {
    std::string name;
    LocalThresholdMethod gpuMethod;
    void (*cpu)(const Image&, Image&, int, double, int);
    bool (*isBlack)(std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel, int hundredthsK);
};

const std::vector<Method>& methods() {
    static const std::vector<Method> all = {
        {"niblack", LocalThresholdMethod::Niblack, niblackThreshold, niblackIsBlack},
        {"sauvola", LocalThresholdMethod::Sauvola, sauvolaThreshold, sauvolaIsBlack},
    };
    return all;
}

bool windowIsBlack(const Image& image, int x, int y, int radius, const Method& method, double k) {
    const int left = x - std::min(radius, x);
    const int right = x + std::min(radius, image.width - 1 - x);
    const int top = y - std::min(radius, y);
    const int bottom = y + std::min(radius, image.height - 1 - y);
    std::int64_t sum = 0;
    std::int64_t sumOfSquares = 0;
    for (int ny = top; ny <= bottom; ++ny) {
        for (int nx = left; nx <= right; ++nx) {
            const std::int64_t value = image.pixels[static_cast<std::size_t>(ny) * image.width + nx];
            sum += value;
            sumOfSquares += value * value;
        }
    }
    const std::int64_t count = static_cast<std::int64_t>(right - left + 1) * (bottom - top + 1);
    const int pixel = image.pixels[static_cast<std::size_t>(y) * image.width + x];
    return method.isBlack(count, sum, sumOfSquares, pixel, localThresholdHundredths(k));
}

Image naive(const Image& image, int radius, const Method& method, double k) {
    Image result = image;
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const auto index = static_cast<std::size_t>(y) * image.width + x;
            result.pixels[index] = windowIsBlack(image, x, y, radius, method, k) ? 0 : 255;
        }
    }
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

std::string describe(const Method& method, const Image& image, int radius, double k) {
    std::ostringstream text;
    text << method.name << ' ' << image.width << 'x' << image.height << ", radius " << radius << ", k " << k;
    return text.str();
}

long long checkGPU(const Image& image, const Image& cpu, const Image& gpu, int radius, const Method& method, double k) {
    long long different = 0;
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        if (cpu.pixels[i] == gpu.pixels[i]) continue;
        const int x = static_cast<int>(i % image.width);
        const int y = static_cast<int>(i / image.width);
        std::ostringstream message;
        message << describe(method, image, radius, k) << ": GPU differs from CPU at (" << x << ", " << y << ") with pixel "
                << int(image.pixels[i]);
        throw std::runtime_error(message.str());
    }
    return different;
}

void checkRandomImages(int threads) {
    std::mt19937 generator(2030);
    long long different = 0;
    long long checked = 0;
    for (auto [width, height] : {std::pair{1, 1}, {1, 7}, {7, 1}, {3, 5}, {64, 64}, {300, 129}}) {
        const Image image = randomImage(width, height, generator);
        for (const Method& method : methods()) {
            for (int radius : {0, 1, 2, 7, 20, 50}) {
                for (double k : {0.0, 0.2, 0.5}) {
                    const std::string name = describe(method, image, radius, k);
                    const Image expected = naive(image, radius, method, k);
                    Image result = image;
                    method.cpu(image, result, radius, k, 1);
                    if (result.pixels != expected.pixels) throw std::runtime_error(name + ": CPU 1 thread differs from the naive version");
                    method.cpu(image, result, radius, k, threads);
                    if (result.pixels != expected.pixels) throw std::runtime_error(name + ": CPU with threads differs from the naive version");
                    LocalThresholdGPU gpu(image, method.gpuMethod, radius, k);
                    Image gpuResult = image;
                    gpu.run(image, gpuResult);
                    different += checkGPU(image, expected, gpuResult, radius, method, k);
                    checked += static_cast<long long>(image.pixels.size());
                }
            }
        }
    }

    for (int level : {0, 100, 255}) {
        const Image flat{40, 30, std::vector<unsigned char>(40 * 30, static_cast<unsigned char>(level))};
        for (const Method& method : methods()) {
            for (double k : {0.0, 0.2, 0.5}) {
                Image result = flat;
                method.cpu(flat, result, 3, k, threads);
                LocalThresholdGPU gpu(flat, method.gpuMethod, 3, k);
                Image gpuResult = flat;
                gpu.run(flat, gpuResult);
                const bool allWhite = std::all_of(result.pixels.begin(), result.pixels.end(), [](unsigned char p) { return p == 255; });
                if (!allWhite || gpuResult.pixels != result.pixels) {
                    throw std::runtime_error(describe(method, flat, 3, k) + ": a flat image is not all white");
                }
            }
        }
    }

    bool thrown = false;
    try {
        LocalThresholdGPU gpu(randomImage(8, 8, generator), LocalThresholdMethod::Niblack, 128, 0.2);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    if (!thrown) throw std::runtime_error("GPU accepted a radius it cannot handle");

    std::cout << "CPU 1 thread and " << threads << " threads match the naive version on random and flat images "
              << "(radius 0-50, k 0-0.5); GPU matches them in all " << checked << " pixels (" << different
              << " different)\n";
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png",
                                 "results/niblack_sauvola_scaling.csv", {"--radius", "--k"});
        const int radius = parseNumber<int>(options.parameter("--radius", "7"));
        const double k = parseNumber<double>(options.parameter("--k", "0.2"));
        checkRandomImages(options.threads);

        const Image image = loadImage(options.image);
        const std::string imageName = std::filesystem::path(options.image).stem().string();
        const std::filesystem::path outputDirectory = "results/niblack_sauvola";
        std::filesystem::create_directories(outputDirectory);
        for (const Method& method : methods()) {
            Image cpuResult = image;
            Image result = image;
            const double cpu1 = measure([&] { method.cpu(image, cpuResult, radius, k, 1); }, options.repeats);
            const double cpuN = measure([&] { method.cpu(image, result, radius, k, options.threads); }, options.repeats,
                                        [] {}, [&](int) {
                if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU results differ");
            });
            const double overheadN = measureThreadOverhead(options.threads, options.repeats);

            LocalThresholdGPU gpu(image, method.gpuMethod, radius, k);
            Image gpuResult = image;
            cl_event event = nullptr;
            std::vector<double> kernelTimes;
            const double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
                const double time = kernelTimeUs(event);
                checkCL(clReleaseEvent(event), "clReleaseEvent");
                if (repeat >= 0) kernelTimes.push_back(time);
            });
            const double gpuKernel = median(kernelTimes);
            const long long different = checkGPU(image, cpuResult, gpuResult, radius, method, k);

            saveImage((outputDirectory / (imageName + "_" + method.name + "_r" + std::to_string(radius) + ".png")).string(), cpuResult);
            std::ostringstream row;
            row << std::setprecision(12) << method.name << ',' << std::quoted(options.image, '"', '"') << ','
                << image.width << ',' << image.height << ',' << radius << ',' << k << ',' << options.threads << ','
                << options.repeats << ',' << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ','
                << gpuTotal << ',' << different;
            appendCsv(options.output, "method,image,width,height,radius,k,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,gpu_different_pixels", row.str());
            std::cout << method.name << ": median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                      << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << ", GPU differs in "
                      << different << " pixels\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
