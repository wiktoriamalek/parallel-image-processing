#include "distortion.hpp"
#include "common_benchmarking.hpp"
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
struct Reference {
    bool nearBorder;
    int value;
};

Reference referencePixel(const Image& image, const BrownConradyCoefficients& coefficients, int pixelX, int pixelY) {
    double centerX = (image.width - 1) / 2.0;
    double centerY = (image.height - 1) / 2.0;
    double scale = std::max(image.width, image.height) / 2.0;
    double x = (pixelX - centerX) / scale;
    double y = (pixelY - centerY) / scale;
    double r2 = x * x + y * y;
    double factor = 1.0 + coefficients.k1 * r2 + coefficients.k2 * r2 * r2 + coefficients.k3 * r2 * r2 * r2;
    double sourceX = (x * factor + 2.0 * coefficients.p1 * x * y + coefficients.p2 * (r2 + 2.0 * x * x)) * scale + centerX;
    double sourceY = (y * factor + coefficients.p1 * (r2 + 2.0 * y * y) + 2.0 * coefficients.p2 * x * y) * scale + centerY;
    const double margin = 0.01;
    bool nearBorder = std::abs(sourceX) < margin || std::abs(sourceX - (image.width - 1)) < margin ||
                      std::abs(sourceY) < margin || std::abs(sourceY - (image.height - 1)) < margin;
    if (sourceX < 0 || sourceX > image.width - 1 || sourceY < 0 || sourceY > image.height - 1) return {nearBorder, 0};
    int x0 = static_cast<int>(std::floor(sourceX));
    int y0 = static_cast<int>(std::floor(sourceY));
    int x1 = std::min(image.width - 1, x0 + 1);
    int y1 = std::min(image.height - 1, y0 + 1);
    double fx = sourceX - x0;
    double fy = sourceY - y0;
    auto pixel = [&](int px, int py) { return static_cast<double>(image.pixels[static_cast<std::size_t>(py) * image.width + px]); };
    double top = (1 - fx) * pixel(x0, y0) + fx * pixel(x1, y0);
    double bottom = (1 - fx) * pixel(x0, y1) + fx * pixel(x1, y1);
    return {nearBorder, static_cast<int>(std::lround((1 - fy) * top + fy * bottom))};
}

void checkAgainstReference(const Image& distorted, const Image& corrected, const BrownConradyCoefficients& coefficients) {
    std::vector<std::size_t> indices;
    if (distorted.pixels.size() <= 4'000'000) {
        indices.resize(distorted.pixels.size());
        for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    } else {
        std::mt19937_64 random(11);
        std::uniform_int_distribution<std::size_t> pick(0, distorted.pixels.size() - 1);
        indices.resize(1'000'000);
        for (auto& index : indices) index = pick(random);
    }
    std::size_t checked = 0;
    for (std::size_t index : indices) {
        int pixelX = static_cast<int>(index % distorted.width);
        int pixelY = static_cast<int>(index / distorted.width);
        Reference reference = referencePixel(distorted, coefficients, pixelX, pixelY);
        if (reference.nearBorder) continue;
        int difference = std::abs(reference.value - corrected.pixels[index]);
        if (difference > 1) {
            throw std::runtime_error("CPU Brown-Conrady pixel (" + std::to_string(pixelX) + ", " + std::to_string(pixelY) +
                                     ") is " + std::to_string(corrected.pixels[index]) + ", double reference gives " +
                                     std::to_string(reference.value));
        }
        ++checked;
    }
    std::cout << "Double precision reference: " << checked << " pixels within 1 level\n";
}

void checkRadialSpecialCase(const Image& distorted, const BrownConradyCoefficients& coefficients, int threads) {
    BrownConradyCoefficients radial{coefficients.k1, coefficients.k2, 0.0f, 0.0f, 0.0f};
    Image brownConrady = distorted;
    Image radialOnly = distorted;
    correctBrownConradyCPU(distorted, radial, brownConrady, threads);
    correctRadialDistortionCPU(distorted, coefficients.k1, coefficients.k2, radialOnly, threads);
    if (brownConrady.pixels != radialOnly.pixels) {
        throw std::runtime_error("Brown-Conrady with k3 = p1 = p2 = 0 differs from the radial model");
    }
    std::cout << "k3 = p1 = p2 = 0: identical to the radial model\n";
}

std::string coefficientText(float value) {
    std::ostringstream text;
    text << std::setw(3) << std::setfill('0') << std::lround(std::abs(value) * 1000);
    return (value < 0 ? "m" : "") + text.str();
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/mitsubishi_resolution_change/mitsubishi_16384.png",
                                 "results/brown_conrady_scaling.csv",
                                 {"--k1", "--k2", "--k3", "--p1", "--p2", "--save-distorted"});
        BrownConradyCoefficients coefficients;
        coefficients.k1 = parseNumber<float>(options.parameter("--k1", "0.15"));
        coefficients.k2 = parseNumber<float>(options.parameter("--k2", "0.0"));
        coefficients.k3 = parseNumber<float>(options.parameter("--k3", "0.0"));
        coefficients.p1 = parseNumber<float>(options.parameter("--p1", "0.02"));
        coefficients.p2 = parseNumber<float>(options.parameter("--p2", "0.01"));

        Image image = loadImage(options.image);
        Image distorted = distortBrownConrady(image, coefficients, options.threads);
        std::string saveDirectory = options.parameter("--save-distorted", "");
        if (!saveDirectory.empty()) {
            std::filesystem::create_directories(saveDirectory);
            std::string path = saveDirectory + "/" + std::filesystem::path(options.image).stem().string() +
                               "_brown_conrady_" + coefficientText(coefficients.k1) + "_" + coefficientText(coefficients.k2) +
                               "_" + coefficientText(coefficients.k3) + "_" + coefficientText(coefficients.p1) + "_" +
                               coefficientText(coefficients.p2) + ".png";
            saveImage(path, distorted);
            std::cout << "Distorted image saved to " << path << '\n';
        }

        checkRadialSpecialCase(distorted, coefficients, options.threads);

        Image correctedCPU1 = distorted;
        Image correctedCPUN = distorted;
        Image correctedGPU = distorted;
        double cpu1 = measure([&] { correctBrownConradyCPU(distorted, coefficients, correctedCPU1); }, options.repeats);
        checkAgainstReference(distorted, correctedCPU1, coefficients);
        double cpuN = measure([&] { correctBrownConradyCPU(distorted, coefficients, correctedCPUN, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (correctedCPU1.pixels != correctedCPUN.pixels) throw std::runtime_error("CPU Brown-Conrady results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        BrownConradyGPU gpu(distorted, coefficients);
        std::vector<double> gpuKernelTimes;
        cl_event kernelEvent = nullptr;
        double gpuTotal = measure([&] { gpu.run(distorted, correctedGPU, &kernelEvent); },
                                  options.repeats, [] {}, [&](int repeat) {
            double elapsed = kernelTimeUs(kernelEvent);
            checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
            if (repeat >= 0) gpuKernelTimes.push_back(elapsed);
        });
        double gpuKernel = median(gpuKernelTimes);

        std::size_t differentPixels = 0;
        int maxDifference = 0;
        double totalDifference = 0.0;
        for (std::size_t i = 0; i < correctedCPU1.pixels.size(); i++) {
            int difference = std::abs(int(correctedCPU1.pixels[i]) - int(correctedGPU.pixels[i]));
            differentPixels += difference != 0;
            maxDifference = std::max(maxDifference, difference);
            totalDifference += difference;
        }
        double differentPercent = 100.0 * differentPixels / correctedCPU1.pixels.size();
        double meanDifference = totalDifference / correctedCPU1.pixels.size();

        std::size_t roundTripPixels = 0;
        double roundTripDifference = 0.0;
        for (std::size_t i = 0; i < image.pixels.size(); i++) {
            if (correctedCPU1.pixels[i] == 0) continue;
            roundTripDifference += std::abs(int(correctedCPU1.pixels[i]) - int(image.pixels[i]));
            ++roundTripPixels;
        }

        std::ostringstream row;
        row << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height << ','
            << coefficients.k1 << ',' << coefficients.k2 << ',' << coefficients.k3 << ','
            << coefficients.p1 << ',' << coefficients.p2 << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal << ','
            << differentPixels << ',' << differentPercent << ',' << maxDifference << ',' << meanDifference;
        appendCsv(options.output,
                  "image,width,height,k1,k2,k3,p1,p2,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,different_pixels,different_percent,max_abs,mean_abs",
                  row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        std::cout << "CPU1/GPU: " << differentPixels << " different pixels ("
                  << differentPercent << "%), max abs: " << maxDifference << ", mean abs: " << meanDifference << '\n';
        std::cout << "Distort + correct vs original: mean abs " << roundTripDifference / std::max<std::size_t>(1, roundTripPixels)
                  << " over " << roundTripPixels << " pixels\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
