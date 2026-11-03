#include "region_growing.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct GPUResult {
    double totalUs = -1.0;
    double kernelUs = -1.0;
    int iterations = -1;
    long long differentPixels = -1;
};

long long differentPixels(const Image& reference, const Image& image) {
    long long different = 0;
    for (std::size_t i = 0; i < reference.pixels.size(); i++) different += reference.pixels[i] != image.pixels[i];
    return different;
}

GPUResult measureGPU(const Image& image, int seedX, int seedY, int tolerance, bool localTiles, int repeats,
                     const Image& reference) {
    RegionGrowingGPU gpu(image, seedX, seedY, tolerance, localTiles);
    Image output = image;
    std::vector<double> kernelTimes;
    double kernelUs = 0.0;
    GPUResult result;
    result.totalUs = measure([&] { gpu.run(image, output, &kernelUs); }, repeats, [] {}, [&](int repeat) {
        if (repeat >= 0) kernelTimes.push_back(kernelUs);
    });
    result.kernelUs = median(kernelTimes);
    result.iterations = gpu.iterations();
    result.differentPixels = differentPixels(reference, output);
    return result;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/nuclear.png", "results/region_growing_scaling.csv",
                                 {"--seed-x", "--seed-y", "--tolerance", "--naive"});
        Image image = loadImage(options.image);
        int seedX = parseNumber<int>(options.parameter("--seed-x", std::to_string(image.width / 2)));
        int seedY = parseNumber<int>(options.parameter("--seed-y", std::to_string(image.height / 2)));
        int tolerance = parseNumber<int>(options.parameter("--tolerance", "60"));
        bool naive = parseNumber<int>(options.parameter("--naive", "0")) != 0;

        Image regionCPU1;
        Image regionCPUN;
        double cpu1 = measure([&] { regionCPU1 = regionGrowingCPU(image, seedX, seedY, tolerance); }, options.repeats);
        double cpuN = measure([&] {
            regionCPUN = regionGrowingCPUParallel(image, seedX, seedY, tolerance, options.threads);
        }, options.repeats);
        if (regionCPU1.pixels != regionCPUN.pixels) throw std::runtime_error("CPU region growing results differ");
        double overheadN = measureThreadOverhead(options.threads, options.repeats);
        long long regionPixels = std::count(regionCPU1.pixels.begin(), regionCPU1.pixels.end(), 255);

        GPUResult tiles = measureGPU(image, seedX, seedY, tolerance, true, options.repeats, regionCPU1);
        GPUResult pixelSteps;
        if (naive) pixelSteps = measureGPU(image, seedX, seedY, tolerance, false, options.repeats, regionCPU1);

        std::ostringstream row;
        row << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height << ','
            << seedX << ',' << seedY << ',' << tolerance << ',' << regionPixels << ','
            << options.threads << ',' << options.repeats << ',' << cpu1 << ',' << cpuN << ',' << overheadN << ','
            << tiles.kernelUs << ',' << tiles.totalUs << ',' << tiles.iterations << ',' << tiles.differentPixels << ','
            << pixelSteps.kernelUs << ',' << pixelSteps.totalUs << ',' << pixelSteps.iterations << ','
            << pixelSteps.differentPixels;
        appendCsv(options.output,
                  "image,width,height,seed_x,seed_y,tolerance,region_pixels,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,gpu_iterations,gpu_different_pixels,gpu_naive_kernel_us,gpu_naive_total_us,gpu_naive_iterations,gpu_naive_different_pixels",
                  row.str());
        std::cout << "Region: " << regionPixels << " px (" << 100.0 * regionPixels / regionCPU1.pixels.size() << "%)\n";
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << tiles.totalUs << " / " << tiles.kernelUs
                  << " (" << tiles.iterations << " iterations, " << tiles.differentPixels << " different pixels)\n";
        if (naive) {
            std::cout << "GPU without local tiles total / kernel [us]: " << pixelSteps.totalUs << " / " << pixelSteps.kernelUs
                      << " (" << pixelSteps.iterations << " iterations, " << pixelSteps.differentPixels << " different pixels)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
