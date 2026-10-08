#include "distortion.hpp"
#include "common_benchmarking.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/mitsubishi_resolution_change/mitsubishi_16384.png",
                                 "results/distortion_scaling.csv", {"--k1", "--k2"});
        // Real image
        Image image = loadImage(options.image);
        float k1 = parseNumber<float>(options.parameter("--k1", "0.15"));
        float k2 = parseNumber<float>(options.parameter("--k2", "0.0"));
        // Create distorted image
        Image distorted = distortRadial(image, k1, k2);
        //saveImage("images/mitsubishi_distorted/mitusbishi_distorted_075_075.png", distorted);
        Image correctedCPU1 = distorted;
        Image correctedCPUN = distorted;
        Image correctedGPU = distorted;
        RadialDistortionGPU gpu(distorted, k1, k2);

        // CPU1
        double cpu1 = measure([&] { correctRadialDistortionCPU(distorted, k1, k2, correctedCPU1); }, options.repeats);
        // CPU2
        double cpuN = measure([&] { correctRadialDistortionCPU(distorted, k1, k2, correctedCPUN, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (correctedCPU1.pixels != correctedCPUN.pixels) {
                throw std::runtime_error("CPU distortion results differ");
            }
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        // GPU
        std::vector<double> gpuKernelTimes;
        cl_event kernelEvent = nullptr;
        double gpuTotal = measure([&] { gpu.run(distorted, correctedGPU, &kernelEvent); },
                                  options.repeats, [] {}, [&](int repeat) {
            double elapsed = kernelTimeUs(kernelEvent);
            clReleaseEvent(kernelEvent);
            if (repeat >= 0) gpuKernelTimes.push_back(elapsed);
        });
        double gpuKernel = median(gpuKernelTimes);
        //saveImage("results/distortion/CPU_1T/mitsubishi_distortion_075_075.png", correctedCPU1);
        //saveImage("results/distortion/CPU2T/mitsubishi_distortion_075_075.png", correctedCPUN);
        //saveImage("results/distortion/GPU/mitsubishi_distortion_GPU_075_075.png", correctedGPU);

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
        std::ostringstream row;
        row << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height << ','
            << k1 << ',' << k2 << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal << ','
            << differentPixels << ',' << differentPercent << ',' << maxDifference << ',' << meanDifference;
        appendCsv(options.output,
                  "image,width,height,k1,k2,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,different_pixels,different_percent,max_abs,mean_abs",
                  row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel << '\n';
        std::cout << "CPU1/GPU: " << differentPixels << " different pixels ("
                  << differentPercent << "%), max abs: " << maxDifference << ", mean abs: " << meanDifference << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
