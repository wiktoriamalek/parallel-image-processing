#include "otsu.hpp"
#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_256.png", "results/otsu_scaling.csv");
        Image image = loadImage(options.image);
        Histogram histogram = calculateHistogram(image);
        if (calculateHistogram(image, options.threads) != histogram) throw std::runtime_error("CPU histograms differ");

        Image cpuResult;
        Image result;
        double cpu1 = measure([&] { cpuResult = otsuSegmentation(image); }, options.repeats);
        double cpuN = measure([&] { result = otsuSegmentation(image, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU Otsu results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        OtsuGPU gpu(image);
        Image gpuResult = image;
        cl_event histogramEvent = nullptr;
        cl_event otsuThresholdEvent = nullptr;
        cl_event thresholdEvent = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &histogramEvent, &otsuThresholdEvent, &thresholdEvent); },
                                  options.repeats, [] {}, [&](int repeat) {
            // the three kernels of the algorithm, the tiny kernel that zeroes the histogram is not counted
            double time = kernelTimeUs(histogramEvent) + kernelTimeUs(otsuThresholdEvent) + kernelTimeUs(thresholdEvent);
            checkCL(clReleaseEvent(histogramEvent), "clReleaseEvent");
            checkCL(clReleaseEvent(otsuThresholdEvent), "clReleaseEvent");
            checkCL(clReleaseEvent(thresholdEvent), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (gpuResult.pixels != cpuResult.pixels) {
                throw std::runtime_error("GPU and CPU Otsu results differ");
            }
        });
        double gpuKernel = median(kernelTimes);

        // the image stays on the GPU: it is sent once before the measurement and the result is read once after it,
        // so only the three kernels, their launch and the waiting for the GPU are measured
        gpu.upload(image);
        double gpuNoCopy = measure([&] { gpu.compute(); gpu.finish(); }, options.repeats);
        Image gpuNoCopyResult = image; // a fresh image, so the check below really tests what came back from the GPU
        gpu.download(gpuNoCopyResult);
        if (gpuNoCopyResult.pixels != cpuResult.pixels) {
            throw std::runtime_error("GPU without copying and CPU Otsu results differ");
        }
        int threshold = calculateOtsuThreshold(histogram);

        // CPU and GPU results are identical at this point, so one image and one histogram are enough
        std::filesystem::path outputDirectory = "results/otsu";
        std::filesystem::create_directories(outputDirectory);
        std::string imageName = std::filesystem::path(options.image).stem().string();
        saveImage((outputDirectory / (imageName + "_otsu.png")).string(), cpuResult);
        std::ofstream histogramFile(outputDirectory / (imageName + "_histogram.csv"));
        histogramFile << "intensity,count\n";
        for (std::size_t intensity = 0; intensity < histogram.size(); intensity++) {
            histogramFile << intensity << ',' << histogram[intensity] << '\n';
        }
        histogramFile.close();
        if (!histogramFile) throw std::runtime_error("Could not write histogram CSV");

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ','
            << image.width << ',' << image.height << ',' << threshold << ','
            << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal << ',' << gpuNoCopy;
        appendCsv(options.output, "image,width,height,threshold,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,gpu_no_copy_us", row.str());
        std::cout << "Median CPU1 / CPU" << options.threads << " / GPU total / GPU without copying / GPU kernels [us]: "
                  << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuNoCopy << " / " << gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
