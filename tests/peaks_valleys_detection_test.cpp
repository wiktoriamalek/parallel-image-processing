#include "peaks_valleys_detection.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "otsu.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
struct Reference {
    std::vector<int> thresholds;
    int iterations = 0;
};

Reference referenceThresholds(const Histogram& histogram, int maxClasses) {
    std::vector<std::uint64_t> padded(258, 0);
    for (int i = 0; i < 256; ++i) padded[i + 1] = static_cast<std::uint64_t>(histogram[i]) << 30;
    Reference reference;
    int previousPeaks = INT_MAX;
    for (;;) {
        std::vector<std::pair<int, int>> signs;
        for (int i = 1; i <= 257; ++i) {
            if (padded[i] != padded[i - 1]) signs.emplace_back(i, padded[i] > padded[i - 1] ? 1 : -1);
        }
        int peaks = 0;
        std::vector<int> valleys;
        for (std::size_t s = 1; s < signs.size(); ++s) {
            if (signs[s - 1].second > 0 && signs[s].second < 0) ++peaks;
            if (signs[s - 1].second < 0 && signs[s].second > 0) {
                valleys.push_back((signs[s - 1].first + signs[s].first - 1) / 2 - 1);
            }
        }
        if (peaks > previousPeaks) throw std::runtime_error("Smoothing increased the number of peaks");
        previousPeaks = peaks;
        if (peaks <= maxClasses) {
            reference.thresholds = valleys;
            return reference;
        }
        std::vector<std::uint64_t> next(258, 0);
        for (int i = 1; i <= 256; ++i) next[i] = (padded[i - 1] + 2 * padded[i] + padded[i + 1]) / 4;
        padded = std::move(next);
        ++reference.iterations;
    }
}

int rawPeaks(const Histogram& histogram) {
    int peaks = 0;
    for (int i = 1; i <= 254; ++i) peaks += histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1];
    return peaks;
}

std::string join(const std::vector<int>& values) {
    std::ostringstream text;
    for (std::size_t i = 0; i < values.size(); ++i) text << (i ? " " : "") << values[i];
    return text.str();
}

std::vector<int> checkThresholds(const Histogram& histogram, int maxClasses, const std::string& name) {
    int iterations = -1;
    const std::vector<int> thresholds = peaks_valleys_detectionThresholds(histogram, maxClasses, &iterations);
    const Reference reference = referenceThresholds(histogram, maxClasses);
    if (thresholds != reference.thresholds || iterations != reference.iterations) {
        throw std::runtime_error(name + ", " + std::to_string(maxClasses) + " classes: thresholds " + join(thresholds) +
                                 " after " + std::to_string(iterations) + " iterations, reference " +
                                 join(reference.thresholds) + " after " + std::to_string(reference.iterations));
    }
    if (static_cast<int>(thresholds.size()) > maxClasses - 1 || !std::is_sorted(thresholds.begin(), thresholds.end()) ||
        std::adjacent_find(thresholds.begin(), thresholds.end()) != thresholds.end()) {
        throw std::runtime_error(name + ": wrong thresholds " + join(thresholds));
    }
    return thresholds;
}

Histogram gaussians(const std::vector<std::pair<double, double>>& peaks, double height = 1e5) {
    Histogram histogram{};
    for (int level = 0; level < 256; ++level) {
        double value = 0.0;
        for (auto [mean, width] : peaks) {
            const double distance = (level - mean) / width;
            value += height * std::exp(-0.5 * distance * distance);
        }
        histogram[level] = static_cast<std::size_t>(value);
    }
    return histogram;
}

void expectThresholds(const Histogram& histogram, int maxClasses, const std::vector<std::pair<int, int>>& ranges,
                      const std::string& name) {
    const std::vector<int> thresholds = checkThresholds(histogram, maxClasses, name);
    bool ok = thresholds.size() == ranges.size();
    for (std::size_t k = 0; ok && k < ranges.size(); ++k) {
        ok = thresholds[k] >= ranges[k].first && thresholds[k] <= ranges[k].second;
    }
    if (!ok) throw std::runtime_error(name + ": unexpected thresholds " + join(thresholds));
}

void checkSyntheticHistograms() {
    expectThresholds(gaussians({{60, 12}, {190, 12}}), 2, {{110, 140}}, "two peaks");
    expectThresholds(gaussians({{40, 10}, {128, 10}, {210, 10}}), 3, {{75, 95}, {160, 180}}, "three peaks");
    expectThresholds(gaussians({{40, 10}, {128, 10}, {210, 10}}), 2, {{1, 254}}, "three peaks into two classes");

    Histogram edges{};
    edges[0] = 5000;
    edges[128] = 3000;
    edges[255] = 5000;
    expectThresholds(edges, 3, {{1, 127}, {129, 254}}, "peaks at 0 and 255");

    Histogram plateau{};
    for (int level : {50, 51, 52}) plateau[level] = 100;
    for (int level : {150, 151}) plateau[level] = 100;
    expectThresholds(plateau, 2, {{53, 149}}, "flat peaks");
    if (checkThresholds(plateau, 2, "flat peaks") != std::vector<int>{101}) {
        throw std::runtime_error("The valley is not in the middle of an empty gap");
    }

    Histogram single{};
    single[77] = 1000;
    expectThresholds(single, 2, {}, "one level");

    Histogram uniform{};
    uniform.fill(10);
    expectThresholds(uniform, 2, {}, "uniform histogram");

    Histogram comb = gaussians({{70, 20}, {180, 20}});
    for (int level = 1; level < 256; level += 2) comb[level] = 0;
    expectThresholds(comb, 2, {{110, 140}}, "comb histogram");

    Histogram empty{};
    expectThresholds(empty, 2, {}, "empty histogram");

    for (int classes : {1, 0, 257}) {
        bool thrown = false;
        try {
            peaks_valleys_detectionThresholds(edges, classes);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        if (!thrown) throw std::runtime_error(std::to_string(classes) + " classes were accepted");
    }
}

void checkRandomHistograms() {
    std::mt19937 generator(2028);
    std::uniform_int_distribution<int> peakCount(1, 8);
    std::uniform_real_distribution<double> peakMean(-20.0, 275.0);
    std::uniform_real_distribution<double> peakWidth(0.5, 40.0);
    std::uniform_real_distribution<double> peakHeight(1.0, 1e7);
    std::uniform_real_distribution<double> noise(0.0, 1.0);
    std::uniform_int_distribution<int> sparse(0, 3);
    for (int test = 0; test < 300; ++test) {
        std::vector<std::pair<double, double>> peaks;
        Histogram histogram{};
        for (int peak = peakCount(generator); peak > 0; --peak) {
            const double mean = peakMean(generator);
            const double width = peakWidth(generator);
            const double height = peakHeight(generator);
            for (int level = 0; level < 256; ++level) {
                const double distance = (level - mean) / width;
                histogram[level] += static_cast<std::size_t>(height * std::exp(-0.5 * distance * distance) * noise(generator));
            }
        }
        if (test % 3 == 0) {
            for (auto& count : histogram) count *= sparse(generator) == 0;
        }
        for (int classes : {2, 3, 4, 5, 8, 16}) {
            checkThresholds(histogram, classes, "random histogram " + std::to_string(test));
        }
    }
}

Image randomImage(int width, int height, std::mt19937& generator) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    std::uniform_real_distribution<double> mean(0.0, 255.0);
    std::uniform_real_distribution<double> spread(1.0, 30.0);
    std::vector<std::normal_distribution<double>> peaks;
    for (int peak = 1 + static_cast<int>(generator() % 5); peak > 0; --peak) {
        peaks.emplace_back(mean(generator), spread(generator));
    }
    std::uniform_int_distribution<std::size_t> pick(0, peaks.size() - 1);
    for (auto& pixel : image.pixels) {
        pixel = static_cast<unsigned char>(std::clamp(std::lround(peaks[pick(generator)](generator)), 0L, 255L));
    }
    return image;
}

void checkSameAsCPU(const Image& image, int maxClasses, int threads, Peaks_valleys_detectionGPU& gpu,
                    const std::string& name) {
    int iterations = 0;
    const std::vector<int> thresholds = peaks_valleys_detectionThresholds(calculateHistogram(image), maxClasses, &iterations);
    const std::array<unsigned char, 256> lookup = multiOtsuLookupTable(thresholds);
    Image expected = image;
    for (auto& pixel : expected.pixels) pixel = lookup[pixel];

    if (peaks_valleys_detection(image, maxClasses).pixels != expected.pixels) {
        throw std::runtime_error(name + ": peaks_valleys_detection differs from its own thresholds");
    }
    Image parallel = image;
    peaks_valleys_detectionParallel(image, parallel, threads, maxClasses);
    if (parallel.pixels != expected.pixels) throw std::runtime_error(name + ": CPU with threads differs");

    Image gpuResult = image;
    gpu.run(image, gpuResult);
    if (gpu.thresholds() != thresholds || gpu.iterations() != iterations) {
        throw std::runtime_error(name + ", " + std::to_string(maxClasses) + " classes: GPU thresholds " +
                                 join(gpu.thresholds()) + " after " + std::to_string(gpu.iterations()) +
                                 " iterations, CPU " + join(thresholds) + " after " + std::to_string(iterations));
    }
    if (gpuResult.pixels != expected.pixels) throw std::runtime_error(name + ": GPU segmentation differs from CPU");
}

void checkRandomImages(int threads) {
    std::mt19937 generator(2029);
    for (auto [width, height] : {std::pair{1, 1}, {2, 1}, {1, 7}, {3, 5}, {64, 64}, {257, 129}, {1000, 662}}) {
        std::vector<Image> images;
        for (int test = 0; test < 6; ++test) images.push_back(randomImage(width, height, generator));
        for (bool zeroCopy : {false, true}) {
            for (int classes : {2, 3, 4, 6}) {
                Peaks_valleys_detectionGPU gpu(images.front(), classes, zeroCopy);
                for (const Image& image : images) {
                    checkSameAsCPU(image, classes, threads, gpu, "random image " + std::to_string(width) + "x" +
                                   std::to_string(height) + (zeroCopy ? " (zero-copy)" : ""));
                }
            }
        }
    }
    Image edges{4, 4, {0, 0, 0, 0, 0, 0, 128, 128, 128, 255, 255, 255, 255, 255, 255, 255}};
    for (bool zeroCopy : {false, true}) {
        Peaks_valleys_detectionGPU gpu(edges, 3, zeroCopy);
        checkSameAsCPU(edges, 3, threads, gpu, "image with peaks at 0 and 255");
    }
}

double takeKernelTime(cl_event& event) {
    const double time = kernelTimeUs(event);
    checkCL(clReleaseEvent(event), "clReleaseEvent");
    event = nullptr;
    return time;
}

struct GPUResult {
    double totalUs = 0.0;
    double kernelUs = 0.0;
    double thresholdsKernelUs = 0.0;
    double noCopyUs = 0.0;
};

GPUResult measureGPU(const Image& image, int classes, bool zeroCopy, int repeats, const Image& expected) {
    Peaks_valleys_detectionGPU gpu(image, classes, zeroCopy);
    Image output = image;
    cl_event histogramEvent = nullptr;
    cl_event thresholdsEvent = nullptr;
    cl_event lookupEvent = nullptr;
    std::vector<double> kernelTimes;
    std::vector<double> thresholdsTimes;
    GPUResult result;
    result.totalUs = measure([&] { gpu.run(image, output, &histogramEvent, &thresholdsEvent, &lookupEvent); },
                             repeats, [] {}, [&](int repeat) {
        const double thresholdsTime = takeKernelTime(thresholdsEvent);
        const double time = takeKernelTime(histogramEvent) + thresholdsTime + takeKernelTime(lookupEvent);
        if (repeat >= 0) {
            kernelTimes.push_back(time);
            thresholdsTimes.push_back(thresholdsTime);
        }
        if (output.pixels != expected.pixels) throw std::runtime_error("GPU and CPU results differ");
    });
    result.kernelUs = median(kernelTimes);
    result.thresholdsKernelUs = median(thresholdsTimes);

    std::fill(output.pixels.begin(), output.pixels.end(), 1);
    gpu.upload(image, output);
    result.noCopyUs = measure([&] { gpu.compute(); gpu.finish(); }, repeats);
    gpu.download(output);
    if (output.pixels != expected.pixels) throw std::runtime_error("GPU result without copying differs");
    return result;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png",
                                 "results/peaks_valleys_scaling.csv", {"--classes"});
        const int classes = parseNumber<int>(options.parameter("--classes", "3"));

        checkSyntheticHistograms();
        checkRandomHistograms();
        std::cout << "Thresholds verified against a reference implementation (synthetic and 300 random histograms, 2-16 classes)\n";
        checkRandomImages(options.threads);
        std::cout << "CPU 1 thread, CPU " << options.threads << " threads and GPU give identical results on 42 random images (2, 3, 4 and 6 classes, GPU with and without zero-copy)\n";

        const Image image = loadImage(options.image);
        const Histogram histogram = calculateHistogram(image);
        int iterations = 0;
        const std::vector<int> thresholds = peaks_valleys_detectionThresholds(histogram, classes, &iterations);
        checkThresholds(histogram, classes, "image");
        for (bool zeroCopy : {false, true}) {
            Peaks_valleys_detectionGPU gpu(image, classes, zeroCopy);
            checkSameAsCPU(image, classes, options.threads, gpu, "image");
        }

        Image cpuResult = image;
        Image result = image;
        const double cpu1 = measure([&] { peaks_valleys_detectionParallel(image, cpuResult, 1, classes); }, options.repeats);
        const double cpuN = measure([&] { peaks_valleys_detectionParallel(image, result, options.threads, classes); },
                                    options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU results differ");
        });
        const double overheadN = measureThreadOverhead(options.threads, options.repeats);
        const double search = measure([&] { peaks_valleys_detectionThresholds(histogram, classes); }, options.repeats);

        const GPUResult gpu = measureGPU(image, classes, false, options.repeats, cpuResult);
        const GPUResult zeroCopy = measureGPU(image, classes, true, options.repeats, cpuResult);

        std::filesystem::path outputDirectory = "results/peaks_valleys";
        std::filesystem::create_directories(outputDirectory);
        const std::string imageName = std::filesystem::path(options.image).stem().string();
        saveImage((outputDirectory / (imageName + "_" + std::to_string(classes) + "_classes.png")).string(), cpuResult);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height
            << ',' << classes << ',' << std::quoted(join(thresholds), '"', '"') << ',' << iterations << ','
            << options.threads << ',' << options.repeats << ',' << cpu1 << ',' << cpuN << ',' << overheadN << ','
            << search << ',' << gpu.thresholdsKernelUs << ',' << gpu.kernelUs << ',' << gpu.totalUs << ','
            << gpu.noCopyUs << ',' << zeroCopy.totalUs;
        appendCsv(options.output, "image,width,height,max_classes,thresholds,iterations,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,threshold_search_us,gpu_threshold_search_us,gpu_kernel_us,gpu_total_us,gpu_no_copy_us,gpu_zero_copy_us", row.str());

        const int found = static_cast<int>(thresholds.size()) + 1;
        std::cout << "Peaks in the raw histogram: " << rawPeaks(histogram) << ", smoothing iterations: " << iterations << '\n';
        std::cout << "Valley thresholds (" << found << " classes): " << join(thresholds);
        if (found >= 2) std::cout << ", multi-Otsu with " << found << " classes: " << join(calculateMultiOtsuThresholds(histogram, found));
        std::cout << '\n';
        std::cout << "Median CPU1 / CPU" << options.threads << " / threshold search only [us]: "
                  << cpu1 << " / " << cpuN << " / " << search << '\n';
        std::cout << "Median GPU total / GPU zero-copy / GPU without copying / GPU kernels / GPU threshold search kernel [us]: "
                  << gpu.totalUs << " / " << zeroCopy.totalUs << " / " << gpu.noCopyUs << " / " << gpu.kernelUs << " / "
                  << gpu.thresholdsKernelUs << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
