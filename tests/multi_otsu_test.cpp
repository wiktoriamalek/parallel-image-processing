#include "otsu.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Cumulative {
    std::array<double, 257> count{};
    std::array<double, 257> sum{};

    explicit Cumulative(const Histogram& histogram) {
        for (int level = 0; level < 256; ++level) {
            count[level + 1] = count[level] + static_cast<double>(histogram[level]);
            sum[level + 1] = sum[level] + level * static_cast<double>(histogram[level]);
        }
    }

    double betweenClassVariance(const std::vector<int>& thresholds) const {
        const double mean = sum[256] / count[256];
        double variance = 0.0;
        int begin = 0;
        for (std::size_t k = 0; k <= thresholds.size(); ++k) {
            const int end = k < thresholds.size() ? thresholds[k] + 1 : 256;
            const double classCount = count[end] - count[begin];
            if (classCount == 0) throw std::runtime_error("Empty class");
            const double difference = (sum[end] - sum[begin]) / classCount - mean;
            variance += classCount / count[256] * difference * difference;
            begin = end;
        }
        return variance;
    }
};

void searchAllThresholds(const Cumulative& cumulative, const std::vector<int>& candidates, int classes,
                         std::size_t from, std::vector<int>& current, double& bestValue, std::vector<int>& best) {
    if (static_cast<int>(current.size()) == classes - 1) {
        const double value = cumulative.betweenClassVariance(current);
        if (value > bestValue) {
            bestValue = value;
            best = current;
        }
        return;
    }
    for (std::size_t i = from; i < candidates.size(); ++i) {
        current.push_back(candidates[i]);
        searchAllThresholds(cumulative, candidates, classes, i + 1, current, bestValue, best);
        current.pop_back();
    }
}

std::vector<int> naiveThresholds(const Histogram& histogram, int classes) {
    std::vector<int> candidates;
    for (int level = 0; level < 256; ++level) {
        if (histogram[level] > 0) candidates.push_back(level);
    }
    candidates.pop_back();
    Cumulative cumulative(histogram);
    std::vector<int> current;
    std::vector<int> best;
    double bestValue = -1.0;
    searchAllThresholds(cumulative, candidates, classes, 0, current, bestValue, best);
    return best;
}

std::vector<int> fullDynamicProgramming(const Histogram& histogram, int classes) {
    std::vector<int> levels;
    for (int level = 0; level < 256; ++level) {
        if (histogram[level] > 0) levels.push_back(level);
    }
    const int distinct = static_cast<int>(levels.size());
    const Cumulative cumulative(histogram);
    auto gain = [&](int begin, int end) {
        const double count = cumulative.count[levels[end - 1] + 1] - cumulative.count[levels[begin]];
        const double sum = cumulative.sum[levels[end - 1] + 1] - cumulative.sum[levels[begin]];
        return sum * sum / count;
    };
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> best(classes + 1, std::vector<double>(distinct + 1, -infinity));
    std::vector<std::vector<int>> split(classes + 1, std::vector<int>(distinct + 1));
    best[0][0] = 0.0;
    for (int k = 1; k <= classes; ++k) {
        for (int end = k; end <= distinct; ++end) {
            for (int begin = k - 1; begin < end; ++begin) {
                if (best[k - 1][begin] == -infinity) continue;
                const double value = best[k - 1][begin] + gain(begin, end);
                if (value > best[k][end]) {
                    best[k][end] = value;
                    split[k][end] = begin;
                }
            }
        }
    }
    std::vector<int> thresholds(classes - 1);
    int end = distinct;
    for (int k = classes; k >= 2; --k) {
        end = split[k][end];
        thresholds[k - 2] = levels[end - 1];
    }
    return thresholds;
}

std::string join(const std::vector<int>& values) {
    std::ostringstream text;
    for (std::size_t i = 0; i < values.size(); ++i) text << (i ? " " : "") << values[i];
    return text.str();
}

void checkOptimal(const Histogram& histogram, int classes, const std::vector<int>& expected,
                  const std::string& reference, const std::string& name) {
    const std::vector<int> thresholds = calculateMultiOtsuThresholds(histogram, classes);
    const Cumulative cumulative(histogram);
    const double value = cumulative.betweenClassVariance(thresholds);
    const double expectedValue = cumulative.betweenClassVariance(expected);
    if (value < expectedValue * (1.0 - 1e-12)) {
        throw std::runtime_error(name + ", " + std::to_string(classes) + " classes: thresholds " + join(thresholds) +
                                 " are worse than the " + reference + " result " + join(expected));
    }
}

void checkExhaustive(const Histogram& histogram, int classes, const std::string& name) {
    checkOptimal(histogram, classes, naiveThresholds(histogram, classes), "exhaustive search", name);
}

void checkFullDynamicProgramming(const Histogram& histogram, int classes, const std::string& name) {
    checkOptimal(histogram, classes, fullDynamicProgramming(histogram, classes), "full dynamic programming", name);
}

bool throwsInvalidArgument(const Histogram& histogram, int classes) {
    try {
        calculateMultiOtsuThresholds(histogram, classes);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void checkImage(const Histogram& histogram, int maxClasses) {
    for (int classes = 2; classes <= maxClasses; ++classes) checkExhaustive(histogram, classes, "image");
    for (int classes = 2; classes <= 16; ++classes) checkFullDynamicProgramming(histogram, classes, "image");

    const Cumulative cumulative(histogram);
    const double otsuValue = cumulative.betweenClassVariance({calculateOtsuThreshold(histogram)});
    const double multiOtsuValue = cumulative.betweenClassVariance(calculateMultiOtsuThresholds(histogram, 2));
    if (std::abs(otsuValue - multiOtsuValue) > 1e-12 * otsuValue) {
        throw std::runtime_error("Multi-Otsu with 2 classes differs from Otsu");
    }

    if (!throwsInvalidArgument(histogram, 1)) throw std::runtime_error("1 class was accepted");
}

void checkRandomHistograms() {
    std::mt19937 generator(2026);
    std::uniform_int_distribution<int> levelCount(1, 12);
    std::uniform_int_distribution<int> level(0, 255);
    std::uniform_int_distribution<int> magnitude(0, 8);
    for (int test = 0; test < 300; ++test) {
        Histogram histogram{};
        const int levels = levelCount(generator);
        int distinct = 0;
        while (distinct < levels) {
            const int chosen = level(generator);
            if (histogram[chosen] != 0) continue;
            const int digits = magnitude(generator);
            std::uniform_int_distribution<std::size_t> count(1, static_cast<std::size_t>(std::pow(10, digits)));
            histogram[chosen] = count(generator);
            ++distinct;
        }
        for (int classes = 2; classes <= 7; ++classes) {
            if (classes > distinct) {
                if (!throwsInvalidArgument(histogram, classes)) {
                    throw std::runtime_error("Too many classes for the histogram were accepted");
                }
                continue;
            }
            checkExhaustive(histogram, classes, "sparse random histogram " + std::to_string(test));
        }
    }

    std::uniform_int_distribution<int> peakCount(1, 6);
    std::uniform_real_distribution<double> peakMean(0.0, 255.0);
    std::uniform_real_distribution<double> peakWidth(2.0, 40.0);
    std::uniform_real_distribution<double> peakHeight(10.0, 1e6);
    std::uniform_real_distribution<double> noise(0.0, 1.0);
    for (int test = 0; test < 100; ++test) {
        std::vector<double> means, widths, heights;
        for (int peak = peakCount(generator); peak > 0; --peak) {
            means.push_back(peakMean(generator));
            widths.push_back(peakWidth(generator));
            heights.push_back(peakHeight(generator));
        }
        Histogram histogram{};
        for (int brightness = 0; brightness < 256; ++brightness) {
            double value = 0.0;
            for (std::size_t peak = 0; peak < means.size(); ++peak) {
                const double distance = (brightness - means[peak]) / widths[peak];
                value += heights[peak] * std::exp(-0.5 * distance * distance);
            }
            histogram[brightness] = static_cast<std::size_t>(value * noise(generator));
        }
        int distinct = 0;
        for (std::size_t count : histogram) distinct += count > 0;
        for (int classes = 2; classes <= std::min(16, distinct); ++classes) {
            checkFullDynamicProgramming(histogram, classes, "dense random histogram " + std::to_string(test));
        }
    }
}

Image classifyDirectly(const Image& image, const std::vector<int>& thresholds) {
    const int classes = static_cast<int>(thresholds.size()) + 1;
    Image result = image;
    for (auto& pixel : result.pixels) {
        int k = 0;
        while (k < classes - 1 && pixel > thresholds[k]) ++k;
        pixel = static_cast<unsigned char>(255 * k / (classes - 1));
    }
    return result;
}

long long differentPixels(const Image& reference, const Image& image) {
    long long different = 0;
    for (std::size_t i = 0; i < reference.pixels.size(); ++i) different += reference.pixels[i] != image.pixels[i];
    return different;
}

void checkGPUThresholds(const Image& image, MultiOtsuGPU& gpu, int classes, const std::string& name) {
    const Histogram histogram = calculateHistogram(image);
    int distinct = 0;
    for (std::size_t count : histogram) distinct += count > 0;
    Image output = image;
    if (distinct < classes) {
        try {
            gpu.run(image, output);
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error(name + ": GPU accepted too many classes for the histogram");
    }
    gpu.run(image, output);
    const Cumulative cumulative(histogram);
    const double value = cumulative.betweenClassVariance(gpu.thresholds());
    const std::vector<int> optimal = calculateMultiOtsuThresholds(histogram, classes);
    const double best = cumulative.betweenClassVariance(optimal);
    if (value < best * (1.0 - 1e-5)) {
        std::ostringstream message;
        message << std::setprecision(12) << name << ", " << classes << " classes: GPU thresholds " << join(gpu.thresholds())
                << " (variance " << value << ") are far from the optimal " << join(optimal) << " (variance " << best << ")";
        throw std::runtime_error(message.str());
    }
    if (output.pixels != classifyDirectly(image, gpu.thresholds()).pixels) {
        throw std::runtime_error(name + ": GPU segmentation does not match its own thresholds");
    }
}

void checkGPURandomImages() {
    std::mt19937 generator(2027);
    std::vector<Image> images;
    for (int test = 0; test < 100; ++test) {
        Image image{64, 64, std::vector<unsigned char>(64 * 64)};
        if (test < 50) {
            std::vector<int> levels(1 + test % 12);
            std::uniform_int_distribution<int> level(0, 255);
            for (int& value : levels) value = level(generator);
            std::uniform_int_distribution<std::size_t> pick(0, levels.size() - 1);
            for (auto& pixel : image.pixels) pixel = static_cast<unsigned char>(levels[pick(generator)]);
        } else {
            std::uniform_real_distribution<double> mean(0.0, 255.0);
            std::uniform_real_distribution<double> width(2.0, 40.0);
            std::vector<std::normal_distribution<double>> peaks;
            for (int peak = 1 + test % 6; peak > 0; --peak) peaks.emplace_back(mean(generator), width(generator));
            std::uniform_int_distribution<std::size_t> pick(0, peaks.size() - 1);
            for (auto& pixel : image.pixels) {
                pixel = static_cast<unsigned char>(std::clamp(std::lround(peaks[pick(generator)](generator)), 0L, 255L));
            }
        }
        images.push_back(std::move(image));
    }
    for (bool zeroCopy : {false, true}) {
        for (int classes : {2, 3, 4, 5, 6, 7, 8, 12, 16}) {
            MultiOtsuGPU gpu(images.front(), classes, zeroCopy);
            for (std::size_t test = 0; test < images.size(); ++test) {
                checkGPUThresholds(images[test], gpu, classes, "random GPU image " + std::to_string(test));
            }
        }
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
    double noCopyUs = 0.0;
    std::vector<int> thresholds;
    long long differentPixels = 0;
    Image output;
};

GPUResult measureGPU(const Image& image, int classes, bool zeroCopy, int repeats, const Image& reference) {
    MultiOtsuGPU gpu(image, classes, zeroCopy);
    Image output = image;
    Image first;
    cl_event histogramEvent = nullptr;
    std::vector<cl_event> thresholdsEvents;
    cl_event lookupEvent = nullptr;
    std::vector<double> kernelTimes;
    GPUResult result;
    result.totalUs = measure([&] { gpu.run(image, output, &histogramEvent, &thresholdsEvents, &lookupEvent); },
                             repeats, [] {}, [&](int repeat) {
        double time = takeKernelTime(histogramEvent) + takeKernelTime(lookupEvent);
        for (cl_event& event : thresholdsEvents) time += takeKernelTime(event);
        thresholdsEvents.clear();
        if (repeat >= 0) kernelTimes.push_back(time);
        if (first.pixels.empty()) first = output;
        if (output.pixels != first.pixels) throw std::runtime_error("GPU multi-Otsu results differ between runs");
    });
    result.kernelUs = median(kernelTimes);
    result.thresholds = gpu.thresholds();
    result.differentPixels = differentPixels(reference, output);

    result.output = output;

    std::fill(output.pixels.begin(), output.pixels.end(), 1);
    gpu.upload(image, output);
    result.noCopyUs = measure([&] { gpu.compute(); gpu.finish(); }, repeats);
    gpu.download(output);
    if (output.pixels != result.output.pixels) throw std::runtime_error("GPU multi-Otsu without copying differs");
    return result;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_256.png", "results/multi_otsu_scaling.csv",
                                 {"--classes", "--verify-classes"});
        const int classes = parseNumber<int>(options.parameter("--classes", "3"));
        const int verifyClasses = parseNumber<int>(options.parameter("--verify-classes", "4"));
        Image image = loadImage(options.image);
        Histogram histogram = calculateHistogram(image);
        if (calculateHistogram(image, options.threads) != histogram) throw std::runtime_error("CPU histograms differ");

        checkRandomHistograms();
        checkImage(histogram, verifyClasses);
        const std::vector<int> thresholds = calculateMultiOtsuThresholds(histogram, classes);
        const Image expected = classifyDirectly(image, thresholds);
        std::cout << "Thresholds verified against exhaustive search (image: 2-" << verifyClasses
                  << " classes, 300 sparse histograms) and full dynamic programming (image and 100 dense histograms: 2-16 classes)\n";

        Image cpuResult;
        Image result;
        double cpu1 = measure([&] { cpuResult = multiOtsuSegmentation(image, classes); }, options.repeats,
                              [] {}, [&](int) {
            if (cpuResult.pixels != expected.pixels) throw std::runtime_error("Multi-Otsu segmentation is wrong");
        });
        double cpuN = measure([&] { result = multiOtsuSegmentation(image, classes, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != expected.pixels) throw std::runtime_error("CPU multi-Otsu results differ");
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);
        std::vector<int> searched;
        double search = measure([&] { searched = calculateMultiOtsuThresholds(histogram, classes); }, options.repeats);
        if (searched != thresholds) throw std::runtime_error("Threshold search is not repeatable");

        checkGPURandomImages();
        {
            MultiOtsuGPU gpu(image, classes);
            checkGPUThresholds(image, gpu, classes, "image");
        }
        GPUResult zeroCopy = measureGPU(image, classes, true, options.repeats, expected);
        std::cout << "GPU thresholds verified as near-optimal on the image and 100 random images (2-16 classes)\n";
        GPUResult gpu = measureGPU(image, classes, false, options.repeats, expected);
        if (zeroCopy.thresholds != gpu.thresholds || zeroCopy.output.pixels != gpu.output.pixels) {
            throw std::runtime_error("GPU multi-Otsu with and without zero-copy differs");
        }

        std::filesystem::path outputDirectory = "results/multi_otsu";
        std::filesystem::create_directories(outputDirectory);
        std::string imageName = std::filesystem::path(options.image).stem().string();
        saveImage((outputDirectory / (imageName + "_" + std::to_string(classes) + "_classes.png")).string(), cpuResult);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ','
            << image.width << ',' << image.height << ',' << classes << ',' << std::quoted(join(thresholds), '"', '"') << ','
            << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << search << ','
            << std::quoted(join(gpu.thresholds), '"', '"') << ',' << gpu.differentPixels << ','
            << gpu.kernelUs << ',' << gpu.totalUs << ',' << gpu.noCopyUs << ',' << zeroCopy.totalUs;
        appendCsv(options.output, "image,width,height,classes,thresholds,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,threshold_search_us,gpu_thresholds,gpu_different_pixels,gpu_kernel_us,gpu_total_us,gpu_no_copy_us,gpu_zero_copy_us", row.str());
        std::cout << "Thresholds CPU / GPU: " << join(thresholds) << " / " << join(gpu.thresholds)
                  << " (" << gpu.differentPixels << " different pixels)\n";
        std::cout << "Median CPU1 / CPU" << options.threads << " / threshold search only [us]: "
                  << cpu1 << " / " << cpuN << " / " << search << '\n';
        std::cout << "Median GPU total / GPU zero-copy / GPU without copying / GPU kernels [us]: "
                  << gpu.totalUs << " / " << zeroCopy.totalUs << " / " << gpu.noCopyUs << " / " << gpu.kernelUs << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
