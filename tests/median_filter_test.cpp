#include "median_filter.hpp"
#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "opencl_utils.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

namespace {
Image randomImage(int width, int height, std::mt19937& generator) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    std::uniform_int_distribution<int> value(0, 255);
    std::uniform_int_distribution<int> kind(0, 9);
    const int flat = value(generator);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int choice = kind(generator);
            const int pixel = choice == 0 ? 0 : choice == 1 ? 255 : (x / 4 + y / 3) % 3 == 0 ? flat : value(generator);
            image.pixels[static_cast<std::size_t>(y) * width + x] = static_cast<unsigned char>(pixel);
        }
    }
    return image;
}

Image keepNonImpulses(const Image& image, Image median) {
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        if (image.pixels[i] != 0 && image.pixels[i] != 255) median.pixels[i] = image.pixels[i];
    }
    return median;
}

Image expectedMedian(const Image& image, int radius, bool switching) {
    Image median = image;
    medianFilterParallel(image, median, 1, MedianSort::Standard, radius);
    return switching ? keepNonImpulses(image, median) : median;
}

void checkRandomImages(MedianSort sort, int radius, int strip, bool switching, const MedianNetwork& network, int threads) {
    std::mt19937 generator(2026);
    const int size = 2 * radius + 1;
    for (auto [width, height] : {std::pair{1, 1}, {size, size + 1}, {67, 45}, {300, 129}}) {
        const Image image = randomImage(width, height, generator);
        const Image expected = expectedMedian(image, radius, switching);
        Image cpu = image;
        if (sort == MedianSort::Standard) medianFilterParallel(image, cpu, threads, sort, radius, 1, switching);
        else medianFilterParallel(image, cpu, threads, network, switching);
        const std::string name = medianSortName(sort) + ' ' + std::to_string(width) + 'x' + std::to_string(height);
        if (cpu.pixels != expected.pixels) throw std::runtime_error(name + ": CPU differs from std::nth_element");
        MedianFilterGPU gpu(image, sort, radius, strip, switching);
        Image gpuResult = image;
        gpu.run(image, gpuResult);
        if (gpuResult.pixels != expected.pixels) throw std::runtime_error(name + ": GPU differs from std::nth_element");
    }
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/nuclear_noisy.png", "results/median_filter_variants.csv",
                                 {"--sort", "--radius", "--strip", "--mode"});
        const std::string sortName = options.parameter("--sort", "standard");
        const MedianSort sort = parseMedianSort(sortName);
        const int radius = parseNumber<int>(options.parameter("--radius", "1"));
        const std::string defaultStrip = sort != MedianSort::Shared ? "1" : radius == 1 ? "8" : "16";
        const int strip = parseNumber<int>(options.parameter("--strip", defaultStrip));
        const std::string mode = options.parameter("--mode", "plain");
        if (mode != "plain" && mode != "switching") throw std::invalid_argument("--mode must be plain or switching");
        const bool switching = mode == "switching";
        if (radius < 1) throw std::invalid_argument("--radius must be at least 1");
        if (sort == MedianSort::Standard && strip != 1) throw std::invalid_argument("--strip is only for the shared sort");
        const MedianNetwork network = sort == MedianSort::Standard ? MedianNetwork{} : buildMedianNetwork(sort, radius, strip);
        checkRandomImages(sort, radius, strip, switching, network, options.threads);

        Image image = loadImage(options.image);
        auto cpuFilter = [&](Image& output, int threads) {
            if (sort == MedianSort::Standard) medianFilterParallel(image, output, threads, sort, radius, 1, switching);
            else medianFilterParallel(image, output, threads, network, switching);
        };
        Image result = image; // borders are not filtered, so both results start as a copy of the image
        Image parallelResult = image;
        const Image referenceResult = expectedMedian(image, radius, switching); // always std::nth_element, so every other variant is checked against it

        double cpu1 = measure([&] { cpuFilter(result, 1); }, options.repeats);
        if (result.pixels != referenceResult.pixels) {
            throw std::runtime_error(sortName + " and std::nth_element results differ!");
        }
        double cpuN = measure([&] { cpuFilter(parallelResult, options.threads); },
                              options.repeats, [] {}, [&](int) {
            if (result.pixels != parallelResult.pixels) {
                throw std::runtime_error("CPU 1T and CPU many threads result differ!");
            }
        });
        double overheadN = measureThreadOverhead(options.threads, options.repeats);

        MedianFilterGPU gpu(image, sort, radius, strip, switching);
        Image gpuResult = image;
        cl_event event = nullptr;
        std::vector<double> kernelTimes;
        double gpuTotal = measure([&] { gpu.run(image, gpuResult, &event); }, options.repeats, [] {}, [&](int repeat) {
            double time = kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
            if (result.pixels != gpuResult.pixels) {
                throw std::runtime_error("CPU and GPU results differ!");
            }
        });
        double gpuKernel = median(kernelTimes);

        const int size = 2 * radius + 1;
        const std::string window = std::to_string(size) + 'x' + std::to_string(size);
        std::filesystem::create_directories("results/median_filter");
        saveImage("results/median_filter/" + std::filesystem::path(options.image).stem().string() + "_median" +
                  (switching ? "_switching" : "") + (radius == 1 ? "" : "_" + window) + ".png", result);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << sortName << ',' << radius << ','
            << strip << ',' << mode << ',' << image.width << ',' << image.height << ',' << options.threads << ',' << options.repeats << ','
            << cpu1 << ',' << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal << ',';
        if (sort != MedianSort::Standard) {
            row << static_cast<double>(network.nominalComparators) / strip << ','
                << static_cast<double>(network.minMaxOperations()) / strip;
        } else {
            row << ',';
        }
        appendCsv(options.output, "image,sort,radius,strip,mode,width,height,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,"
                                  "gpu_kernel_us,gpu_total_us,comparators_per_pixel,min_max_per_pixel", row.str());
        std::cout << sortName << ' ' << mode << ' ' << window << " strip " << strip << ", median CPU1 / CPU" << options.threads
                  << " / GPU total / GPU kernel [us]: " << cpu1 << " / " << cpuN << " / " << gpuTotal << " / " << gpuKernel;
        if (sort != MedianSort::Standard) {
            std::cout << ", comparators per pixel " << static_cast<double>(network.nominalComparators) / strip
                      << ", min/max per pixel after pruning " << static_cast<double>(network.minMaxOperations()) / strip;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
