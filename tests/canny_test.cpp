#include "canny.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Reference {
    Image edges;
    long long weakKept = 0;
};

Reference naiveCanny(const Image& image, float lowThreshold, float highThreshold) {
    const int width = image.width;
    const int height = image.height;
    auto at = [&](const std::vector<int>& values, int x, int y) { return values[static_cast<std::size_t>(y) * width + x]; };
    std::vector<int> pixels(image.pixels.begin(), image.pixels.end());
    std::vector<int> blurred = pixels;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) sum += (2 - std::abs(dx)) * (2 - std::abs(dy)) * at(pixels, x + dx, y + dy);
            }
            blurred[static_cast<std::size_t>(y) * width + x] = (sum + 8) / 16;
        }
    }
    std::vector<int> magnitude(pixels.size(), 0);
    std::vector<int> neighbourX(pixels.size(), 0);
    std::vector<int> neighbourY(pixels.size(), 0);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0;
            int gy = 0;
            for (int d = -1; d <= 1; ++d) {
                const int weight = d == 0 ? 2 : 1;
                gx += weight * (at(blurred, x + 1, y + d) - at(blurred, x - 1, y + d));
                gy += weight * (at(blurred, x + d, y + 1) - at(blurred, x + d, y - 1));
            }
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            magnitude[index] = gx * gx + gy * gy;
            const double ratio = 27146.0 / 65536.0;
            if (std::abs(gy) < ratio * std::abs(gx)) {
                neighbourX[index] = 1;
            } else if (std::abs(gx) <= ratio * std::abs(gy)) {
                neighbourY[index] = 1;
            } else {
                neighbourX[index] = (gx > 0) == (gy > 0) ? 1 : -1;
                neighbourY[index] = 1;
            }
        }
    }
    const double low = static_cast<double>(lowThreshold) * lowThreshold;
    const double high = static_cast<double>(highThreshold) * highThreshold;
    std::vector<unsigned char> state(pixels.size(), 0);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            const int dx = neighbourX[index];
            const int dy = neighbourY[index];
            const int value = magnitude[index];
            if (value > at(magnitude, x - dx, y - dy) && value >= at(magnitude, x + dx, y + dy)) {
                state[index] = value >= high ? 255 : value >= low ? 128 : 0;
            }
        }
    }
    std::queue<std::size_t> queue;
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] == 255) queue.push(i);
    }
    Reference reference;
    while (!queue.empty()) {
        const std::size_t index = queue.front();
        queue.pop();
        const int x = static_cast<int>(index % width);
        const int y = static_cast<int>(index / width);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                const std::size_t neighbour = static_cast<std::size_t>(ny) * width + nx;
                if (state[neighbour] == 128) {
                    state[neighbour] = 255;
                    ++reference.weakKept;
                    queue.push(neighbour);
                }
            }
        }
    }
    for (unsigned char& value : state) {
        if (value == 128) value = 0;
    }
    reference.edges = Image{width, height, state};
    return reference;
}

Image randomImage(int width, int height, std::mt19937& generator) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    std::uniform_int_distribution<int> value(0, 255);
    std::uniform_int_distribution<int> noise(-12, 12);
    const int blocksX = std::max(1, width / 9);
    const int blocksY = std::max(1, height / 7);
    std::vector<int> levels(static_cast<std::size_t>(blocksX + 1) * (blocksY + 1));
    for (int& level : levels) level = value(generator);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int block = (y * blocksY / std::max(1, height)) * (blocksX + 1) + x * blocksX / std::max(1, width);
            const int ring = ((x - width / 2) * (x - width / 2) + (y - height / 3) * (y - height / 3)) < width * height / 9 ? 40 : 0;
            image.pixels[static_cast<std::size_t>(y) * width + x] =
                static_cast<unsigned char>(std::clamp(levels[block] + ring + noise(generator), 0, 255));
        }
    }
    return image;
}

std::string describe(const Image& image, float low, float high) {
    std::ostringstream text;
    text << image.width << 'x' << image.height << ", thresholds " << low << '/' << high;
    return text.str();
}

void expectSame(const Image& expected, const Image& actual, const std::string& what) {
    if (expected.pixels == actual.pixels) return;
    for (std::size_t i = 0; i < expected.pixels.size(); ++i) {
        if (expected.pixels[i] != actual.pixels[i]) {
            throw std::runtime_error(what + " differs at (" + std::to_string(i % expected.width) + ", " +
                                     std::to_string(i / expected.width) + ")");
        }
    }
}

void checkImages(int threads) {
    std::mt19937 generator(2031);
    long long weakKept = 0;
    long long checked = 0;
    for (auto [width, height] : {std::pair{1, 1}, {2, 2}, {3, 3}, {5, 4}, {1, 40}, {40, 1}, {64, 64}, {300, 129}, {129, 300}, {257, 513}}) {
        const Image image = randomImage(width, height, generator);
        for (auto [low, high] : {std::pair{0.0f, 0.0f}, {20.0f, 40.0f}, {50.0f, 100.0f}, {100.0f, 100.0f}, {80.0f, 300.0f}}) {
            const std::string name = describe(image, low, high);
            const Reference reference = naiveCanny(image, low, high);
            weakKept += reference.weakKept;
            expectSame(reference.edges, canny(image, low, high), name + ": CPU 1 thread");
            for (Schedule schedule : {Schedule::Static, Schedule::Dynamic}) {
                ScopedParallelSettings scope({schedule, 8, nullptr});
                Image result = image;
                canny(image, result, low, high, threads);
                expectSame(reference.edges, result, name + ": CPU " + std::to_string(threads) + " threads");
            }
            CannyGPU gpu(image, low, high);
            Image gpuResult = image;
            gpu.run(image, gpuResult);
            expectSame(reference.edges, gpuResult, name + ": GPU");
            checked += static_cast<long long>(image.pixels.size());
        }
    }
    if (weakKept == 0) throw std::runtime_error("The random images never kept a weak edge, so hysteresis was not tested");

    Image step{20, 12, std::vector<unsigned char>(20 * 12, 0)};
    for (int y = 0; y < step.height; ++y) {
        for (int x = step.width / 2; x < step.width; ++x) step.pixels[static_cast<std::size_t>(y) * step.width + x] = 255;
    }
    const Image stepEdges = canny(step, 50.0f, 100.0f);
    for (int y = 1; y < step.height - 1; ++y) {
        const auto row = stepEdges.pixels.begin() + static_cast<std::ptrdiff_t>(y) * step.width;
        if (std::count(row, row + step.width, 255) != 1) {
            throw std::runtime_error("A vertical step edge is not exactly one pixel wide in row " + std::to_string(y));
        }
    }
    const Image flat{30, 20, std::vector<unsigned char>(30 * 20, 90)};
    const Image flatEdges = canny(flat, 0.0f, 0.0f);
    if (std::count(flatEdges.pixels.begin(), flatEdges.pixels.end(), 255) != 0) {
        throw std::runtime_error("A flat image has edges");
    }
    for (auto [low, high] : {std::pair{-1.0f, 10.0f}, {20.0f, 10.0f}}) {
        bool thrown = false;
        try {
            canny(flat, low, high);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        if (!thrown) throw std::runtime_error("Invalid thresholds were accepted");
    }
    std::cout << "CPU 1 thread, CPU " << threads << " threads (static and dynamic) and GPU match the naive version on "
              << checked << " pixels of random images; " << weakKept << " weak pixels were kept by hysteresis; "
              << "a step edge is one pixel wide\n";
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/canny_scaling.csv",
                                 {"--low", "--high"});
        const float low = parseNumber<float>(options.parameter("--low", "50"));
        const float high = parseNumber<float>(options.parameter("--high", "100"));
        checkImages(options.threads);

        const Image image = loadImage(options.image);
        Image cpuResult = image;
        Image result = image;
        const double cpu1 = measure([&] { canny(image, cpuResult, low, high, 1); }, options.repeats);
        const double cpuN = measure([&] { canny(image, result, low, high, options.threads); }, options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU results differ");
        });
        const double overheadN = measureThreadOverhead(options.threads, options.repeats);

        CannyGPU gpu(image, low, high);
        Image gpuResult = image;
        double kernelUs = 0.0;
        std::vector<double> kernelTimes;
        const double gpuTotal = measure([&] { gpu.run(image, gpuResult, &kernelUs); }, options.repeats, [] {}, [&](int repeat) {
            if (repeat >= 0) kernelTimes.push_back(kernelUs);
        });
        const double gpuKernel = median(kernelTimes);
        expectSame(cpuResult, gpuResult, "GPU on " + options.image);
        const auto edges = std::count(cpuResult.pixels.begin(), cpuResult.pixels.end(), 255);

        const std::string imageName = std::filesystem::path(options.image).stem().string();
        const std::filesystem::path outputDirectory = "results/canny";
        std::filesystem::create_directories(outputDirectory);
        std::ostringstream lowText;
        lowText << low << '_' << high;
        saveImage((outputDirectory / (imageName + "_canny_" + lowText.str() + ".png")).string(), cpuResult);

        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height
            << ',' << low << ',' << high << ',' << options.threads << ',' << options.repeats << ',' << cpu1 << ','
            << cpuN << ',' << overheadN << ',' << gpuKernel << ',' << gpuTotal << ',' << gpu.iterations() << ',' << edges;
        appendCsv(options.output, "image,width,height,low,high,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us,gpu_track_iterations,edge_pixels",
                  row.str());
        std::cout << "canny: median CPU1 / CPU" << options.threads << " / GPU total / GPU kernel [us]: " << cpu1 << " / "
                  << cpuN << " / " << gpuTotal << " / " << gpuKernel << ", GPU edge tracking iterations: "
                  << gpu.iterations() << ", edge pixels: " << edges << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
