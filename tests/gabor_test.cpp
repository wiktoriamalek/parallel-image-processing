#include "common_benchmarking.hpp"
#include "gabor.hpp"
#include "image_io.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
struct BankCase {
    float sigma;
    float wavelength;
    float gamma;
    int orientations;
};

Image naiveGabor(const Image& image, const GaborBank& bank) {
    Image result{image.width, image.height, std::vector<unsigned char>(image.pixels.size())};
    const int radius = bank.radius;
    const int size = bank.size();
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float best = 0.0f;
            for (int orientation = 0; orientation < bank.orientations; ++orientation) {
                float even = 0.0f;
                float odd = 0.0f;
                for (int ky = 0; ky < size; ++ky) {
                    for (int kx = 0; kx < size; ++kx) {
                        const int sourceX = std::clamp(x + kx - radius, 0, image.width - 1);
                        const int sourceY = std::clamp(y + ky - radius, 0, image.height - 1);
                        const float value = image.pixels[static_cast<std::size_t>(sourceY) * image.width + sourceX];
                        const std::size_t tap = 2 * ((static_cast<std::size_t>(orientation) * size + ky) * size + kx);
                        even += value * bank.weights[tap];
                        odd += value * bank.weights[tap + 1];
                    }
                }
                float magnitude = even * even;
                magnitude += odd * odd;
                best = orientation == 0 ? magnitude : std::max(best, magnitude);
            }
            const double level = std::floor(std::sqrt(static_cast<double>(best)));
            result.pixels[static_cast<std::size_t>(y) * image.width + x] = static_cast<unsigned char>(std::min(level, 255.0));
        }
    }
    return result;
}

Image randomImage(int width, int height, std::mt19937& generator) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    std::uniform_int_distribution<int> value(0, 255);
    std::uniform_int_distribution<int> noise(-20, 20);
    const int stripe = std::uniform_int_distribution<int>(3, 12)(generator);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int base = ((x + 2 * y) / stripe) % 2 ? 200 : 60;
            const int pixel = (x * 7 + y * 13) % 31 == 0 ? value(generator) : base + noise(generator);
            image.pixels[static_cast<std::size_t>(y) * width + x] = static_cast<unsigned char>(std::clamp(pixel, 0, 255));
        }
    }
    return image;
}

std::string describe(const Image& image, const BankCase& parameters, const GaborBank& bank) {
    std::ostringstream text;
    text << image.width << 'x' << image.height << ", sigma " << parameters.sigma << ", lambda " << parameters.wavelength
         << ", gamma " << parameters.gamma << ", " << parameters.orientations << " orientations, radius " << bank.radius;
    return text.str();
}

void expectSame(const Image& expected, const Image& actual, const std::string& what) {
    if (expected.width != actual.width || expected.height != actual.height) throw std::runtime_error(what + " has a different size");
    if (expected.pixels == actual.pixels) return;
    std::size_t different = 0;
    int largest = 0;
    std::size_t first = expected.pixels.size();
    for (std::size_t i = 0; i < expected.pixels.size(); ++i) {
        const int delta = std::abs(expected.pixels[i] - actual.pixels[i]);
        if (delta == 0) continue;
        if (first == expected.pixels.size()) first = i;
        ++different;
        largest = std::max(largest, delta);
    }
    throw std::runtime_error(what + " differs on " + std::to_string(different) + " pixels (max " + std::to_string(largest) +
                             "), first at (" + std::to_string(first % expected.width) + ", " +
                             std::to_string(first / expected.width) + ")");
}

double meanResponse(const Image& image, int margin) {
    double sum = 0.0;
    long long count = 0;
    for (int y = margin; y < image.height - margin; ++y) {
        for (int x = margin; x < image.width - margin; ++x) {
            sum += image.pixels[static_cast<std::size_t>(y) * image.width + x];
            ++count;
        }
    }
    return sum / static_cast<double>(count);
}

Image grating(int width, int height, float wavelength, double theta) {
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
    const double pi = std::acos(-1.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double along = x * std::cos(theta) + y * std::sin(theta);
            image.pixels[static_cast<std::size_t>(y) * width + x] =
                static_cast<unsigned char>(std::lround(128.0 + 100.0 * std::cos(2.0 * pi * along / wavelength)));
        }
    }
    return image;
}

void expectInvalid(const std::function<void()>& operation, const std::string& what) {
    bool thrown = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    if (!thrown) throw std::runtime_error(what + " was accepted");
}

void checkImages(int threads) {
    std::mt19937 generator(20261122);
    long long checked = 0;
    const std::vector<std::pair<std::vector<std::pair<int, int>>, BankCase>> groups = {
        {{{1, 1}, {2, 3}, {17, 5}, {1, 40}, {40, 1}, {64, 64}, {300, 129}, {129, 300}, {2100, 9}}, {0.3f, 2.0f, 1.0f, 1}},
        {{{1, 1}, {5, 4}, {33, 17}, {64, 64}, {300, 129}, {129, 300}}, {1.0f, 3.0f, 1.0f, 8}},
        {{{3, 3}, {31, 47}, {64, 64}, {300, 129}, {1500, 30}}, {2.0f, 6.0f, 0.5f, 4}},
        {{{1, 1}, {7, 70}, {96, 80}, {150, 64}}, {3.0f, 9.0f, 0.4f, 3}},
        {{{2, 2}, {65, 67}, {130, 64}}, {6.0f, 16.0f, 0.6f, 16}},
        {{{40, 30}, {100, 70}}, {10.0f, 30.0f, 1.0f, 2}},
    };
    for (const auto& [sizes, parameters] : groups) {
        const GaborBank bank = gaborBank(parameters.sigma, parameters.wavelength, parameters.gamma, parameters.orientations);
        for (auto [width, height] : sizes) {
            const Image image = randomImage(width, height, generator);
            const std::string name = describe(image, parameters, bank);
            const Image reference = naiveGabor(image, bank);
            expectSame(reference, gaborFilter(image, bank), name + ": CPU 1 thread");
            for (Schedule schedule : {Schedule::Static, Schedule::Dynamic}) {
                ScopedParallelSettings scope({schedule, 8, nullptr});
                Image result;
                gaborFilter(image, bank, result, threads);
                expectSame(reference, result, name + ": CPU " + std::to_string(threads) + " threads");
            }
            GaborGPU gpu(image, bank);
            Image gpuResult;
            gpu.run(image, gpuResult);
            expectSame(reference, gpuResult, name + ": GPU");
            checked += static_cast<long long>(image.pixels.size());
        }
    }

    const GaborBank reusedBank = gaborBank(2.0f, 6.0f, 0.5f, 4);
    Image reused = randomImage(200, 150, generator);
    GaborGPU reusedGPU(reused, reusedBank);
    Image reusedOutput;
    for (int frame = 0; frame < 8; ++frame) {
        reused = randomImage(200, 150, generator);
        reusedGPU.run(reused, reusedOutput);
        expectSame(naiveGabor(reused, reusedBank), reusedOutput, "reused GPU buffers, frame " + std::to_string(frame));
    }

    const Image flat{50, 40, std::vector<unsigned char>(50 * 40, 200)};
    for (const BankCase& parameters : {BankCase{1.0f, 3.0f, 1.0f, 8}, BankCase{2.0f, 6.0f, 0.5f, 4}, BankCase{6.0f, 16.0f, 0.6f, 16}}) {
        const Image response = gaborFilter(flat, gaborBank(parameters.sigma, parameters.wavelength, parameters.gamma, parameters.orientations));
        if (*std::max_element(response.pixels.begin(), response.pixels.end()) != 0) {
            throw std::runtime_error("A flat image gives a nonzero Gabor response");
        }
    }

    const double pi = std::acos(-1.0);
    const GaborBank single = gaborBank(2.0f, 6.0f, 0.5f, 1);
    const int margin = single.radius;
    const double matched = meanResponse(gaborFilter(grating(160, 160, 6.0f, 0.0), single), margin);
    const double crossed = meanResponse(gaborFilter(grating(160, 160, 6.0f, pi / 2), single), margin);
    const double slower = meanResponse(gaborFilter(grating(160, 160, 24.0f, 0.0), single), margin);
    if (matched < 60.0 || crossed > 0.1 * matched || slower > 0.5 * matched) {
        throw std::runtime_error("Gabor selectivity check failed: matched " + std::to_string(matched) + ", rotated " +
                                 std::to_string(crossed) + ", wavelength x4 " + std::to_string(slower));
    }

    expectInvalid([] { gaborBank(0.0f, 6.0f, 0.5f, 4); }, "sigma 0");
    expectInvalid([] { gaborBank(2.0f, 1.5f, 0.5f, 4); }, "wavelength below 2");
    expectInvalid([] { gaborBank(2.0f, 6.0f, 0.0f, 4); }, "gamma 0");
    expectInvalid([] { gaborBank(2.0f, 6.0f, 0.5f, 0); }, "0 orientations");
    expectInvalid([] { gaborBank(2.0f, 6.0f, 0.5f, gaborMaxOrientations + 1); }, "too many orientations");
    expectInvalid([] { gaborBank(11.0f, 30.0f, 1.0f, 4); }, "window radius 33");
    expectInvalid([] { gaborBank(std::nanf(""), 6.0f, 0.5f, 4); }, "NaN sigma");

    std::cout << "CPU 1 thread, CPU " << threads << " threads (static and dynamic) and GPU match the naive Gabor filter on "
              << checked << " pixels; a flat image gives 0; one orientation (lambda 6) gives " << std::fixed
              << std::setprecision(1) << matched << " on a matched grating, " << crossed << " on the same grating rotated by 90 deg and "
              << slower << " on a 4x longer wavelength\n";
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/gabor_scaling.csv",
                                 {"--sigma", "--wavelength", "--gamma", "--orientations", "--check"});
        const float sigma = parseNumber<float>(options.parameter("--sigma", "2"));
        const float wavelength = parseNumber<float>(options.parameter("--wavelength", "6"));
        const float gamma = parseNumber<float>(options.parameter("--gamma", "0.5"));
        const int orientations = parseNumber<int>(options.parameter("--orientations", "4"));
        const bool check = parseNumber<int>(options.parameter("--check", "1")) != 0;
        if (check) checkImages(options.threads);

        const GaborBank bank = gaborBank(sigma, wavelength, gamma, orientations);
        const Image image = loadImage(options.image);
        Image cpuResult;
        Image result;
        const double cpu1 = measure([&] { gaborFilter(image, bank, cpuResult, 1); }, options.repeats);
        const double cpuN = measure([&] { gaborFilter(image, bank, result, options.threads); }, options.repeats, [] {}, [&](int) {
            if (result.pixels != cpuResult.pixels) throw std::runtime_error("CPU results differ");
        });
        const double overheadN = measureThreadOverhead(options.threads, options.repeats);

        GaborGPU gpu(image, bank);
        Image gpuResult;
        std::vector<double> kernelTimes;
        cl_event kernelEvent = nullptr;
        const double gpuTotal = measure([&] { gpu.run(image, gpuResult, &kernelEvent); }, options.repeats, [] {}, [&](int repeat) {
            const double time = kernelTimeUs(kernelEvent);
            checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
            if (repeat >= 0) kernelTimes.push_back(time);
        });
        const double gpuKernel = median(kernelTimes);
        expectSame(cpuResult, gpuResult, "GPU on " + options.image);

        const std::string imageName = std::filesystem::path(options.image).stem().string();
        const std::filesystem::path outputDirectory = "results/gabor";
        std::filesystem::create_directories(outputDirectory);
        std::ostringstream parameters;
        parameters << "s" << sigma << "_l" << wavelength << "_g" << gamma << "_n" << orientations;
        saveImage((outputDirectory / (imageName + "_gabor_" + parameters.str() + ".png")).string(), cpuResult);

        const double multiplications = 2.0 * bank.taps() * bank.orientations * static_cast<double>(image.pixels.size());
        std::ostringstream row;
        row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height
            << ',' << sigma << ',' << wavelength << ',' << gamma << ',' << orientations << ',' << bank.radius << ','
            << options.threads << ',' << options.repeats << ',' << cpu1 << ',' << cpuN << ',' << overheadN << ','
            << gpuKernel << ',' << gpuTotal;
        appendCsv(options.output, "image,width,height,sigma,wavelength,gamma,orientations,radius,threads,repeats,cpu_1t_us,cpu_nt_us,overhead_nt_us,gpu_kernel_us,gpu_total_us",
                  row.str());
        std::cout << "gabor " << parameters.str() << ", window " << bank.size() << 'x' << bank.size() << ": median CPU1 / CPU"
                  << options.threads << " / GPU total / GPU kernel [us]: " << cpu1 << " / " << cpuN << " / " << gpuTotal << " / "
                  << gpuKernel << "; GFLOP/s CPU1 / CPU" << options.threads << " / GPU kernel: "
                  << 2e-3 * multiplications / cpu1 << " / " << 2e-3 * multiplications / cpuN << " / "
                  << 2e-3 * multiplications / gpuKernel << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
