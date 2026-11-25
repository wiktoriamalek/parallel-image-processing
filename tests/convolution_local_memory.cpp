#include "algorithms.hpp"
#include "convolution.hpp"
#include "image_io.hpp"
#include "metal_algorithms.hpp"
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int filterIndex(const std::string& name) {
    const auto& filters = convolutionFilters();
    for (std::size_t i = 0; i < filters.size(); ++i) {
        if (filters[i].name == name) return static_cast<int>(i);
    }
    throw std::runtime_error("Unknown filter " + name);
}

Image randomImage(int width, int height, std::mt19937& generator) {
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);
    std::uniform_int_distribution<int> value(0, 255);
    for (auto& pixel : image.pixels) pixel = static_cast<unsigned char>(value(generator));
    return image;
}

Image emptyLike(const Image& image) {
    Image output;
    output.width = image.width;
    output.height = image.height;
    output.pixels.assign(image.pixels.size(), 0);
    return output;
}

Image runOpenCL(const Image& image, const ConvolutionFilter& filter, bool localTile) {
    Image output = emptyLike(image);
    ConvolutionGPU gpu(image, filter.kernel, filter.absoluteValue, localTile);
    gpu.run(image, output);
    return output;
}

Image runMetal(const Image& image, int filter, bool localTile) {
    Parameters parameters;
    parameters.filter = filter;
    parameters.convolutionLocalTile = localTile;
    Image output = emptyLike(image);
    MetalProcessor gpu(Algorithm::Convolution, image, parameters);
    gpu.run(image, output);
    return output;
}

std::size_t differences(const Image& a, const Image& b) {
    if (a.pixels.size() != b.pixels.size()) return a.pixels.size() + b.pixels.size();
    std::size_t count = 0;
    for (std::size_t i = 0; i < a.pixels.size(); ++i) count += a.pixels[i] != b.pixels[i];
    return count;
}

std::size_t checkAll(const Image& image, const std::string& label) {
    std::size_t failures = 0;
    const auto& filters = convolutionFilters();
    for (std::size_t f = 0; f < filters.size(); ++f) {
        Image reference = image;
        convolution(image, filters[f].kernel, filters[f].absoluteValue, reference, 1);
        const std::size_t openclGlobal = differences(reference, runOpenCL(image, filters[f], false));
        const std::size_t openclLocal = differences(reference, runOpenCL(image, filters[f], true));
        const std::size_t metalGlobal = differences(reference, runMetal(image, static_cast<int>(f), false));
        const std::size_t metalLocal = differences(reference, runMetal(image, static_cast<int>(f), true));
        if (openclGlobal || openclLocal || metalGlobal || metalLocal) {
            ++failures;
            std::printf("DIFF %s %s: OpenCL global %zu, OpenCL local %zu, Metal global %zu, Metal local %zu\n",
                        label.c_str(), filters[f].name.c_str(), openclGlobal, openclLocal, metalGlobal, metalLocal);
        }
    }
    return failures;
}
}

int main(int argc, char* argv[]) {
    const int repeats = argc > 1 ? std::stoi(argv[1]) : 30;
    const int rounds = argc > 2 ? std::stoi(argv[2]) : 3;

    std::mt19937 generator(12345);
    std::size_t failures = 0;
    const std::vector<std::pair<int, int>> sizes = {{1, 1}, {2, 3}, {3, 3}, {16, 16}, {17, 5}, {31, 33},
                                                    {100, 37}, {257, 129}, {1000, 662}};
    for (const auto& [width, height] : sizes) {
        failures += checkAll(randomImage(width, height, generator), std::to_string(width) + "x" + std::to_string(height));
    }
    failures += checkAll(loadImage("images/cms_resolution_change/cms_512.png"), "cms_512");
    std::printf("correctness: %zu failing filter/image combinations\n", failures);
    if (failures) return 1;

    const std::vector<std::string> images = {"images/cms_resolution_change/cms_256.png", "images/cms_resolution_change/cms_512.png",
                                             "images/cms_resolution_change/cms_1024.png", "images/cms_resolution_change/cms_2048.png",
                                             "images/cms_resolution_change/cms_4096.png", "images/cms_resolution_change/cms_8192.png"};
    const std::vector<std::string> filterNames = {"box_blur", "sobel_h", "laplacian"};
    std::ofstream csv("results/convolution_local_memory.csv");
    csv << "image,width,height,filter,backend,variant,round,repeats,kernel_us,total_us\n";
    for (const std::string& path : images) {
        const Image image = loadImage(path);
        for (const std::string& name : filterNames) {
            Parameters parameters;
            parameters.filter = filterIndex(name);
            for (int round = 0; round < rounds; ++round) {
                for (const Mode mode : {Mode::GPU, Mode::Metal}) {
                    const bool localFirst = round % 2 == 1;
                    for (const bool localTile : {localFirst, !localFirst}) {
                        parameters.convolutionLocalTile = localTile;
                        Image output = emptyLike(image);
                        const Timing timing = runAlgorithm(Algorithm::Convolution, mode, image, parameters, 1, repeats, output);
                        const char* backend = mode == Mode::GPU ? "opencl" : "metal";
                        const char* variant = localTile ? "local" : "global";
                        csv << path << ',' << image.width << ',' << image.height << ',' << name << ',' << backend << ','
                            << variant << ',' << round << ',' << repeats << ',' << timing.kernelUs << ',' << timing.totalUs << '\n';
                        std::printf("%dx%d %-9s %-6s %-6s round %d: kernel %9.1f us, total %9.1f us\n", image.width, image.height,
                                    name.c_str(), backend, variant, round, timing.kernelUs, timing.totalUs);
                    }
                }
            }
        }
    }
    return 0;
}
