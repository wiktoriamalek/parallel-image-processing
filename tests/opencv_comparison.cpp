#include "algorithms.hpp"
#include "common_benchmarking.hpp"
#include "convolution.hpp"
#include "image_io.hpp"
#include "median_filter.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Case {
    std::string name;
    Algorithm algorithm;
    Parameters parameters;
};

int filterIndex(const std::string& name) {
    const auto& filters = convolutionFilters();
    for (std::size_t i = 0; i < filters.size(); ++i) {
        if (filters[i].name == name) return static_cast<int>(i);
    }
    throw std::invalid_argument("Unknown filter: " + name);
}

std::vector<Case> cases() {
    std::vector<Case> list;
    list.push_back({"threshold", Algorithm::Threshold, {}});
    list.push_back({"otsu", Algorithm::Otsu, {}});
    for (const char* filter : {"sharpen", "box_blur"}) {
        Parameters parameters;
        parameters.filter = filterIndex(filter);
        list.push_back({std::string("convolution_") + filter, Algorithm::Convolution, parameters});
    }
    for (int radius = 1; radius <= 3; ++radius) {
        Parameters parameters;
        parameters.medianSort = static_cast<int>(MedianSort::Shared);
        parameters.medianRadius = radius;
        parameters.medianStrip = 8;
        const std::string size = std::to_string(2 * radius + 1);
        list.push_back({"median_" + size + "x" + size, Algorithm::MedianFilter, parameters});
    }
    Parameters canny;
    canny.cannyLow = 50.0f;
    canny.cannyHigh = 100.0f;
    list.push_back({"canny", Algorithm::Canny, canny});
    return list;
}

std::string modeName(Mode mode) {
    switch (mode) {
        case Mode::CPU1: return "project_cpu1";
        case Mode::CPUN: return "project_cpu_static";
        case Mode::CPUNDynamic: return "project_cpu_dynamic";
        case Mode::GPU: return "project_opencl";
        default: return "other";
    }
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/opencv_comparison_project.csv",
                                 {"--outputs"});
        const std::string outputs = options.parameter("--outputs", "results/opencv_comparison_outputs");
        std::filesystem::create_directories(outputs);
        const std::vector<std::string> images = {"images/cms_resolution_change/cms_512.png", "images/cms_resolution_change/cms_2048.png",
                                                 "images/cms_resolution_change/cms_8192.png"};
        const int threads = 11;
        for (const std::string& path : images) {
            const Image image = loadImage(path);
            std::cout << "\n" << path << " (" << image.width << "x" << image.height << ")" << std::endl;
            for (const Case& item : cases()) {
                for (Mode mode : {Mode::CPU1, Mode::CPUN, Mode::CPUNDynamic, Mode::GPU}) {
                    Image output;
                    const int modeThreads = mode == Mode::CPU1 ? 1 : threads;
                    const Timing timing = runAlgorithm(item.algorithm, mode, image, item.parameters, modeThreads, options.repeats, output);
                    if (mode == Mode::CPUN) {
                        saveImage(outputs + "/" + item.name + "_" + std::to_string(image.width) + ".png", output);
                    }
                    std::ostringstream row;
                    row << std::fixed << std::setprecision(3) << '"' << path << '"' << ',' << image.width << ',' << image.height << ','
                        << item.name << ',' << modeName(mode) << ',' << modeThreads << ',' << options.repeats << ',' << timing.totalUs
                        << ',' << timing.kernelUs;
                    appendCsv(options.output, "image,width,height,operation,implementation,threads,repeats,total_us,kernel_us", row.str());
                    std::cout << std::left << std::setw(22) << item.name << std::setw(22) << modeName(mode) << std::right << std::fixed
                              << std::setprecision(1) << std::setw(12) << timing.totalUs << " us" << std::setw(12) << timing.kernelUs
                              << " us" << std::endl;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
