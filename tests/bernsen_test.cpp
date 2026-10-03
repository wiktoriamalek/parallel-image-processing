#include "image_io.hpp"
#include "common_benchmarking.hpp"
#include "bernsen.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_512.png","results/bernsen_scaling.csv",{"--radius", "--contrast"});

    int radius = parseNumber<int>(options.parameter("--radius", "7"));
    int minContrast = parseNumber<int>(options.parameter("--contrast", "15"));

    Image image = loadImage(options.image);
    Image cpuResult = image;
    Image result = image;

    double cpu1 = measure([&] { // lambda can use through a reference variables mentioned!
        bernsenThreshold(image, cpuResult, radius, minContrast, 1);
    }, options.repeats);

    double cpuN = measure([&] {bernsenThreshold(image, result, radius, minContrast, options.threads);}, options.repeats);

    if (result.pixels != cpuResult.pixels) {
        throw std::runtime_error("CPU Bernsen results differ");
    }

    std::cout << "CPU 1 thread: " << cpu1 << " us = "<< cpu1/1e3<< "ms\n";
    std::cout << "CPU " << options.threads << " threads: " << cpuN << " us = "<< cpuN/1e3<< "ms\n";
    saveImage( "results/bernsen_4t_rep3_rad15_cont15_result.png", result);
        return 0;
    }

    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
