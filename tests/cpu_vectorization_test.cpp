#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "negative.hpp"
#include "parallel_utils.hpp"
#include "thresholding.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void negativePointers(const Image& image, Image& output, int threads) {
    const unsigned char* input = image.pixels.data();
    unsigned char* result = output.pixels.data();
    parallelRanges(image.pixels.size(), threads, [input, result](std::size_t begin, std::size_t end, int) {
        const unsigned char* __restrict source = input;
        unsigned char* __restrict destination = result;
        for (std::size_t i = begin; i < end; ++i) destination[i] = 255 - source[i];
    });
}

void thresholdPointers(Image& image, int threshold, int threads) {
    unsigned char* data = image.pixels.data();
    parallelRanges(image.pixels.size(), threads, [data, threshold](std::size_t begin, std::size_t end, int) {
        unsigned char* __restrict pixels = data;
        const int limit = threshold;
        for (std::size_t i = begin; i < end; ++i) pixels[i] = pixels[i] < limit ? 0 : 255;
    });
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_8192.png", "results/cpu_vectorization.csv");
        const Image image = loadImage(options.image);
        for (int threads : {1, options.threads}) {
            Image current = image;
            Image pointers = image;
            const double negativeUs = measure([&] { negativeParallel(image, current, threads); }, options.repeats);
            const double negativePointersUs = measure([&] { negativePointers(image, pointers, threads); }, options.repeats);
            if (current.pixels != pointers.pixels) throw std::runtime_error("The negative with local pointers gives a different result");
            const double thresholdUs = measure([&] { ThresholdParallel(current, 128, threads); }, options.repeats,
                                               [&] { current = image; }, [](int) {});
            const double thresholdPointersUs = measure([&] { thresholdPointers(pointers, 128, threads); }, options.repeats,
                                                       [&] { pointers = image; }, [](int) {});
            if (current.pixels != pointers.pixels) throw std::runtime_error("The threshold with local pointers gives a different result");
            std::ostringstream row;
            row << std::fixed << std::setprecision(1) << image.width << 'x' << image.height << ',' << threads << ',' << negativeUs << ','
                << negativePointersUs << ',' << thresholdUs << ',' << thresholdPointersUs << ',' << image.pixels.size();
            appendCsv(options.output, "image,threads,negative_us,negative_pointers_us,threshold_us,threshold_pointers_us,bytes", row.str());
            std::cout << image.width << 'x' << image.height << ", " << threads << " threads: negative " << negativeUs << " -> "
                      << negativePointersUs << " us, threshold " << thresholdUs << " -> " << thresholdPointersUs << " us" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
