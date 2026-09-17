#include "otsu.hpp"
#include <iostream>
#include <numeric>

int main() {
    Image image;
    image.width = 3;
    image.height = 2;
    image.pixels = {0, 0, 50, 50, 50, 255};

    Histogram histogram = calculateHistogram(image);
    calculateOtsuThreshold(histogram);
    std::size_t total = std::accumulate(
        histogram.begin(), histogram.end(), std::size_t{0}
    );

    std::cout << "histogram[0]: " << histogram[0] << '\n';
    std::cout << "histogram[50]: " << histogram[50] << '\n';
    std::cout << "histogram[255]: " << histogram[255] << '\n';
    std::cout << "Total: " << total << '\n';

    if (histogram[0] != 2 || histogram[50] != 3 ||
        histogram[255] != 1 || total != image.pixels.size()) {
        std::cerr << "Histogram test failed\n";
        return 1;
        }

    return 0;
}