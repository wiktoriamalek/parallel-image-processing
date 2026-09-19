#include "otsu.hpp"
#include <iostream>
#include <numeric>
#include "image_io.hpp"
#include <array>

int main() {
    Image image = loadImage("images/cms_resolution_change/cms_4096.png");

    Histogram histogram = calculateHistogram(image);
    Histogram histogram2 = calculateHistogram2Threads(image);
    Histogram histogramGPU = calculateHistogramGPU(image);

    if (histogramGPU != histogram) {
        std::cout << "GPU histogram differs from CPU!" << std::endl;
        return 1;
    }

    std::cout << "GPU and CPU histograms are the same!" << std::endl;

    if (histogram2 == histogram)
        std::cout << "CPU 1 and 2 threads histograms are the same!"<< std::endl;
    //calculateOtsuThreshold(histogram);
    std::size_t total = std::accumulate(histogram.begin(), histogram.end(), std::size_t{0});
    Image segmented = otsuSegmentation(image);
    Image segmented2Threads = otsuSegmentation2Threads(image);

    if (segmented.pixels == segmented2Threads.pixels)
        std::cout << "Masks are the same!"<< std::endl;
    else {
        std::cout << "Masks are not the same!"<< std::endl;
        return 1;
    }
    //saveImage("results/cms_otsu_256.png", segmented);


    std::cout<<'\n';

    return 0;
}