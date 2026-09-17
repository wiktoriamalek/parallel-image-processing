#include "otsu.hpp"
#include <iostream>
#include <numeric>
#include "image_io.hpp"

int main() {
    Image image = loadImage("images/cms_resolution_change/cms_256.png");

    Histogram histogram = calculateHistogram(image);
    //calculateOtsuThreshold(histogram);
    std::size_t total = std::accumulate(histogram.begin(), histogram.end(), std::size_t{0});
    Image segmented = otsuSegmentation(image);
    saveImage("results/cms_otsu_256.png", segmented);


    std::cout<<'\n';

    return 0;
}