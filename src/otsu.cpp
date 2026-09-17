#include "otsu.hpp"
#include <iostream>

Histogram calculateHistogram(const Image& image) {
    Histogram histogram{}; // resets all 256 counters to zero.

    for (unsigned char pixel : image.pixels) {
        ++histogram[pixel]; // if pixel is e.g 50, we increase histogram[50]
    }
    return histogram;
}

int calculateOtsuThreshold(const Histogram& histogram) {
    std::size_t totalPixels = 0;
    double totalSum = 0.0;

    std::size_t backgroundPixels = 0; // number of pixels with brightness <= current threshold t
    double backgroundSum = 0.0; // brightness sum of those pixels

    for (int i =0; i<256; i++) {
        totalPixels += histogram[i];
        totalSum += i * histogram[i];
    }

    double maxVariance = 0.0;
    int bestThreshold = 0;
    for (int t =0; t<256; t++) {
        backgroundPixels += histogram[t];
        backgroundSum += t * histogram[t];

        std::size_t foregroundPixels = totalPixels - backgroundPixels;
        double foregroundSum = totalSum - backgroundSum;

        if (foregroundPixels == 0 || backgroundPixels == 0) {
            continue;
        }

        double foregroundAverage = foregroundSum / foregroundPixels;
        double backgroundAverage = backgroundSum / backgroundPixels;

        double backgroundWeight = static_cast<double>(backgroundPixels) / totalPixels;
        double foregroundWeight = static_cast<double>(foregroundPixels) / totalPixels;
        double difference = backgroundAverage - foregroundAverage;
        double betweenClassVariance = backgroundWeight * foregroundWeight * (difference * difference);
        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            bestThreshold = t;
        }
    }

    std::cout << "Total pixels: " << totalPixels << '\n';
    std::cout << "Total sum: " << totalSum << '\n';
    std::cout << "Best threshold: " << bestThreshold << '\n';
    std::cout << "Maximum variance: " << maxVariance << '\n';


    return bestThreshold;
}