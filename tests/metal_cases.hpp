#pragma once

#include "algorithm_parameters.hpp"
#include "bernsen.hpp"
#include "canny.hpp"
#include "contrast.hpp"
#include "convolution.hpp"
#include "distortion.hpp"
#include "gabor.hpp"
#include "median_filter.hpp"
#include "negative.hpp"
#include "otsu.hpp"
#include "peaks_valleys_detection.hpp"
#include "region_growing.hpp"
#include "resize.hpp"
#include "thresholding.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct MetalCase {
    std::string name;
    Algorithm algorithm;
    Parameters parameters;
};

inline std::vector<MetalCase> metalCases(bool variants, const Parameters& base = {}) {
    std::vector<MetalCase> cases{
        {"threshold", Algorithm::Threshold, base}, {"negative", Algorithm::Negative, base},
        {"contrast", Algorithm::Contrast, base}, {"otsu", Algorithm::Otsu, base},
        {"multi_otsu", Algorithm::MultiOtsu, base}, {"peaks_valleys", Algorithm::PeaksValleys, base},
        {"bernsen", Algorithm::Bernsen, base}, {"niblack", Algorithm::Niblack, base},
        {"sauvola", Algorithm::Sauvola, base}, {"convolution", Algorithm::Convolution, base},
        {"median", Algorithm::MedianFilter, base}, {"distortion", Algorithm::Distortion, base},
        {"resize", Algorithm::Resize, base}, {"region_growing", Algorithm::RegionGrowing, base},
        {"canny", Algorithm::Canny, base}, {"gabor", Algorithm::Gabor, base}
    };
    if (!variants) return cases;
    for (int filter = 1; filter < static_cast<int>(convolutionFilters().size()); ++filter) {
        Parameters p = base;
        p.filter = filter;
        cases.push_back({"convolution_" + std::to_string(filter), Algorithm::Convolution, p});
    }
    for (int sort = 0; sort < 4; ++sort) {
        for (int radius = 1; radius <= 7; ++radius) {
            for (bool switching : {false, true}) {
                Parameters p = base;
                p.medianSort = sort;
                p.medianRadius = radius;
                p.medianSwitching = switching;
                cases.push_back({"median_" + std::to_string(sort) + "_r" + std::to_string(radius) +
                                 (switching ? "_switching" : ""), Algorithm::MedianFilter, p});
            }
        }
    }
    Parameters p = base;
    p.distortionModel = 1;
    p.k1 = 0.12f;
    p.k2 = -0.03f;
    p.k3 = 0.01f;
    p.p1 = 0.02f;
    p.p2 = -0.01f;
    cases.push_back({"brown_conrady", Algorithm::Distortion, p});
    p = base;
    p.bilinear = false;
    cases.push_back({"resize_nearest", Algorithm::Resize, p});
    p = base;
    p.scale = 0.6f;
    cases.push_back({"resize_down", Algorithm::Resize, p});
    p = base;
    p.localTiles = false;
    cases.push_back({"region_growing_pixel", Algorithm::RegionGrowing, p});
    for (int classes : {2, 4, 16, 256}) {
        p = base;
        p.classes = classes;
        cases.push_back({"multi_otsu_" + std::to_string(classes), Algorithm::MultiOtsu, p});
    }
    for (int classes : {2, 4, 16, 256}) {
        p = base;
        p.maxClasses = classes;
        cases.push_back({"peaks_valleys_" + std::to_string(classes), Algorithm::PeaksValleys, p});
    }
    for (auto [sigma, wavelength, gamma, orientations] : {std::tuple{0.3f, 2.0f, 1.0f, 1}, {1.0f, 3.0f, 1.0f, 8},
                                                          {3.0f, 9.0f, 0.4f, 3}, {6.0f, 16.0f, 0.6f, 16}}) {
        p = base;
        p.gaborSigma = sigma;
        p.gaborWavelength = wavelength;
        p.gaborGamma = gamma;
        p.gaborOrientations = orientations;
        cases.push_back({"gabor_r" + std::to_string(gaborRadius(sigma, gamma)) + "_n" + std::to_string(orientations),
                         Algorithm::Gabor, p});
    }
    for (int radius : {0, 1, 31, 127}) {
        p = base;
        p.radius = radius;
        cases.push_back({"bernsen_r" + std::to_string(radius), Algorithm::Bernsen, p});
        cases.push_back({"niblack_r" + std::to_string(radius), Algorithm::Niblack, p});
        cases.push_back({"sauvola_r" + std::to_string(radius), Algorithm::Sauvola, p});
    }
    return cases;
}

inline Image metalCPUReference(const MetalCase& test, const Image& image, int threads = 1) {
    const Parameters& p = test.parameters;
    Image output = image;
    switch (test.algorithm) {
        case Algorithm::Threshold: ThresholdParallel(output, p.threshold, threads); break;
        case Algorithm::Negative: negativeParallel(image, output, threads); break;
        case Algorithm::Contrast: contrastParallel(image, output, threads, p.contrastFactor); break;
        case Algorithm::Otsu: output = otsuSegmentation(image, threads); break;
        case Algorithm::MultiOtsu: output = multiOtsuSegmentation(image, p.classes, threads); break;
        case Algorithm::PeaksValleys: peaks_valleys_detectionParallel(image, output, threads, p.maxClasses); break;
        case Algorithm::Bernsen: bernsenThreshold(image, output, p.radius, p.minContrast, threads); break;
        case Algorithm::Niblack: niblackThreshold(image, output, p.radius, p.localThresholdK, threads); break;
        case Algorithm::Sauvola: sauvolaThreshold(image, output, p.radius, p.localThresholdK, threads); break;
        case Algorithm::Convolution: {
            const auto& filter = convolutionFilters().at(p.filter);
            convolution(image, filter.kernel, filter.absoluteValue, output, threads);
            break;
        }
        case Algorithm::MedianFilter:
            medianFilterParallel(image, output, threads, static_cast<MedianSort>(p.medianSort), p.medianRadius,
                                 p.medianSort == static_cast<int>(MedianSort::Shared) ? p.medianStrip : 1, p.medianSwitching);
            break;
        case Algorithm::Distortion:
            if (p.distortionModel == 1) correctBrownConradyCPU(image, {p.k1, p.k2, p.k3, p.p1, p.p2}, output, threads);
            else correctRadialDistortionCPU(image, p.k1, p.k2, output, threads);
            break;
        case Algorithm::Resize: {
            const int width = std::max(1, static_cast<int>(std::lround(image.width * p.scale)));
            const int height = std::max(1, static_cast<int>(std::lround(image.height * p.scale)));
            output = p.bilinear ? resizeBilinearCPU(image, width, height, threads) : resizeNearestCPU(image, width, height, threads);
            break;
        }
        case Algorithm::RegionGrowing:
            output = threads == 1 ? regionGrowingCPU(image, p.seedX, p.seedY, p.tolerance)
                                  : regionGrowingCPUParallel(image, p.seedX, p.seedY, p.tolerance, threads);
            break;
        case Algorithm::Canny: canny(image, output, p.cannyLow, p.cannyHigh, threads); break;
        case Algorithm::Gabor:
            gaborFilter(image, gaborBank(p.gaborSigma, p.gaborWavelength, p.gaborGamma, p.gaborOrientations), output, threads);
            break;
    }
    return output;
}

struct ImageDifference {
    std::size_t pixels = 0;
    int maximum = 0;
    double mean = 0.0;
};

inline ImageDifference metalDifference(const Image& reference, const Image& output) {
    if (reference.width != output.width || reference.height != output.height || reference.pixels.size() != output.pixels.size()) {
        throw std::runtime_error("Image dimensions differ");
    }
    ImageDifference difference;
    double sum = 0;
    for (std::size_t i = 0; i < reference.pixels.size(); ++i) {
        const int delta = std::abs(static_cast<int>(reference.pixels[i]) - output.pixels[i]);
        difference.pixels += delta != 0;
        difference.maximum = std::max(difference.maximum, delta);
        sum += delta;
    }
    difference.mean = sum / reference.pixels.size();
    return difference;
}

inline bool isLocalThreshold(const MetalCase& test) {
    return test.algorithm == Algorithm::Niblack || test.algorithm == Algorithm::Sauvola;
}

inline void checkMetalLocalThreshold(const MetalCase& test, const Image& image, const Image& reference, const Image& output,
                                     const std::string& backend) {
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        if (reference.pixels[i] == output.pixels[i]) continue;
        const int x = static_cast<int>(i % image.width);
        const int y = static_cast<int>(i / image.width);
        throw std::runtime_error(test.name + " " + backend + " differs from CPU at (" + std::to_string(x) + ", " +
                                 std::to_string(y) + ")");
    }
}

inline double metalSegmentationVariance(const Histogram& histogram, const std::vector<int>& thresholds) {
    long double totalCount = 0;
    long double totalSum = 0;
    for (int level = 0; level < 256; ++level) {
        totalCount += histogram[level];
        totalSum += static_cast<long double>(level) * histogram[level];
    }
    const long double mean = totalSum / totalCount;
    long double variance = 0;
    int begin = 0;
    for (std::size_t segment = 0; segment <= thresholds.size(); ++segment) {
        const int end = segment < thresholds.size() ? thresholds[segment] : 255;
        if (end < begin - 1 || end > 255) throw std::runtime_error("Invalid segmentation thresholds");
        long double count = 0;
        long double sum = 0;
        for (int level = begin; level <= end; ++level) {
            count += histogram[level];
            sum += static_cast<long double>(level) * histogram[level];
        }
        if (count > 0) {
            const long double difference = sum / count - mean;
            variance += count / totalCount * difference * difference;
        }
        begin = end + 1;
    }
    return static_cast<double>(variance);
}

inline void checkMetalSegmentation(const MetalCase& test, const Image& image, const Image& output,
                                   const std::vector<int>& thresholds) {
    const std::size_t expected = test.algorithm == Algorithm::Otsu ? 1 : test.parameters.classes - 1;
    if (thresholds.size() != expected) throw std::runtime_error("Incorrect number of segmentation thresholds");
    const Histogram histogram = calculateHistogram(image);
    const std::vector<int> optimal = test.algorithm == Algorithm::Otsu
        ? std::vector<int>{calculateOtsuThreshold(histogram)}
        : calculateMultiOtsuThresholds(histogram, test.parameters.classes);
    const double best = metalSegmentationVariance(histogram, optimal);
    const double actual = metalSegmentationVariance(histogram, thresholds);
    if (actual + 1e-9 < best * (1.0 - 1e-5)) throw std::runtime_error("Metal segmentation thresholds are suboptimal");
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        const auto segment = std::lower_bound(thresholds.begin(), thresholds.end(), image.pixels[i]) - thresholds.begin();
        const unsigned char expectedPixel = static_cast<unsigned char>(255 * segment / thresholds.size());
        if (output.pixels[i] != expectedPixel) throw std::runtime_error("Metal output does not match its segmentation thresholds");
    }
}
