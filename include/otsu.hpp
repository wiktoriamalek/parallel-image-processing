#pragma once
#include <array>
#include <cstddef>
#include "image.hpp"

using Histogram = std::array<std::size_t, 256>;
Histogram calculateHistogram(const Image& image);
Histogram calculateHistogram2Threads(const Image& image);
Histogram calculateHistogramGPU(const Image& image);
int calculateOtsuThreshold(const Histogram& histogram);
Image otsuSegmentation(const Image& image);
Image otsuSegmentation2Threads(const Image& image);
Image otsuSegmentationGPU(const Image& image);
Image applyThresholdGPU(const Image& image, int threshold);

