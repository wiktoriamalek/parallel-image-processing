#pragma once
#include <array>
#include <cstddef>
#include "image.hpp"

using Histogram = std::array<std::size_t, 256>;
Histogram calculateHistogram(const Image& image);
int calculateOtsuThreshold(const Histogram& histogram);
Image otsuSegmentation(const Image& image);
