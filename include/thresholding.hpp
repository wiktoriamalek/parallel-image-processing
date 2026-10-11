#pragma once
#include "image.hpp"
#include <cstddef>

Image thresholding(const Image& image, int threshold = 128);
void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max);
void ThresholdParallel(Image& image, int threshold, int threadCount);
