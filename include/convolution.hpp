#pragma once
#include "image.hpp"
#include <array>

Image convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue);
void convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int threads = 1);
void convolutionRows(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int firstRow, int endRow);
