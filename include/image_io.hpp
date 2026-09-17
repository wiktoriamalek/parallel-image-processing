// functions to save and load

#pragma once
#include "image.hpp"
#include <array>
#include <string>

Image loadImage(const std::string &path);
void saveImage(const std::string& path, const Image& image); // by reference, not copy
Image thresholding(const Image& image);
void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max);
Image convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue);
void convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result);
void convolutionRows(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int firstRow, int endRow);