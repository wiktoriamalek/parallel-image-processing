// functions to save and load

#pragma once
#include "image.hpp"
#include <string>

Image loadImage(const std::string &path);
void saveImage(const std::string& path, const Image& image); // by reference, not copy
Image thresholding(const Image& image);
void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max);
