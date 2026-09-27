#pragma once
#include "image_io.hpp"

Image correctRadialDistortion(const Image& image, float k1, float k2);
Image distortRadial(const Image& image, float k1, float k2);
void correctRadialDistortionRange(const Image& image, Image& output, float k1, float k2, int startY, int endY);
