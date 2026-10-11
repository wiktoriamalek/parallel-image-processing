#pragma once
#include "image.hpp"

Image medianFilter(const Image& image);
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow); // it doesnt need to return a result, it can modify it through a reference
void medianFilterParallel(const Image& image, Image& result, int threads);
