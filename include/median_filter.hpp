#pragma once
#include "image.hpp"
#include <array>

Image medianFilter(const Image& image);
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow, bool sortingNetwork=false); // it doesnt need to return a result, it can modify it through a reference
void medianFilterParallel(const Image& image, Image& result, int threads, bool sortingNetwork=false);
unsigned char medianOf9(std::array<unsigned char, 9>& p);