#pragma once

#include "image.hpp"

void bernsenRows(const Image& image, Image& result, int radius, int minContrast, int firstRow, int endRow);
void bernsenThreshold(const Image& image, Image& result, int radius, int minContrast, int threads = 1);