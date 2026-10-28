#pragma once
#include "image.hpp"
#include <string>
#include <vector>

enum class Algorithm { Threshold, Otsu, Bernsen, Convolution, MedianFilter, Distortion };
enum class Mode { CPU1, CPUN, GPU };

struct Parameters {
    int threshold = 128;
    int radius = 7;
    int minContrast = 15;
    int filter = 0;
    bool sortingNetwork = false;
    float k1 = 0.15f;
    float k2 = 0.0f;
};

struct Timing {
    double totalUs = 0.0;
    double kernelUs = -1.0;
};

const std::vector<std::string>& convolutionFilterNames();
Timing runAlgorithm(Algorithm algorithm, Mode mode, const Image& image, const Parameters& parameters,
                    int threads, int repeats, Image& output);
