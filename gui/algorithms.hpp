#pragma once
#include "image.hpp"
#include <string>
#include <vector>

enum class Algorithm { Threshold, Negative, Contrast, Otsu, MultiOtsu, Bernsen, Convolution, MedianFilter, Distortion, Resize, RegionGrowing };
enum class Mode { CPU1, CPUN, GPU };

struct Parameters {
    int threshold = 128;
    float contrastFactor = 2.0f;
    int classes = 3;
    bool zeroCopy = true;
    int radius = 7;
    int minContrast = 15;
    int filter = 0;
    bool sortingNetwork = false;
    float k1 = 0.15f;
    float k2 = 0.0f;
    int distortionModel = 0;
    float k3 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
    bool bilinear = true;
    float scale = 2.0f;
    int seedX = 0;
    int seedY = 0;
    int tolerance = 20;
    bool localTiles = true;
};

struct Timing {
    double totalUs = 0.0;
    double kernelUs = -1.0;
};

const std::vector<std::string>& convolutionFilterNames();
Timing runAlgorithm(Algorithm algorithm, Mode mode, const Image& image, const Parameters& parameters,
                    int threads, int repeats, Image& output);
