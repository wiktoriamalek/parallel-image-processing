#pragma once
#include "image.hpp"
#include <optional>
#include <string>
#include <vector>

enum class Algorithm { Threshold, Negative, Contrast, Otsu, MultiOtsu, PeaksValleys, Bernsen, Niblack, Sauvola,
                       Convolution, MedianFilter, Distortion, Resize, RegionGrowing };
enum class Mode { CPU1, CPUN, CPUNDynamic, GPU, GPUZeroCopy };

struct Parameters {
    int threshold = 128;
    float contrastFactor = 2.0f;
    int classes = 3;
    int maxClasses = 3;
    int radius = 7;
    int minContrast = 15;
    double localThresholdK = 0.2;
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
    std::optional<std::vector<int>> thresholds;
    int smoothingIterations = -1;
};

const std::vector<std::string>& convolutionFilterNames();
bool supportsZeroCopy(Algorithm algorithm);
Timing runAlgorithm(Algorithm algorithm, Mode mode, const Image& image, const Parameters& parameters,
                    int threads, int repeats, Image& output);
