#pragma once
#include "image.hpp"
#include "algorithm_parameters.hpp"
#include <optional>
#include <string>
#include <vector>

enum class Mode { CPU1, CPUN, CPUNDynamic, GPU, GPUZeroCopy, Metal };

struct Timing {
    double totalUs = 0.0;
    double kernelUs = -1.0;
    std::optional<std::vector<int>> thresholds;
    int smoothingIterations = -1;
    std::string gpuTimingMethod;
};

const std::vector<std::string>& convolutionFilterNames();
bool supportsZeroCopy(Algorithm algorithm);
Timing runAlgorithm(Algorithm algorithm, Mode mode, const Image& image, const Parameters& parameters,
                    int threads, int repeats, Image& output);
