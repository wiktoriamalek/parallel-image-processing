#pragma once

#include "algorithm_parameters.hpp"
#include "image.hpp"
#include <memory>
#include <string>
#include <vector>

class MetalProcessor {
public:
    MetalProcessor(Algorithm algorithm, const Image& image, const Parameters& parameters);
    ~MetalProcessor();
    MetalProcessor(const MetalProcessor&) = delete;
    MetalProcessor& operator=(const MetalProcessor&) = delete;

    void run(const Image& image, Image& output);
    double kernelTimeUs() const;
    const std::vector<int>& thresholds() const;
    int iterations() const;
    std::string profilingMethod() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
