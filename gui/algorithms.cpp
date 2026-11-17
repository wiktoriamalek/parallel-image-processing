#include "algorithms.hpp"
#include "bernsen.hpp"
#include "common_benchmarking.hpp"
#include "contrast.hpp"
#include "convolution.hpp"
#include "distortion.hpp"
#include "median_filter.hpp"
#include "negative.hpp"
#include "otsu.hpp"
#include "region_growing.hpp"
#include "resize.hpp"
#include "thresholding.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
BrownConradyCoefficients brownConradyCoefficients(const Parameters& parameters) {
    return {parameters.k1, parameters.k2, parameters.k3, parameters.p1, parameters.p2};
}

template<class GPU>
Timing measureGPU(GPU& gpu, const Image& image, int repeats, Image& output) {
    Timing timing;
    std::vector<double> kernelTimes;
    cl_event kernelEvent = nullptr;
    timing.totalUs = measure([&] { gpu.run(image, output, &kernelEvent); }, repeats, [] {}, [&](int repeat) {
        double time = kernelTimeUs(kernelEvent);
        checkCL(clReleaseEvent(kernelEvent), "clReleaseEvent");
        if (repeat >= 0) kernelTimes.push_back(time);
    });
    timing.kernelUs = median(kernelTimes);
    return timing;
}

Timing runCPU(Algorithm algorithm, const Image& image, const Parameters& parameters, int threads, int repeats,
              Image& output) {
    output = image;
    switch (algorithm) {
        case Algorithm::Threshold:
            return {measure([&] { ThresholdParallel(output, parameters.threshold, threads); }, repeats,
                            [&] { output = image; }, [](int) {})};
        case Algorithm::Negative:
            return {measure([&] { negativeParallel(image, output, threads); }, repeats)};
        case Algorithm::Contrast:
            return {measure([&] { contrastParallel(image, output, threads, parameters.contrastFactor); }, repeats)};
        case Algorithm::Otsu:
            return {measure([&] { output = otsuSegmentation(image, threads); }, repeats)};
        case Algorithm::MultiOtsu:
            return {measure([&] { output = multiOtsuSegmentation(image, parameters.classes, threads); }, repeats)};
        case Algorithm::Bernsen:
            return {measure([&] {
                bernsenThreshold(image, output, parameters.radius, parameters.minContrast, threads);
            }, repeats)};
        case Algorithm::Convolution: {
            const ConvolutionFilter& selected = convolutionFilters().at(parameters.filter);
            return {measure([&] {
                convolution(image, selected.kernel, selected.absoluteValue, output, threads);
            }, repeats)};
        }
        case Algorithm::MedianFilter:
            return {measure([&] { medianFilterParallel(image, output, threads, parameters.sortingNetwork); }, repeats)};
        case Algorithm::Distortion:
            if (parameters.distortionModel == 1) {
                BrownConradyCoefficients coefficients = brownConradyCoefficients(parameters);
                return {measure([&] { correctBrownConradyCPU(image, coefficients, output, threads); }, repeats)};
            }
            return {measure([&] {
                correctRadialDistortionCPU(image, parameters.k1, parameters.k2, output, threads);
            }, repeats)};
        case Algorithm::Resize: {
            int width = std::max(1, static_cast<int>(std::lround(image.width * parameters.scale)));
            int height = std::max(1, static_cast<int>(std::lround(image.height * parameters.scale)));
            if (parameters.bilinear) return {measure([&] { output = resizeBilinearCPU(image, width, height, threads); }, repeats)};
            return {measure([&] { output = resizeNearestCPU(image, width, height, threads); }, repeats)};
        }
        case Algorithm::RegionGrowing:
            if (threads == 1) {
                return {measure([&] {
                    output = regionGrowingCPU(image, parameters.seedX, parameters.seedY, parameters.tolerance);
                }, repeats)};
            }
            return {measure([&] {
                output = regionGrowingCPUParallel(image, parameters.seedX, parameters.seedY, parameters.tolerance, threads);
            }, repeats)};
    }
    throw std::logic_error("Unknown algorithm");
}

Timing runOtsuGPU(const Image& image, int repeats, Image& output) {
    OtsuGPU gpu(image);
    Timing timing;
    std::vector<double> kernelTimes;
    cl_event histogramEvent = nullptr;
    cl_event otsuThresholdEvent = nullptr;
    cl_event thresholdEvent = nullptr;
    timing.totalUs = measure([&] {
        gpu.run(image, output, &histogramEvent, &otsuThresholdEvent, &thresholdEvent);
    }, repeats, [] {}, [&](int repeat) {
        double time = kernelTimeUs(histogramEvent) + kernelTimeUs(otsuThresholdEvent) + kernelTimeUs(thresholdEvent);
        checkCL(clReleaseEvent(histogramEvent), "clReleaseEvent");
        checkCL(clReleaseEvent(otsuThresholdEvent), "clReleaseEvent");
        checkCL(clReleaseEvent(thresholdEvent), "clReleaseEvent");
        if (repeat >= 0) kernelTimes.push_back(time);
    });
    timing.kernelUs = median(kernelTimes);
    return timing;
}

Timing runMultiOtsuGPU(const Image& image, const Parameters& parameters, int repeats, Image& output) {
    MultiOtsuGPU gpu(image, parameters.classes, parameters.zeroCopy);
    Timing timing;
    std::vector<double> kernelTimes;
    cl_event histogramEvent = nullptr;
    std::vector<cl_event> thresholdsEvents;
    cl_event lookupEvent = nullptr;
    timing.totalUs = measure([&] {
        gpu.run(image, output, &histogramEvent, &thresholdsEvents, &lookupEvent);
    }, repeats, [] {}, [&](int repeat) {
        double time = kernelTimeUs(histogramEvent) + kernelTimeUs(lookupEvent);
        checkCL(clReleaseEvent(histogramEvent), "clReleaseEvent");
        checkCL(clReleaseEvent(lookupEvent), "clReleaseEvent");
        for (cl_event event : thresholdsEvents) {
            time += kernelTimeUs(event);
            checkCL(clReleaseEvent(event), "clReleaseEvent");
        }
        thresholdsEvents.clear();
        if (repeat >= 0) kernelTimes.push_back(time);
    });
    timing.kernelUs = median(kernelTimes);
    return timing;
}

Timing runRegionGrowingGPU(const Image& image, const Parameters& parameters, int repeats, Image& output) {
    RegionGrowingGPU gpu(image, parameters.seedX, parameters.seedY, parameters.tolerance, parameters.localTiles);
    Timing timing;
    std::vector<double> kernelTimes;
    double kernelUs = 0.0;
    timing.totalUs = measure([&] { gpu.run(image, output, &kernelUs); }, repeats, [] {}, [&](int repeat) {
        if (repeat >= 0) kernelTimes.push_back(kernelUs);
    });
    timing.kernelUs = median(kernelTimes);
    return timing;
}

Timing runGPU(Algorithm algorithm, const Image& image, const Parameters& parameters, int repeats, Image& output) {
    output = image;
    switch (algorithm) {
        case Algorithm::Threshold: {
            ThresholdGPU gpu(image, parameters.threshold);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Negative: {
            NegativeGPU gpu(image);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Contrast: {
            ContrastGPU gpu(image, parameters.contrastFactor);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Otsu:
            return runOtsuGPU(image, repeats, output);
        case Algorithm::MultiOtsu:
            return runMultiOtsuGPU(image, parameters, repeats, output);
        case Algorithm::Bernsen: {
            BernsenGPU gpu(image, parameters.radius, parameters.minContrast);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Convolution: {
            const ConvolutionFilter& selected = convolutionFilters().at(parameters.filter);
            ConvolutionGPU gpu(image, selected.kernel, selected.absoluteValue);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::MedianFilter: {
            MedianFilterGPU gpu(image, parameters.sortingNetwork);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Distortion: {
            if (parameters.distortionModel == 1) {
                BrownConradyGPU gpu(image, brownConradyCoefficients(parameters));
                return measureGPU(gpu, image, repeats, output);
            }
            RadialDistortionGPU gpu(image, parameters.k1, parameters.k2);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::Resize: {
            int width = std::max(1, static_cast<int>(std::lround(image.width * parameters.scale)));
            int height = std::max(1, static_cast<int>(std::lround(image.height * parameters.scale)));
            output.width = width;
            output.height = height;
            output.pixels.resize(static_cast<std::size_t>(width) * height);
            ResizeGPU gpu(image, width, height, parameters.bilinear);
            return measureGPU(gpu, image, repeats, output);
        }
        case Algorithm::RegionGrowing:
            return runRegionGrowingGPU(image, parameters, repeats, output);
    }
    throw std::logic_error("Unknown algorithm");
}
}

const std::vector<std::string>& convolutionFilterNames() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> result;
        for (const auto& filter : convolutionFilters()) result.push_back(filter.name);
        return result;
    }();
    return names;
}


Timing runAlgorithm(Algorithm algorithm, Mode mode, const Image& image, const Parameters& parameters,
                    int threads, int repeats, Image& output) {
    if (mode == Mode::GPU) return runGPU(algorithm, image, parameters, repeats, output);
    return runCPU(algorithm, image, parameters, mode == Mode::CPU1 ? 1 : threads, repeats, output);
}
