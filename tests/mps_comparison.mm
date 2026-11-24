#include "common_benchmarking.hpp"
#include "convolution.hpp"
#include "image_io.hpp"
#include "median_filter.hpp"
#include "metal_algorithms.hpp"
#include "thresholding.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int referenceThreads = 11;

struct Case {
    std::string name;
    Algorithm algorithm;
    Parameters parameters;
    int border;
    std::string mpsName;
    std::function<MPSUnaryImageKernel*(id<MTLDevice>)> makeKernel;
};

struct Difference {
    std::size_t pixels = 0;
    int maximum = 0;
};

struct Samples {
    std::vector<double> total;
    std::vector<double> kernel;
};

double elapsedUs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

Difference compare(const Image& a, const Image& b, int border) {
    Difference result;
    for (int y = border; y < a.height - border; ++y) {
        for (int x = border; x < a.width - border; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * a.width + x;
            const int difference = std::abs(static_cast<int>(a.pixels[index]) - static_cast<int>(b.pixels[index]));
            if (difference) {
                ++result.pixels;
                result.maximum = std::max(result.maximum, difference);
            }
        }
    }
    return result;
}

Image reference(const Case& item, const Image& image) {
    Image result = image;
    switch (item.algorithm) {
        case Algorithm::Threshold:
            ThresholdParallel(result, item.parameters.threshold, referenceThreads);
            break;
        case Algorithm::Convolution: {
            const auto& filter = convolutionFilters().at(item.parameters.filter);
            convolution(image, filter.kernel, filter.absoluteValue, result, referenceThreads);
            break;
        }
        case Algorithm::MedianFilter:
            medianFilterParallel(image, result, referenceThreads, static_cast<MedianSort>(item.parameters.medianSort),
                                 item.parameters.medianRadius, item.parameters.medianStrip);
            break;
        default:
            throw std::invalid_argument("Unsupported case");
    }
    return result;
}

Image roundedConvolution(const Case& item, const Image& image) {
    Image result = image;
    if (item.algorithm != Algorithm::Convolution) return result;
    const auto& kernel = convolutionFilters().at(item.parameters.filter).kernel;
    for (int y = 1; y < image.height - 1; ++y) {
        for (int x = 1; x < image.width - 1; ++x) {
            double sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += image.pixels[static_cast<std::size_t>(y + ky) * image.width + x + kx] * static_cast<double>(kernel[ky + 1][kx + 1]);
                }
            }
            result.pixels[static_cast<std::size_t>(y) * image.width + x] =
                static_cast<unsigned char>(std::lround(std::clamp(sum, 0.0, 255.0)));
        }
    }
    return result;
}

std::vector<Case> cases() {
    std::vector<Case> list;
    Parameters threshold;
    threshold.threshold = 128;
    list.push_back({"threshold", Algorithm::Threshold, threshold, 0, "MPSImageThresholdBinary", [](id<MTLDevice> device) {
                        static const float transform[3] = {1.0f, 0.0f, 0.0f};
                        return (MPSUnaryImageKernel*)[[MPSImageThresholdBinary alloc] initWithDevice:device
                                                                                     thresholdValue:(128.0f - 0.5f) / 255.0f
                                                                                       maximumValue:1.0f
                                                                           linearGrayColorTransform:transform];
                    }});
    for (const std::string filterName : {"sharpen", "box_blur"}) {
        Parameters convolution;
        int index = 0;
        while (convolutionFilters().at(index).name != filterName) ++index;
        convolution.filter = index;
        list.push_back({"convolution_" + filterName, Algorithm::Convolution, convolution, 1, "MPSImageConvolution",
                        [index](id<MTLDevice> device) {
                            float weights[9];
                            const auto& kernel = convolutionFilters().at(index).kernel;
                            for (int i = 0; i < 9; ++i) weights[i] = kernel[i / 3][i % 3];
                            return (MPSUnaryImageKernel*)[[MPSImageConvolution alloc] initWithDevice:device
                                                                                         kernelWidth:3
                                                                                        kernelHeight:3
                                                                                             weights:weights];
                        }});
        if (filterName == "box_blur") {
            list.push_back({"convolution_box_blur", Algorithm::Convolution, convolution, 1, "MPSImageBox", [](id<MTLDevice> device) {
                                return (MPSUnaryImageKernel*)[[MPSImageBox alloc] initWithDevice:device kernelWidth:3 kernelHeight:3];
                            }});
        }
    }
    for (int radius = 1; radius <= 3; ++radius) {
        Parameters median;
        median.medianSort = static_cast<int>(MedianSort::Shared);
        median.medianRadius = radius;
        median.medianStrip = 8;
        const int diameter = 2 * radius + 1;
        list.push_back({"median_" + std::to_string(diameter) + "x" + std::to_string(diameter), Algorithm::MedianFilter, median, radius,
                        "MPSImageMedian", [diameter](id<MTLDevice> device) {
                            return (MPSUnaryImageKernel*)[[MPSImageMedian alloc] initWithDevice:device kernelDiameter:diameter];
                        }});
    }
    return list;
}

struct MpsRunner {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    MPSUnaryImageKernel* kernel;
    id<MTLTexture> input;
    id<MTLTexture> output;
    id<MTLBuffer> inputBuffer = nil;
    id<MTLBuffer> outputBuffer = nil;
    bool linear;
    int width;
    int height;

    MpsRunner(id<MTLDevice> selected, id<MTLCommandQueue> commandQueue, MPSUnaryImageKernel* imageKernel, int w, int h, bool useLinear)
        : device(selected), queue(commandQueue), kernel(imageKernel), linear(useLinear), width(w), height(h) {
        kernel.edgeMode = MPSImageEdgeModeClamp;
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                              width:width
                                                                                             height:height
                                                                                          mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        descriptor.storageMode = MTLStorageModeShared;
        const std::size_t bytes = static_cast<std::size_t>(width) * height;
        if (linear) {
            inputBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            outputBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            input = [inputBuffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:width];
            output = [outputBuffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:width];
        } else {
            input = [device newTextureWithDescriptor:descriptor];
            output = [device newTextureWithDescriptor:descriptor];
        }
        if (!input || !output) throw std::runtime_error("Could not create MPS textures");
    }

    double run(const Image& image, Image& result) {
        @autoreleasepool {
            const MTLRegion region = MTLRegionMake2D(0, 0, width, height);
            if (linear) {
                std::memcpy([inputBuffer contents], image.pixels.data(), image.pixels.size());
            } else {
                [input replaceRegion:region mipmapLevel:0 withBytes:image.pixels.data() bytesPerRow:width];
            }
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            [kernel encodeToCommandBuffer:commandBuffer sourceTexture:input destinationTexture:output];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            if ([commandBuffer status] != MTLCommandBufferStatusCompleted) throw std::runtime_error("MPS command buffer failed");
            result.width = width;
            result.height = height;
            result.pixels.resize(static_cast<std::size_t>(width) * height);
            if (linear) {
                std::memcpy(result.pixels.data(), [outputBuffer contents], result.pixels.size());
            } else {
                [output getBytes:result.pixels.data() bytesPerRow:width fromRegion:region mipmapLevel:0];
            }
            return ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1e6;
        }
    }
};

void report(const std::string& path, const std::string& imagePath, const Image& image, const Case& item, const std::string& implementation,
            int repeats, const Samples& samples, const Difference& difference, const Difference& rounded) {
    const double total = median(samples.total);
    const double kernel = median(samples.kernel);
    const std::size_t compared = static_cast<std::size_t>(image.width - 2 * item.border) * (image.height - 2 * item.border);
    std::ostringstream row;
    row << std::fixed << std::setprecision(3) << '"' << imagePath << '"' << ',' << image.width << ',' << image.height << ',' << item.name
        << ',' << implementation << ',' << repeats << ',' << total << ',' << kernel << ',' << compared << ',' << difference.pixels << ','
        << difference.maximum << ',' << rounded.pixels << ',' << rounded.maximum;
    appendCsv(path,
              "image,width,height,operation,implementation,repeats,total_us,kernel_us,compared_pixels,different_pixels,max_abs,"
              "different_from_rounded,max_abs_from_rounded",
              row.str());
    std::cout << std::left << std::setw(26) << item.name << std::setw(36) << implementation << std::right << std::fixed
              << std::setprecision(1) << std::setw(11) << total << " us" << std::setw(11) << kernel << " us   diff "
              << difference.pixels << " (max " << difference.maximum << ")" << std::endl;
}
}

int main(int argc, char* argv[]) {
    try {
        @autoreleasepool {
            BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/mps_comparison.csv");
            const std::vector<std::string> images = {"images/cms_resolution_change/cms_512.png", "images/cms_resolution_change/cms_2048.png",
                                                     "images/cms_resolution_change/cms_8192.png"};
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal GPU found");
            if (!MPSSupportsMTLDevice(device)) throw std::runtime_error("MPS does not support this GPU");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            const NSUInteger alignment = [device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR8Unorm];
            std::cout << [[device name] UTF8String] << ", linear texture row alignment " << alignment << " B" << std::endl;

            for (const std::string& path : images) {
                const Image image = loadImage(path);
                std::cout << "\n" << path << " (" << image.width << "x" << image.height << ")" << std::endl;
                for (const Case& item : cases()) {
                    const Image expected = reference(item, image);
                    const Image rounded = roundedConvolution(item, image);
                    MetalProcessor processor(item.algorithm, image, item.parameters);
                    std::vector<MpsRunner> runners;
                    std::vector<std::string> names = {"project_metal"};
                    runners.emplace_back(device, queue, item.makeKernel(device), image.width, image.height, false);
                    names.push_back(item.mpsName + "_texture");
                    if (image.width % alignment == 0) {
                        runners.emplace_back(device, queue, item.makeKernel(device), image.width, image.height, true);
                        names.push_back(item.mpsName + "_linear");
                    }
                    const std::size_t implementations = 1 + runners.size();
                    std::vector<Samples> samples(implementations);
                    std::vector<Image> outputs(implementations);
                    auto runOne = [&](std::size_t which, bool record) {
                        double total = 0.0;
                        double kernel = 0.0;
                        if (which == 0) {
                            const auto start = std::chrono::steady_clock::now();
                            processor.run(image, outputs[0]);
                            total = elapsedUs(start);
                            kernel = processor.kernelTimeUs();
                        } else {
                            const auto start = std::chrono::steady_clock::now();
                            kernel = runners[which - 1].run(image, outputs[which]);
                            total = elapsedUs(start);
                        }
                        if (record) {
                            samples[which].total.push_back(total);
                            samples[which].kernel.push_back(kernel);
                        }
                    };
                    for (int warmUp = 0; warmUp < 5; ++warmUp) {
                        for (std::size_t which = 0; which < implementations; ++which) runOne(which, false);
                    }
                    for (int repeat = 0; repeat < options.repeats; ++repeat) {
                        for (std::size_t step = 0; step < implementations; ++step) runOne((step + repeat) % implementations, true);
                    }
                    for (std::size_t which = 0; which < implementations; ++which) {
                        report(options.output, path, image, item, names[which], options.repeats, samples[which],
                               compare(expected, outputs[which], item.border), compare(rounded, outputs[which], item.border));
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
