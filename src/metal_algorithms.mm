#include "metal_algorithms.hpp"
#include "convolution.hpp"
#include "median_filter.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
std::string metalError(NSError* error) {
    return error ? std::string([[error localizedDescription] UTF8String]) : "Unknown Metal error";
}

struct Argument {
    id<MTLBuffer> buffer = nil;
    std::array<unsigned char, 4> bytes{};

    Argument(id<MTLBuffer> value) : buffer(value) {}
    Argument(int value) { std::memcpy(bytes.data(), &value, sizeof(value)); }
    Argument(float value) { std::memcpy(bytes.data(), &value, sizeof(value)); }
};

struct Pass {
    id<MTLComputePipelineState> pipeline;
    std::vector<Argument> arguments;
    MTLSize grid;
    MTLSize group;
    std::vector<NSUInteger> scratch;
    bool profile = true;
};

std::size_t divideUp(std::size_t size, std::size_t divisor) {
    return (size + divisor - 1) / divisor;
}

void checkImage(const Image& image) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height ||
        image.pixels.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Metal requires a nonempty grayscale image with at most INT_MAX pixels");
    }
}
}

struct MetalProcessor::Impl {
    Algorithm algorithm;
    Parameters parameters;
    int width;
    int height;
    int outputWidth;
    int outputHeight;
    int pixelCount;
    int tilesX;
    int tilesY;
    int iterationCount = 0;
    double kernelUs = 0.0;
    std::vector<int> thresholdValues;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLLibrary> medianLibrary;
    id<MTLCounterSampleBuffer> samples = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    id<MTLBuffer> input;
    id<MTLBuffer> output;
    id<MTLBuffer> result;
    id<MTLBuffer> changed;
    id<MTLBuffer> activeA;
    id<MTLBuffer> activeB;
    id<MTLBuffer> maskB;
    std::vector<Pass> passes;
    std::vector<Pass> iterationPasses;
    std::vector<Pass> finalPasses;

    Impl(Algorithm selected, const Image& image, const Parameters& values)
        : algorithm(selected), parameters(values), width(image.width), height(image.height),
          outputWidth(width), outputHeight(height), pixelCount(static_cast<int>(image.pixels.size())),
          tilesX(static_cast<int>(divideUp(width, 64))), tilesY(static_cast<int>(divideUp(height, 64))) {
        checkImage(image);
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU found");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Could not create the Metal command queue");
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:@METAL_LIBRARY_PATH];
        library = [device newLibraryWithURL:url error:&error];
        if (!library) throw std::runtime_error("Could not load the Metal library: " + metalError(error));
        configureProfiling();
        if (algorithm == Algorithm::Resize) {
            const double scaledWidth = static_cast<double>(width) * parameters.scale;
            const double scaledHeight = static_cast<double>(height) * parameters.scale;
            if (!std::isfinite(parameters.scale) || parameters.scale <= 0 ||
                scaledWidth > std::numeric_limits<int>::max() || scaledHeight > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("Invalid resize scale");
            }
            outputWidth = std::max(1, static_cast<int>(std::lround(scaledWidth)));
            outputHeight = std::max(1, static_cast<int>(std::lround(scaledHeight)));
            if (static_cast<std::size_t>(outputWidth) * outputHeight > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("Resized image exceeds INT_MAX pixels");
            }
        }
        input = buffer(image.pixels.size());
        output = buffer(static_cast<std::size_t>(outputWidth) * outputHeight);
        configure();
    }

    id<MTLBuffer> buffer(std::size_t size) {
        if (size > [device maxBufferLength]) throw std::invalid_argument("Metal buffer exceeds the device limit");
        id<MTLBuffer> value = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!value) throw std::runtime_error("Could not allocate a Metal buffer of " + std::to_string(size) + " bytes");
        return value;
    }

    void configureProfiling() {
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return;
        for (id<MTLCounterSet> counterSet in [device counterSets]) {
            if (![[counterSet name] isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet = counterSet;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = 1024;
            NSError* error = nil;
            samples = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
            return;
        }
    }

    Pass pass(const std::string& name, std::vector<Argument> arguments, MTLSize grid,
              MTLSize group = MTLSizeMake(16, 16, 1)) {
        auto found = pipelines.find(name);
        if (found == pipelines.end()) {
            NSString* functionName = [NSString stringWithUTF8String:name.c_str()];
            id<MTLLibrary> source = medianLibrary ? medianLibrary : library;
            id<MTLFunction> function = [source newFunctionWithName:functionName];
            if (!function) throw std::runtime_error("Metal kernel not found: " + name);
            NSError* error = nil;
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error("Could not create Metal pipeline " + name + ": " + metalError(error));
            found = pipelines.emplace(name, pipeline).first;
        }
        if (group.width * group.height * group.depth > [found->second maxTotalThreadsPerThreadgroup]) {
            throw std::runtime_error("Metal threadgroup exceeds the pipeline limit for " + name);
        }
        return {found->second, std::move(arguments), grid, group, {}};
    }

    void configureHistogram() {
        id<MTLBuffer> histogram = buffer(256 * sizeof(std::uint32_t));
        const MTLSize bins = MTLSizeMake(256, 1, 1);
        passes.push_back(pass("clearHistogramKernel", {histogram}, bins, bins));
        passes.back().profile = false;
        passes.push_back(pass("histogramLocalKernel", {input, histogram, pixelCount}, MTLSizeMake(65536, 1, 1), bins));
        if (algorithm == Algorithm::Otsu) {
            result = buffer(sizeof(int));
            passes.push_back(pass("otsuThresholdKernel", {histogram, result}, bins, bins));
            passes.push_back(pass("thresholdKernel", {input, output, pixelCount, result}, MTLSizeMake(pixelCount, 1, 1), bins));
            return;
        }
        id<MTLBuffer> lookup = buffer(256);
        if (algorithm == Algorithm::PeaksValleys) {
            if (parameters.maxClasses < 1 || parameters.maxClasses > 256) {
                throw std::invalid_argument("Peaks/valleys needs between 1 and 256 maximum classes");
            }
            result = buffer(258 * sizeof(int));
            passes.push_back(pass("peaksValleysThresholdsKernel", {histogram, parameters.maxClasses, result, lookup}, bins, bins));
        } else {
            if (parameters.classes < 2 || parameters.classes > 256) {
                throw std::invalid_argument("Multi-Otsu needs between 2 and 256 classes");
            }
            id<MTLBuffer> count = buffer(257 * sizeof(std::uint64_t));
            id<MTLBuffer> sum = buffer(257 * sizeof(std::int64_t));
            id<MTLBuffer> layers = buffer(parameters.classes * 257 * sizeof(float));
            id<MTLBuffer> split = buffer((parameters.classes - 1) * 256 * sizeof(std::uint16_t));
            result = buffer((parameters.classes - 1) * sizeof(int));
            passes.push_back(pass("multiOtsuPrefixKernel", {histogram, count, sum, layers}, bins, bins));
            for (int k = 2; k <= parameters.classes; ++k) {
                const int firstEnd = k == parameters.classes ? 256 : k;
                passes.push_back(pass("multiOtsuLayerKernel", {count, sum, layers, split, k, firstEnd},
                                      MTLSizeMake((257 - firstEnd) * 256, 1, 1), bins));
            }
            passes.push_back(pass("multiOtsuLookupTableKernel", {layers, split, parameters.classes, result, lookup}, bins, bins));
        }
        passes.push_back(pass("multiOtsuApplyLookupKernel", {input, output, pixelCount, lookup}, MTLSizeMake(pixelCount, 1, 1), bins));
    }

    void configureMedian() {
        const auto sort = static_cast<MedianSort>(parameters.medianSort);
        if (parameters.medianSort < 0 || parameters.medianSort > static_cast<int>(MedianSort::Shared) ||
            parameters.medianRadius < 1) {
            throw std::invalid_argument("Invalid Metal median variant or radius");
        }
        const int strip = sort == MedianSort::Shared ? parameters.medianStrip : 1;
        std::string definitions;
        if (sort == MedianSort::Standard) {
            definitions = "#define MEDIAN_RADIUS " + std::to_string(parameters.medianRadius) + "\n";
        } else {
            definitions = buildMedianNetwork(sort, parameters.medianRadius, strip).openclDefinitions();
        }
        std::ifstream source(METAL_MEDIAN_SOURCE_PATH);
        if (!source) throw std::runtime_error("Could not open the Metal median kernel");
        std::ostringstream text;
        text << definitions << source.rdbuf();
        NSString* code = [NSString stringWithUTF8String:text.str().c_str()];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        medianLibrary = [device newLibraryWithSource:code options:options error:&error];
        if (!medianLibrary) throw std::runtime_error("Could not compile the Metal median kernel: " + metalError(error));
        passes.push_back(pass(sort == MedianSort::Standard ? "median_insertion" : "median_network",
                              {input, output, width, height, static_cast<int>(parameters.medianSwitching)},
                              MTLSizeMake(width, divideUp(height, strip), 1)));
    }

    void configureBernsen() {
        if (parameters.radius < 0 || parameters.radius > std::numeric_limits<int>::max() / 4) {
            throw std::invalid_argument("Invalid Bernsen radius");
        }
        id<MTLBuffer> rows = buffer(2 * static_cast<std::size_t>(pixelCount));
        const auto radius = static_cast<std::size_t>(parameters.radius);
        const NSUInteger rowBytes = 4 * (2048 + 2 * radius);
        const NSUInteger columnBytes = 4 * (64 + 2 * radius) * 32;
        const bool rowLocal = rowBytes <= [device maxThreadgroupMemoryLength];
        const bool columnLocal = columnBytes <= [device maxThreadgroupMemoryLength];
        passes.push_back(pass(rowLocal ? "bernsenRowMinMaxLocal" : "bernsenRowMinMax",
                              {input, rows, width, height, parameters.radius},
                              rowLocal ? MTLSizeMake(divideUp(width, 2048) * 256, height, 1) : MTLSizeMake(width, height, 1),
                              rowLocal ? MTLSizeMake(256, 1, 1) : MTLSizeMake(16, 16, 1)));
        if (rowLocal) passes.back().scratch = {rowBytes};
        passes.push_back(pass(columnLocal ? "bernsenColumnsLocal" : "bernsenColumns",
                              {input, rows, output, width, height, parameters.radius, parameters.minContrast},
                              columnLocal ? MTLSizeMake(divideUp(width, 32) * 32, divideUp(height, 64) * 8, 1) : MTLSizeMake(width, height, 1),
                              columnLocal ? MTLSizeMake(32, 8, 1) : MTLSizeMake(16, 16, 1)));
        if (columnLocal) passes.back().scratch = {columnBytes};
    }

    void configureLocalThreshold() {
        if (parameters.radius < 0 || parameters.radius > 127 || !std::isfinite(parameters.localThresholdK)) {
            throw std::invalid_argument("GPU local threshold supports radii 0-127 and a finite k");
        }
        int band = 16;
        while (band < parameters.radius && band < 64) band *= 2;
        const NSUInteger localBytes = (2048 + 2 * parameters.radius) * sizeof(std::uint32_t);
        const NSUInteger blockBytes = divideUp(2048 + 2 * parameters.radius, 8) * sizeof(std::uint32_t);
        if (2 * localBytes + 2 * blockBytes > [device maxThreadgroupMemoryLength]) {
            throw std::runtime_error("Insufficient Metal threadgroup memory for local thresholding");
        }
        passes.push_back(pass(algorithm == Algorithm::Niblack ? "niblack" : "sauvola",
                              {input, output, width, height, parameters.radius, static_cast<float>(parameters.localThresholdK), 128.0f, band},
                              MTLSizeMake(divideUp(width, 2048) * 256, divideUp(height, band), 1), MTLSizeMake(256, 1, 1)));
        passes.back().scratch = {localBytes, localBytes, blockBytes, blockBytes};
    }

    void configureIteration() {
        changed = buffer(sizeof(int));
        if (algorithm == Algorithm::RegionGrowing && !parameters.localTiles) {
            maskB = buffer(pixelCount);
            iterationPasses.push_back(pass("region_step", {input, output, maskB, changed, width, height, 0, parameters.tolerance},
                                           MTLSizeMake(width, height, 1)));
            return;
        }
        activeA = buffer(static_cast<std::size_t>(tilesX) * tilesY * sizeof(int));
        activeB = buffer(static_cast<std::size_t>(tilesX) * tilesY * sizeof(int));
        const MTLSize grid = MTLSizeMake(tilesX * 16, tilesY * 16, 1);
        if (algorithm == Algorithm::RegionGrowing) {
            const int seedTile = parameters.seedY / 64 * tilesX + parameters.seedX / 64;
            passes.push_back(pass("region_active_init", {activeA, tilesX * tilesY, seedTile}, MTLSizeMake(tilesX * tilesY, 1, 1), MTLSizeMake(256, 1, 1)));
            iterationPasses.push_back(pass("region_tile_step", {input, output, changed, activeA, activeB, width, height,
                                           0, parameters.tolerance, tilesX, tilesY}, grid));
        }
    }

    void configureCanny() {
        if (!std::isfinite(parameters.cannyLow) || !std::isfinite(parameters.cannyHigh) ||
            parameters.cannyLow < 0 || parameters.cannyHigh < parameters.cannyLow) {
            throw std::invalid_argument("Canny thresholds must satisfy 0 <= low <= high and be finite");
        }
        auto squared = [](float threshold) {
            return static_cast<int>(std::min(std::ceil(static_cast<double>(threshold) * threshold), 2.0 * 1020 * 1020 + 1));
        };
        id<MTLBuffer> blurred = buffer(pixelCount);
        id<MTLBuffer> magnitude = buffer(static_cast<std::size_t>(pixelCount) * sizeof(int));
        id<MTLBuffer> sector = buffer(pixelCount);
        id<MTLBuffer> state = buffer(pixelCount);
        const MTLSize grid = MTLSizeMake(width, height, 1);
        passes.push_back(pass("canny_gauss", {input, blurred, width, height}, grid));
        passes.push_back(pass("canny_gradient", {blurred, magnitude, sector, width, height}, grid));
        passes.push_back(pass("canny_suppress", {magnitude, sector, state, width, height, squared(parameters.cannyLow), squared(parameters.cannyHigh)}, grid));
        configureIteration();
        iterationPasses.push_back(pass("canny_track", {state, changed, activeA, activeB, width, height, tilesX, tilesY}, MTLSizeMake(tilesX * 16, tilesY * 16, 1)));
        finalPasses.push_back(pass("canny_finish", {state, output, width, height}, grid));
    }

    void configure() {
        const MTLSize grid = MTLSizeMake(width, height, 1);
        const MTLSize line = MTLSizeMake(pixelCount, 1, 1);
        const MTLSize lineGroup = MTLSizeMake(256, 1, 1);
        switch (algorithm) {
            case Algorithm::Threshold:
                passes.push_back(pass("threshold_kernel", {output, parameters.threshold}, line, lineGroup));
                break;
            case Algorithm::Negative:
                passes.push_back(pass("negative_gpu", {input, output}, line, lineGroup));
                break;
            case Algorithm::Contrast:
                if (!std::isfinite(parameters.contrastFactor)) throw std::invalid_argument("Contrast factor must be finite");
                passes.push_back(pass("contrast_gpu", {input, output, parameters.contrastFactor}, line, lineGroup));
                break;
            case Algorithm::Otsu:
            case Algorithm::MultiOtsu:
            case Algorithm::PeaksValleys:
                configureHistogram();
                break;
            case Algorithm::Bernsen:
                configureBernsen();
                break;
            case Algorithm::Niblack:
            case Algorithm::Sauvola:
                configureLocalThreshold();
                break;
            case Algorithm::Convolution: {
                const auto& filter = convolutionFilters().at(parameters.filter);
                id<MTLBuffer> coefficients = buffer(9 * sizeof(float));
                std::memcpy([coefficients contents], filter.kernel.data(), 9 * sizeof(float));
                passes.push_back(pass("convolution3x3", {input, output, width, height, coefficients, static_cast<int>(filter.absoluteValue)}, grid));
                break;
            }
            case Algorithm::MedianFilter:
                configureMedian();
                break;
            case Algorithm::Distortion:
                if (parameters.distortionModel == 1) {
                    passes.push_back(pass("brown_conrady_correction", {input, output, width, height, parameters.k1,
                                          parameters.k2, parameters.k3, parameters.p1, parameters.p2}, grid));
                } else {
                    passes.push_back(pass("distortion_correction", {input, output, height, width, parameters.k1, parameters.k2}, grid));
                }
                break;
            case Algorithm::Resize:
                passes.push_back(pass(parameters.bilinear ? "resize_bilinear" : "resize_nearest",
                                      {input, output, width, height, outputWidth, outputHeight}, MTLSizeMake(outputWidth, outputHeight, 1)));
                break;
            case Algorithm::RegionGrowing:
                if (parameters.seedX < 0 || parameters.seedX >= width || parameters.seedY < 0 || parameters.seedY >= height ||
                    parameters.tolerance < 0 || parameters.tolerance > 255) {
                    throw std::invalid_argument("Invalid region-growing seed or tolerance");
                }
                passes.push_back(pass("region_init", {output, width, height, parameters.seedX, parameters.seedY}, grid));
                configureIteration();
                break;
            case Algorithm::Canny:
                configureCanny();
                break;
            default:
                throw std::invalid_argument("Unknown Metal algorithm");
        }
    }

    void submit(const std::vector<Pass>& work) {
        if (work.empty()) return;
        if (work.size() * 2 > 1024) throw std::runtime_error("Too many Metal profiling samples in one batch");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command) throw std::runtime_error("Could not create a Metal command buffer");
        NSUInteger sampleIndex = 0;
        for (const Pass& item : work) {
            MTLComputePassDescriptor* descriptor = [MTLComputePassDescriptor computePassDescriptor];
            if (samples && item.profile) {
                auto* attachment = descriptor.sampleBufferAttachments[0];
                attachment.sampleBuffer = samples;
                attachment.startOfEncoderSampleIndex = sampleIndex++;
                attachment.endOfEncoderSampleIndex = sampleIndex++;
            }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDescriptor:descriptor];
            if (!encoder) throw std::runtime_error("Could not create a Metal compute encoder");
            [encoder setComputePipelineState:item.pipeline];
            for (NSUInteger i = 0; i < item.arguments.size(); ++i) {
                const Argument& argument = item.arguments[i];
                if (argument.buffer) [encoder setBuffer:argument.buffer offset:0 atIndex:i];
                else [encoder setBytes:argument.bytes.data() length:argument.bytes.size() atIndex:i];
            }
            for (NSUInteger i = 0; i < item.scratch.size(); ++i) {
                [encoder setThreadgroupMemoryLength:((item.scratch[i] + 15) / 16) * 16 atIndex:i];
            }
            [encoder dispatchThreads:item.grid threadsPerThreadgroup:item.group];
            [encoder endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        if ([command status] == MTLCommandBufferStatusError) {
            throw std::runtime_error("Metal execution failed: " + metalError([command error]));
        }
        if (samples && sampleIndex > 0) {
            NSData* resolved = [samples resolveCounterRange:NSMakeRange(0, sampleIndex)];
            if (!resolved || [resolved length] != sampleIndex * sizeof(MTLCounterResultTimestamp)) {
                throw std::runtime_error("Could not resolve Metal GPU timestamps");
            }
            const auto* timestamps = static_cast<const MTLCounterResultTimestamp*>([resolved bytes]);
            for (NSUInteger i = 0; i < sampleIndex; i += 2) {
                const std::uint64_t start = timestamps[i].timestamp;
                const std::uint64_t end = timestamps[i + 1].timestamp;
                if (start == MTLCounterErrorValue || end == MTLCounterErrorValue || end < start) {
                    throw std::runtime_error("Invalid Metal GPU timestamp");
                }
                kernelUs += static_cast<double>(end - start) / 1000.0;
            }
        }
    }

    void runIterations(const Image& image) {
        iterationCount = 0;
        if (algorithm == Algorithm::Canny) {
            std::fill_n(static_cast<int*>([activeA contents]), tilesX * tilesY, 1);
        } else {
            const int seed = image.pixels[static_cast<std::size_t>(parameters.seedY) * width + parameters.seedX];
            iterationPasses[0].arguments[parameters.localTiles ? 7 : 6] = seed;
        }
        int* flag = static_cast<int*>([changed contents]);
        bool grow = true;
        while (grow) {
            *flag = 0;
            const int batch = algorithm == Algorithm::RegionGrowing && parameters.localTiles ? 16 : 1;
            for (int step = 0; step < batch; ++step) {
                submit(iterationPasses);
                Pass& item = iterationPasses[0];
                if (algorithm == Algorithm::RegionGrowing && !parameters.localTiles) {
                    std::swap(item.arguments[1], item.arguments[2]);
                } else {
                    const int index = algorithm == Algorithm::Canny ? 2 : 3;
                    std::swap(item.arguments[index], item.arguments[index + 1]);
                }
                ++iterationCount;
            }
            grow = *flag != 0;
            if (iterationCount > pixelCount + 16) throw std::runtime_error("Metal region propagation did not converge");
        }
        if (algorithm == Algorithm::RegionGrowing && !parameters.localTiles) {
            std::memcpy([output contents], [iterationPasses[0].arguments[1].buffer contents], pixelCount);
            iterationPasses[0].arguments[1] = output;
            iterationPasses[0].arguments[2] = maskB;
        } else {
            const int index = algorithm == Algorithm::Canny ? 2 : 3;
            iterationPasses[0].arguments[index] = activeA;
            iterationPasses[0].arguments[index + 1] = activeB;
        }
    }

    void run(const Image& image, Image& destination) {
        checkImage(image);
        if (image.width != width || image.height != height) {
            throw std::invalid_argument("Image dimensions do not match the Metal buffers");
        }
        destination.width = outputWidth;
        destination.height = outputHeight;
        destination.pixels.resize(static_cast<std::size_t>(outputWidth) * outputHeight);
        kernelUs = samples ? 0.0 : -1.0;
        std::memcpy([input contents], image.pixels.data(), pixelCount);
        if (algorithm == Algorithm::Threshold) std::memcpy([output contents], [input contents], pixelCount);
        submit(passes);
        if (!iterationPasses.empty()) runIterations(image);
        submit(finalPasses);
        std::memcpy(destination.pixels.data(), [output contents], destination.pixels.size());
        if (algorithm == Algorithm::MultiOtsu) {
            const int* values = static_cast<const int*>([result contents]);
            thresholdValues.assign(values, values + parameters.classes - 1);
        } else if (algorithm == Algorithm::PeaksValleys) {
            const int* values = static_cast<const int*>([result contents]);
            iterationCount = values[0];
            thresholdValues.assign(values + 2, values + 2 + values[1]);
        } else if (algorithm == Algorithm::Otsu) {
            thresholdValues.assign(1, *static_cast<const int*>([result contents]));
        }
    }
};

MetalProcessor::MetalProcessor(Algorithm algorithm, const Image& image, const Parameters& parameters) {
    @autoreleasepool {
        impl = std::make_unique<Impl>(algorithm, image, parameters);
    }
}

MetalProcessor::~MetalProcessor() = default;

void MetalProcessor::run(const Image& image, Image& output) {
    @autoreleasepool {
        impl->run(image, output);
    }
}

double MetalProcessor::kernelTimeUs() const { return impl->kernelUs; }
const std::vector<int>& MetalProcessor::thresholds() const { return impl->thresholdValues; }
int MetalProcessor::iterations() const { return impl->iterationCount; }
std::string MetalProcessor::profilingMethod() const { return impl->samples ? "gpu_timestamp_samples" : "unavailable"; }
