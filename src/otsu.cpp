#include "otsu.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

void calculateHistogramRange(const Image& image, Histogram& histogram, std::size_t begin, std::size_t end) {
    for (std::size_t i=begin; i<end; i++) {
        int pixel = image.pixels[i];
        ++histogram[pixel]; // if pixel is e.g 50, we increase histogram[50]
    }
}

Histogram calculateHistogram(const Image& image, int threads) {
    if (threads <= 0) throw std::invalid_argument("Thread count must be positive");
    // every thread counts its part into its own histogram, so no two threads increase the same counter
    std::vector<Histogram> partial(threads); // all counters start at zero
    parallelRanges(image.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int thread) {
        calculateHistogramRange(image, partial[thread], begin, end);
    });

    Histogram histogram{}; // resets all 256 counters to zero.
    for (const Histogram& part : partial) {
        for (int i=0; i<256; i++) {
            histogram[i] += part[i];
        }
    }
    return histogram;
}

int calculateOtsuThreshold(const Histogram& histogram) {
    std::size_t totalPixels = 0;
    double totalSum = 0.0;

    std::size_t backgroundPixels = 0; // number of pixels with brightness <= current threshold t
    double backgroundSum = 0.0; // brightness sum of those pixels

    for (int i =0; i<256; i++) {
        totalPixels += histogram[i];
        totalSum += i * histogram[i];
    }

    double maxVariance = 0.0;
    int bestThreshold = 0;
    for (int t =0; t<256; t++) {
        backgroundPixels += histogram[t];
        backgroundSum += t * histogram[t];

        std::size_t foregroundPixels = totalPixels - backgroundPixels;
        double foregroundSum = totalSum - backgroundSum;

        if (foregroundPixels == 0 || backgroundPixels == 0) {
            continue;
        }

        double foregroundAverage = foregroundSum / foregroundPixels;
        double backgroundAverage = backgroundSum / backgroundPixels;

        double backgroundWeight = static_cast<double>(backgroundPixels) / totalPixels;
        double foregroundWeight = static_cast<double>(foregroundPixels) / totalPixels;
        double difference = backgroundAverage - foregroundAverage;
        double betweenClassVariance = backgroundWeight * foregroundWeight * (difference * difference);
        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            bestThreshold = t;
        }
    }

    return bestThreshold;
}
                        // not const, because we will be modifying it
void applyThresholdRange(Image& image, int threshold, std::size_t begin, std::size_t end) {
    for (std::size_t i=begin; i<end; i++) {
        if (image.pixels[i] <= threshold)
        {
            image.pixels[i] = 0;
        }
        else
        {
            image.pixels[i] = 255;
        }
    }
}

Image otsuSegmentation(const Image& image, int threads) {
    Image result = image;
    int threshold = calculateOtsuThreshold(calculateHistogram(image, threads));
    parallelRanges(result.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        applyThresholdRange(result, threshold, begin, end);
    });
    return result;
}

std::vector<int> calculateMultiOtsuThresholds(const Histogram& histogram, int classes) {
    if (classes < 2) throw std::invalid_argument("Multi-Otsu needs at least two classes");

    std::vector<int> levels;
    std::vector<std::int64_t> prefixCount{0};
    std::vector<std::int64_t> prefixSum{0};
    for (int level = 0; level < static_cast<int>(histogram.size()); ++level) {
        const auto count = static_cast<std::int64_t>(histogram[level]);
        if (count == 0) continue;
        levels.push_back(level);
        prefixCount.push_back(prefixCount.back() + count);
        prefixSum.push_back(prefixSum.back() + level * count);
    }
    const int distinct = static_cast<int>(levels.size());
    if (distinct < classes) {
        throw std::invalid_argument("Multi-Otsu needs at least as many distinct brightness levels as classes");
    }

    auto gain = [&](int begin, int end) {
        const double count = static_cast<double>(prefixCount[end] - prefixCount[begin]);
        const double sum = static_cast<double>(prefixSum[end] - prefixSum[begin]);
        return sum * sum / count;
    };

    const int slack = distinct - classes;
    std::vector<double> previous(distinct + 1);
    std::vector<double> current(distinct + 1);
    std::vector<std::vector<int>> split(classes, std::vector<int>(distinct + 1));
    for (int end = 1; end <= slack + 1; ++end) previous[end] = gain(0, end);

    for (int k = 2; k <= classes; ++k) {
        auto solve = [&](auto& self, int lowEnd, int highEnd, int lowBegin, int highBegin) -> void {
            if (lowEnd > highEnd) return;
            const int end = (lowEnd + highEnd) / 2;
            double bestValue = -std::numeric_limits<double>::infinity();
            int bestBegin = lowBegin;
            for (int begin = lowBegin; begin <= std::min(highBegin, end - 1); ++begin) {
                const double value = previous[begin] + gain(begin, end);
                if (value > bestValue) {
                    bestValue = value;
                    bestBegin = begin;
                }
            }
            current[end] = bestValue;
            split[k - 1][end] = bestBegin;
            self(self, lowEnd, end - 1, lowBegin, bestBegin);
            self(self, end + 1, highEnd, bestBegin, highBegin);
        };
        solve(solve, k == classes ? distinct : k, slack + k, k - 1, slack + k - 1);
        std::swap(previous, current);
    }

    std::vector<int> thresholds(classes - 1);
    int end = distinct;
    for (int k = classes; k >= 2; --k) {
        const int begin = split[k - 1][end];
        thresholds[k - 2] = levels[begin - 1];
        end = begin;
    }
    return thresholds;
}

std::array<unsigned char, 256> multiOtsuLookupTable(const std::vector<int>& thresholds) {
    const int classes = static_cast<int>(thresholds.size()) + 1;
    std::array<unsigned char, 256> lookup{};
    int level = 0;
    for (int k = 0; k < classes; ++k) {
        const int last = k < classes - 1 ? thresholds[k] : 255;
        const auto value = static_cast<unsigned char>(255 * k / std::max(1, classes - 1));
        for (; level <= last; ++level) lookup[level] = value;
    }
    return lookup;
}

Image multiOtsuSegmentation(const Image& image, int classes, int threads) {
    const auto lookup = multiOtsuLookupTable(calculateMultiOtsuThresholds(calculateHistogram(image, threads), classes));
    Image segmented = image;
    parallelRanges(segmented.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        for (std::size_t i = begin; i < end; ++i) segmented.pixels[i] = lookup[segmented.pixels[i]];
    });
    return segmented;
}

OtsuGPU::OtsuGPU(const Image& image)
    : cl("kernels/otsu.cl"), pixelCount(image.pixels.size()) {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, pixelCount, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        histogramBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        thresholdBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, pixelCount, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        histogramKernel = clCreateKernel(cl.program, "histogramLocalKernel", &error);
        checkCL(error, "clCreateKernel");
        thresholdKernel = clCreateKernel(cl.program, "thresholdKernel", &error);
        checkCL(error, "clCreateKernel");
        otsuThresholdKernel = clCreateKernel(cl.program, "otsuThresholdKernel", &error);
        checkCL(error, "clCreateKernel");
        clearHistogramKernel = clCreateKernel(cl.program, "clearHistogramKernel", &error);
        checkCL(error, "clCreateKernel");

        int Npixels = static_cast<int>(pixelCount);
        checkCL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 3, sizeof(cl_mem), &thresholdBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(otsuThresholdKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(otsuThresholdKernel, 1, sizeof(cl_mem), &thresholdBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(clearHistogramKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");

    } catch (...) {
        release();
        throw;
    }
}

void OtsuGPU::release() {
    if (histogramKernel) clReleaseKernel(histogramKernel);
    if (thresholdKernel) clReleaseKernel(thresholdKernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (histogramBuffer) clReleaseMemObject(histogramBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
    if (thresholdBuffer) clReleaseMemObject(thresholdBuffer);
    if (otsuThresholdKernel) clReleaseKernel(otsuThresholdKernel);
    if (clearHistogramKernel) clReleaseKernel(clearHistogramKernel);
}

OtsuGPU::~OtsuGPU() {
    release();
}

// histogram, Otsu threshold and thresholding all on GPU: the image is sent once and only the result comes back
void OtsuGPU::run(const Image& image, Image& output, cl_event* histogramEvent, cl_event* otsuThresholdEvent, cl_event* thresholdEvent) {
    upload(image);
    compute(histogramEvent, otsuThresholdEvent, thresholdEvent);
    download(output);
}

void OtsuGPU::upload(const Image& image) {
    if (image.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, pixelCount, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
}

// only enqueues the work, the queue runs commands in order, so nothing has to wait between the kernels
void OtsuGPU::compute(cl_event* histogramEvent, cl_event* otsuThresholdEvent, cl_event* thresholdEvent) {
    const size_t localSize = 256;
    const size_t globalSize = 256*localSize; // 256 work-groups, so every GPU core gets several of them, with only 2 groups most cores stay idle
    // the histogram is zeroed on the GPU, so not even the 256 counters are copied from the CPU
    checkCL(clEnqueueNDRangeKernel(cl.queue, clearHistogramKernel, 1, nullptr, &localSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, histogramKernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, histogramEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, otsuThresholdKernel, 1, nullptr, &localSize, &localSize, 0, nullptr, otsuThresholdEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, thresholdKernel, 1, nullptr, &pixelCount, nullptr, 0, nullptr, thresholdEvent), "clEnqueueNDRangeKernel");
}

void OtsuGPU::finish() {
    checkCL(clFinish(cl.queue), "clFinish"); // waits until the GPU has done everything that was enqueued
}

void OtsuGPU::download(Image& output) {
    if (output.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, pixelCount, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}

Image otsuSegmentationGPU(const Image& image) {
    Image result = image;
    OtsuGPU gpu(image);
    gpu.run(image, result);
    return result;
}

MultiOtsuGPU::MultiOtsuGPU(const Image& image, int classes, bool zeroCopy)
    : cl("kernels/otsu.cl"), classes(classes), zeroCopy(zeroCopy), pixelCount(image.pixels.size()) {
    if (classes < 2 || classes > 256) throw std::invalid_argument("Multi-Otsu needs between 2 and 256 classes");
    thresholdValues.assign(classes - 1, 0);
    try {
        cl_int error;
        if (!zeroCopy) {
            inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, pixelCount, nullptr, &error);
            checkCL(error, "clCreateBuffer");
            outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, pixelCount, nullptr, &error);
            checkCL(error, "clCreateBuffer");
        }
        histogramBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        prefixCountBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 257 * sizeof(cl_ulong), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        prefixSumBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 257 * sizeof(cl_long), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        layersBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, classes * 257 * sizeof(cl_float), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        splitBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, (classes - 1) * 256 * sizeof(cl_ushort), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        const std::size_t thresholdsSize = thresholdValues.size() * sizeof(cl_int);
        if (zeroCopy) {
            thresholdsBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, thresholdsSize, thresholdValues.data(), &error);
        } else {
            thresholdsBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, thresholdsSize, nullptr, &error);
        }
        checkCL(error, "clCreateBuffer");
        lookupBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 256, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        clearHistogramKernel = clCreateKernel(cl.program, "clearHistogramKernel", &error);
        checkCL(error, "clCreateKernel");
        histogramKernel = clCreateKernel(cl.program, "histogramLocalKernel", &error);
        checkCL(error, "clCreateKernel");
        prefixKernel = clCreateKernel(cl.program, "multiOtsuPrefixKernel", &error);
        checkCL(error, "clCreateKernel");
        layerKernel = clCreateKernel(cl.program, "multiOtsuLayerKernel", &error);
        checkCL(error, "clCreateKernel");
        lookupTableKernel = clCreateKernel(cl.program, "multiOtsuLookupTableKernel", &error);
        checkCL(error, "clCreateKernel");
        lookupKernel = clCreateKernel(cl.program, "multiOtsuApplyLookupKernel", &error);
        checkCL(error, "clCreateKernel");

        int Npixels = static_cast<int>(pixelCount);
        checkCL(clSetKernelArg(clearHistogramKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(prefixKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(prefixKernel, 1, sizeof(cl_mem), &prefixCountBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(prefixKernel, 2, sizeof(cl_mem), &prefixSumBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(prefixKernel, 3, sizeof(cl_mem), &layersBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(layerKernel, 0, sizeof(cl_mem), &prefixCountBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(layerKernel, 1, sizeof(cl_mem), &prefixSumBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(layerKernel, 2, sizeof(cl_mem), &layersBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(layerKernel, 3, sizeof(cl_mem), &splitBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupTableKernel, 0, sizeof(cl_mem), &layersBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupTableKernel, 1, sizeof(cl_mem), &splitBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupTableKernel, 2, sizeof(int), &classes), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupTableKernel, 3, sizeof(cl_mem), &thresholdsBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupTableKernel, 4, sizeof(cl_mem), &lookupBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupKernel, 3, sizeof(cl_mem), &lookupBuffer), "clSetKernelArg");
        if (!zeroCopy) setImageArguments();
    } catch (...) {
        release();
        throw;
    }
}

void MultiOtsuGPU::release() {
    for (cl_kernel kernel : {clearHistogramKernel, histogramKernel, prefixKernel, layerKernel, lookupTableKernel, lookupKernel}) {
        if (kernel) clReleaseKernel(kernel);
    }
    for (cl_mem buffer : {inputBuffer, histogramBuffer, prefixCountBuffer, prefixSumBuffer, layersBuffer, splitBuffer,
                          thresholdsBuffer, lookupBuffer, outputBuffer}) {
        if (buffer) clReleaseMemObject(buffer);
    }
}

MultiOtsuGPU::~MultiOtsuGPU() {
    release();
}

void MultiOtsuGPU::setImageArguments() {
    checkCL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(lookupKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(lookupKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
}

void MultiOtsuGPU::bindImages(const Image& image, Image& output) {
    if (image.pixels.data() == output.pixels.data()) {
        throw std::invalid_argument("Zero-copy needs separate input and output images");
    }
    auto* input = const_cast<unsigned char*>(image.pixels.data());
    if (input == boundInput && output.pixels.data() == boundOutput) return;
    cl_int error;
    if (input != boundInput) {
        if (inputBuffer) clReleaseMemObject(inputBuffer);
        inputBuffer = nullptr;
        boundInput = nullptr;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, pixelCount, input, &error);
        checkCL(error, "clCreateBuffer");
        boundInput = input;
    }
    if (output.pixels.data() != boundOutput) {
        if (outputBuffer) clReleaseMemObject(outputBuffer);
        outputBuffer = nullptr;
        boundOutput = nullptr;
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, pixelCount, output.pixels.data(), &error);
        checkCL(error, "clCreateBuffer");
        boundOutput = output.pixels.data();
    }
    setImageArguments();
}

void MultiOtsuGPU::run(const Image& image, Image& output, cl_event* histogramEvent,
                       std::vector<cl_event>* thresholdsEvents, cl_event* lookupEvent) {
    upload(image, output);
    compute(histogramEvent, thresholdsEvents, lookupEvent);
    download(output);
}

void MultiOtsuGPU::upload(const Image& image, Image& output) {
    if (image.pixels.size() != pixelCount || output.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    if (!zeroCopy) {
        checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, pixelCount, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        return;
    }
    bindImages(image, output);
    cl_int error;
    void* mapped = clEnqueueMapBuffer(cl.queue, inputBuffer, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, pixelCount, 0, nullptr, nullptr, &error);
    checkCL(error, "clEnqueueMapBuffer");
    checkCL(clEnqueueUnmapMemObject(cl.queue, inputBuffer, mapped, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
}

void MultiOtsuGPU::compute(cl_event* histogramEvent, std::vector<cl_event>* thresholdsEvents, cl_event* lookupEvent) {
    const size_t localSize = 256;
    const size_t globalSize = 256 * localSize;
    auto nextEvent = [&]() -> cl_event* {
        if (!thresholdsEvents) return nullptr;
        thresholdsEvents->push_back(nullptr);
        return &thresholdsEvents->back();
    };
    checkCL(clEnqueueNDRangeKernel(cl.queue, clearHistogramKernel, 1, nullptr, &localSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, histogramKernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, histogramEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, prefixKernel, 1, nullptr, &localSize, &localSize, 0, nullptr, nextEvent()), "clEnqueueNDRangeKernel");
    for (int k = 2; k <= classes; ++k) {
        const int firstEnd = k == classes ? 256 : k;
        const size_t layerSize = static_cast<size_t>(257 - firstEnd) * localSize;
        checkCL(clSetKernelArg(layerKernel, 4, sizeof(int), &k), "clSetKernelArg");
        checkCL(clSetKernelArg(layerKernel, 5, sizeof(int), &firstEnd), "clSetKernelArg");
        checkCL(clEnqueueNDRangeKernel(cl.queue, layerKernel, 1, nullptr, &layerSize, &localSize, 0, nullptr, nextEvent()), "clEnqueueNDRangeKernel");
    }
    checkCL(clEnqueueNDRangeKernel(cl.queue, lookupTableKernel, 1, nullptr, &localSize, &localSize, 0, nullptr, nextEvent()), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, lookupKernel, 1, nullptr, &pixelCount, nullptr, 0, nullptr, lookupEvent), "clEnqueueNDRangeKernel");
}

void MultiOtsuGPU::finish() {
    checkCL(clFinish(cl.queue), "clFinish");
}

void MultiOtsuGPU::download(Image& output) {
    if (output.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    if (!zeroCopy) {
        checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_FALSE, 0, pixelCount, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
        checkCL(clEnqueueReadBuffer(cl.queue, thresholdsBuffer, CL_TRUE, 0, thresholdValues.size() * sizeof(cl_int), thresholdValues.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    } else {
        if (output.pixels.data() != boundOutput) {
            throw std::runtime_error("Output image differs from the one given to upload");
        }
        const std::size_t thresholdsSize = thresholdValues.size() * sizeof(cl_int);
        cl_int error;
        void* mappedOutput = clEnqueueMapBuffer(cl.queue, outputBuffer, CL_FALSE, CL_MAP_READ, 0, pixelCount, 0, nullptr, nullptr, &error);
        checkCL(error, "clEnqueueMapBuffer");
        void* mappedThresholds = clEnqueueMapBuffer(cl.queue, thresholdsBuffer, CL_TRUE, CL_MAP_READ, 0, thresholdsSize, 0, nullptr, nullptr, &error);
        checkCL(error, "clEnqueueMapBuffer");
        if (mappedOutput != output.pixels.data()) std::memcpy(output.pixels.data(), mappedOutput, pixelCount);
        if (mappedThresholds != thresholdValues.data()) std::memcpy(thresholdValues.data(), mappedThresholds, thresholdsSize);
        checkCL(clEnqueueUnmapMemObject(cl.queue, outputBuffer, mappedOutput, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
        checkCL(clEnqueueUnmapMemObject(cl.queue, thresholdsBuffer, mappedThresholds, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
    }
    if (thresholdValues.front() < 0) {
        throw std::invalid_argument("Multi-Otsu needs at least as many distinct brightness levels as classes");
    }
}

Image multiOtsuSegmentationGPU(const Image& image, int classes, bool zeroCopy) {
    Image result = image;
    MultiOtsuGPU gpu(image, classes, zeroCopy);
    gpu.run(image, result);
    return result;
}
