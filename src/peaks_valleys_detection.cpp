#include "peaks_valleys_detection.hpp"
#include "parallel_utils.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int fixedPointShift = 30;
constexpr int block = 16;

std::uint64_t valueAt(const SmoothedHistogram& histogram, int intensity) {
    return intensity < 0 || intensity > 255 ? 0 : histogram[intensity];
}

int plateauEnd(const SmoothedHistogram& histogram, int start) {
    int end = start;
    while (end < 255 && histogram[end + 1] == histogram[start]) ++end;
    return end;
}

void findPeaks(const SmoothedHistogram& histogram, std::vector<int>& peaks) {
    peaks.clear();
    for (int i = 0; i < 256; ++i) {
        if (histogram[i] <= valueAt(histogram, i - 1)) continue;
        const int end = plateauEnd(histogram, i);
        if (valueAt(histogram, end + 1) < histogram[i]) peaks.push_back((i + end) / 2);
        i = end;
    }
}

void smooth(const SmoothedHistogram& input, SmoothedHistogram& output) {
    output[0] = (2 * input[0] + input[1]) >> 2;
    for (int i = 1; i < 255; ++i) output[i] = (input[i - 1] + 2 * input[i] + input[i + 1]) >> 2;
    output[255] = (input[254] + 2 * input[255]) >> 2;
}
}

std::vector<int> peaks_valleys_detectionThresholds(const Histogram& histogram, int maxClasses, int* iterations)
{
    if (maxClasses < 2 || maxClasses > 256) throw std::invalid_argument("Valley detection needs between 2 and 256 classes");
    SmoothedHistogram buffers[2];
    for (int i = 0; i < 256; ++i) {
        if (histogram[i] > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("Valley detection supports at most 2^32 - 1 pixels of one intensity");
        }
        buffers[0][i] = static_cast<std::uint64_t>(histogram[i]) << fixedPointShift;
    }

    int current = 0;
    int steps = 0;
    SmoothedHistogram blockStart;
    std::vector<int> peaks;
    findPeaks(buffers[current], peaks);
    while (static_cast<int>(peaks.size()) > maxClasses) {
        blockStart = buffers[current];
        for (int step = 0; step < block; ++step) {
            smooth(buffers[current], buffers[1 - current]);
            current = 1 - current;
        }
        steps += block;
        findPeaks(buffers[current], peaks);
    }

    if (steps > 0) {
        buffers[current] = blockStart;
        steps -= block;
        do {
            smooth(buffers[current], buffers[1 - current]);
            current = 1 - current;
            ++steps;
            findPeaks(buffers[current], peaks);
        } while (static_cast<int>(peaks.size()) > maxClasses);
    }

    std::vector<int> thresholds;
    for (std::size_t i = 1; i < peaks.size(); ++i)
    {
        thresholds.push_back(peaks_valleys_detectionRange(buffers[current], peaks[i - 1], peaks[i]));
    }
    if (iterations) *iterations = steps;
    return thresholds;
}

int peaks_valleys_detectionRange(const SmoothedHistogram& histogram, int left_peak_intensity, int right_peak_intensity)
{
    if (left_peak_intensity < 0 || right_peak_intensity > 255 || right_peak_intensity - left_peak_intensity < 2) {
        throw std::invalid_argument("Choose better intensities. Left should be at least 2 values lower than the right one.");
    }
    int first = left_peak_intensity + 1;
    int last = first;
    for (int j = first; j < right_peak_intensity; ++j)
    {
        if (histogram[j] < histogram[first]) {
            first = j;
            last = j;
        } else if (histogram[j] == histogram[first]) {
            last = j;
        }
    }
    return (first + last) / 2;
}

Image peaks_valleys_detection(const Image& image, int maxClasses)
{
    Image peaks_valleys_detection_image = image;
    peaks_valleys_detectionParallel(image, peaks_valleys_detection_image, 1, maxClasses);
    return peaks_valleys_detection_image;
}

void peaks_valleys_detectionParallel(const Image& image, Image& output, int threads, int maxClasses)
{
    if (output.pixels.size() != image.pixels.size()) {
        throw std::runtime_error("Output image size does not match the input");
    }
    const std::vector<int> thresholds = peaks_valleys_detectionThresholds(calculateHistogram(image, threads), maxClasses);
    const std::array<unsigned char, 256> lookup = multiOtsuLookupTable(thresholds);
    parallelRanges(image.pixels.size(), threads, [&](std::size_t begin, std::size_t end, int) {
        const unsigned char* input = image.pixels.data();
        unsigned char* result = output.pixels.data();
        for (std::size_t i = begin; i < end; ++i) result[i] = lookup[input[i]];
    });
}

Peaks_valleys_detectionGPU::Peaks_valleys_detectionGPU(const Image& image, int maxClasses, bool zeroCopy)
    : cl(std::vector<std::string>{"kernels/otsu.cl", "kernels/peaks_valleys_detection.cl"}),
      maxClasses(maxClasses), zeroCopy(zeroCopy), pixelCount(image.pixels.size()) {
    if (maxClasses < 2 || maxClasses > 256) throw std::invalid_argument("Valley detection needs between 2 and 256 classes");
    if (pixelCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Image is too large for the GPU kernels");
    }
    result.assign(maxClasses + 1, 0);
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
        resultBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, result.size() * sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        lookupBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 256, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        clearHistogramKernel = clCreateKernel(cl.program, "clearHistogramKernel", &error);
        checkCL(error, "clCreateKernel");
        histogramKernel = clCreateKernel(cl.program, "histogramLocalKernel", &error);
        checkCL(error, "clCreateKernel");
        thresholdsKernel = clCreateKernel(cl.program, "peaksValleysThresholdsKernel", &error);
        checkCL(error, "clCreateKernel");
        lookupKernel = clCreateKernel(cl.program, "multiOtsuApplyLookupKernel", &error);
        checkCL(error, "clCreateKernel");

        int Npixels = static_cast<int>(pixelCount);
        checkCL(clSetKernelArg(clearHistogramKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdsKernel, 0, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdsKernel, 1, sizeof(int), &maxClasses), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdsKernel, 2, sizeof(cl_mem), &resultBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdsKernel, 3, sizeof(cl_mem), &lookupBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(lookupKernel, 3, sizeof(cl_mem), &lookupBuffer), "clSetKernelArg");
        if (!zeroCopy) setImageArguments();
    } catch (...) {
        release();
        throw;
    }
}

void Peaks_valleys_detectionGPU::release() {
    for (cl_kernel kernel : {clearHistogramKernel, histogramKernel, thresholdsKernel, lookupKernel}) {
        if (kernel) clReleaseKernel(kernel);
    }
    for (cl_mem buffer : {inputBuffer, histogramBuffer, resultBuffer, lookupBuffer, outputBuffer}) {
        if (buffer) clReleaseMemObject(buffer);
    }
}

Peaks_valleys_detectionGPU::~Peaks_valleys_detectionGPU() {
    release();
}

void Peaks_valleys_detectionGPU::setImageArguments() {
    checkCL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(lookupKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(lookupKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
}

void Peaks_valleys_detectionGPU::bindImages(const Image& image, Image& output) {
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

void Peaks_valleys_detectionGPU::run(const Image& image, Image& output, cl_event* histogramEvent,
                                     cl_event* thresholdsEvent, cl_event* lookupEvent) {
    upload(image, output);
    compute(histogramEvent, thresholdsEvent, lookupEvent);
    download(output);
}

void Peaks_valleys_detectionGPU::upload(const Image& image, Image& output) {
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

void Peaks_valleys_detectionGPU::compute(cl_event* histogramEvent, cl_event* thresholdsEvent, cl_event* lookupEvent) {
    const size_t localSize = 256;
    const size_t globalSize = 256 * localSize;
    checkCL(clEnqueueNDRangeKernel(cl.queue, clearHistogramKernel, 1, nullptr, &localSize, &localSize, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, histogramKernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, histogramEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, thresholdsKernel, 1, nullptr, &localSize, &localSize, 0, nullptr, thresholdsEvent), "clEnqueueNDRangeKernel");
    enqueueLine(cl.queue, lookupKernel, pixelCount, lookupEvent);
}

void Peaks_valleys_detectionGPU::finish() {
    checkCL(clFinish(cl.queue), "clFinish");
}

void Peaks_valleys_detectionGPU::download(Image& output) {
    if (output.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    const std::size_t resultSize = result.size() * sizeof(cl_int);
    if (!zeroCopy) {
        checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_FALSE, 0, pixelCount, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
        checkCL(clEnqueueReadBuffer(cl.queue, resultBuffer, CL_TRUE, 0, resultSize, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    } else {
        if (output.pixels.data() != boundOutput) {
            throw std::runtime_error("Output image differs from the one given to upload");
        }
        cl_int error;
        void* mappedOutput = clEnqueueMapBuffer(cl.queue, outputBuffer, CL_FALSE, CL_MAP_READ, 0, pixelCount, 0, nullptr, nullptr, &error);
        checkCL(error, "clEnqueueMapBuffer");
        checkCL(clEnqueueReadBuffer(cl.queue, resultBuffer, CL_TRUE, 0, resultSize, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
        if (mappedOutput != output.pixels.data()) std::memcpy(output.pixels.data(), mappedOutput, pixelCount);
        checkCL(clEnqueueUnmapMemObject(cl.queue, outputBuffer, mappedOutput, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
    }
    iterationCount = result[0];
    thresholdValues.assign(result.begin() + 2, result.begin() + 2 + result[1]);
}
