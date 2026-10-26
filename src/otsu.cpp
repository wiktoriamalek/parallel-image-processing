#include "otsu.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
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

OtsuGPU::OtsuGPU(const Image& image)
    : cl("kernels/otsu.cl"), pixelCount(image.pixels.size()) {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, pixelCount, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        histogramBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, pixelCount, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        histogramKernel = clCreateKernel(cl.program, "histogramLocalKernel", &error);
        checkCL(error, "clCreateKernel");
        thresholdKernel = clCreateKernel(cl.program, "thresholdKernel", &error);
        checkCL(error, "clCreateKernel");

        int Npixels = static_cast<int>(pixelCount);
        checkCL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(histogramKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(thresholdKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
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
}

OtsuGPU::~OtsuGPU() {
    release();
}

// histogram on GPU -> Otsu threshold on CPU -> thresholding on GPU, the image is sent to the GPU only once
Histogram OtsuGPU::run(const Image& image, Image& output, cl_event* histogramEvent, cl_event* thresholdEvent) {
    const size_t localSize = 256;
    const size_t globalSize = 256*localSize; // 256 work-groups, so every GPU core gets several of them, with only 2 groups most cores stay idle
    if (image.pixels.size() != pixelCount || output.pixels.size() != pixelCount) {
        throw std::runtime_error("Image size does not match the GPU buffers");
    }
    std::array<cl_uint, 256> counts{};
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, pixelCount, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueWriteBuffer(cl.queue, histogramBuffer, CL_TRUE, 0, sizeof(counts), counts.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, histogramKernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, histogramEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, histogramBuffer, CL_TRUE, 0, sizeof(counts), counts.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done

    Histogram histogram{};
    std::copy(counts.begin(), counts.end(), histogram.begin());
    int threshold = calculateOtsuThreshold(histogram);
    checkCL(clSetKernelArg(thresholdKernel, 3, sizeof(int), &threshold), "clSetKernelArg");
    checkCL(clEnqueueNDRangeKernel(cl.queue, thresholdKernel, 1, nullptr, &pixelCount, nullptr, 0, nullptr, thresholdEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, pixelCount, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
    return histogram;
}

Image otsuSegmentationGPU(const Image& image) {
    Image result = image;
    OtsuGPU gpu(image);
    gpu.run(image, result);
    return result;
}
