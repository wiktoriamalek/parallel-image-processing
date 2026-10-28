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
