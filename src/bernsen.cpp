#include "bernsen.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr double sauvolaStddevRange = 128.0;
constexpr int localThresholdGroupWidth = 256;
constexpr int localThresholdColumnsPerItem = 8;
constexpr int localThresholdBandHeight = 16;

template<class Threshold>
void localThresholdRows(const Image& image, Image& result, int radius, int firstRow, int endRow,
                        Threshold thresholdFor) {
    const int width = image.width;
    const int height = image.height;
    std::vector<std::uint32_t> columnSum(width, 0);
    std::vector<std::uint32_t> columnSquares(width, 0);
    auto addRow = [&](int row) {
        const unsigned char* pixels = &image.pixels[static_cast<std::size_t>(row) * width];
        for (int x = 0; x < width; ++x) {
            columnSum[x] += pixels[x];
            columnSquares[x] += pixels[x] * pixels[x];
        }
    };
    auto removeRow = [&](int row) {
        const unsigned char* pixels = &image.pixels[static_cast<std::size_t>(row) * width];
        for (int x = 0; x < width; ++x) {
            columnSum[x] -= pixels[x];
            columnSquares[x] -= pixels[x] * pixels[x];
        }
    };

    for (int row = std::max(0, firstRow - radius); row <= std::min(height - 1, firstRow + radius); ++row) addRow(row);
    for (int y = firstRow; y < endRow; ++y) {
        if (y > firstRow) {
            if (y + radius < height) addRow(y + radius);
            if (y - radius - 1 >= 0) removeRow(y - radius - 1);
        }
        const int top = std::max(0, y - radius);
        const int bottom = std::min(height - 1, y + radius);
        std::uint64_t sum = 0;
        std::uint64_t sumOfSquares = 0;
        for (int x = 0; x <= std::min(radius, width - 1); ++x) {
            sum += columnSum[x];
            sumOfSquares += columnSquares[x];
        }

        for (int x = 0; x < width; ++x) {
            if (x > 0 && x + radius < width) {
                sum += columnSum[x + radius];
                sumOfSquares += columnSquares[x + radius];
            }
            if (x > 0 && x - radius - 1 >= 0) {
                sum -= columnSum[x - radius - 1];
                sumOfSquares -= columnSquares[x - radius - 1];
            }
            const int left = std::max(0, x - radius);
            const int right = std::min(width - 1, x + radius);
            const double count = static_cast<double>(right - left + 1) * (bottom - top + 1);
            const double mean = static_cast<double>(sum) / count;
            const double variance = static_cast<double>(sumOfSquares) / count - mean * mean;
            const double stddev = std::sqrt(std::max(0.0, variance));
            const double threshold = thresholdFor(mean, stddev);
            const auto index = static_cast<std::size_t>(y) * width + x;
            result.pixels[index] = image.pixels[index] < threshold ? 0 : 255;
        }
    }
}

template<class Threshold>
void localThreshold(const Image& image, Image& result, int radius, int threads, Threshold thresholdFor) {
    if (radius < 0) throw std::invalid_argument("Window radius must be non-negative");
    if (result.pixels.size() != image.pixels.size()) throw std::runtime_error("Output image size does not match the input");
    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        localThresholdRows(image, result, radius, static_cast<int>(begin), static_cast<int>(end), thresholdFor);
    });
}
}

void niblackThreshold(const Image& image, Image& result, int radius, double k, int threads) {
    localThreshold(image, result, radius, threads, [k](double mean, double stddev) {
        return mean - k * stddev;
    });
}

void sauvolaThreshold(const Image& image, Image& result, int radius, double k, int threads) {
    localThreshold(image, result, radius, threads, [k](double mean, double stddev) {
        return mean * (1.0 + k * (stddev / sauvolaStddevRange - 1.0));
    });
}

LocalThresholdGPU::LocalThresholdGPU(const Image& image, LocalThresholdMethod method, int radius, double k)
    : cl("kernels/bernsen.cl"), bufferSize(image.pixels.size()), width(image.width), height(image.height) {
    if (radius < 0 || 2 * radius >= localThresholdGroupWidth) {
        throw std::invalid_argument("GPU local threshold supports window radius from 0 to " +
                                    std::to_string(localThresholdGroupWidth / 2 - 1));
    }
    const std::size_t outputColumns = localThresholdGroupWidth * localThresholdColumnsPerItem;
    const std::size_t groupsX = (static_cast<std::size_t>(width) + outputColumns - 1) / outputColumns;
    const std::size_t groupsY = (static_cast<std::size_t>(height) + localThresholdBandHeight - 1) / localThresholdBandHeight;
    globalWorkSize[0] = groupsX * localThresholdGroupWidth;
    globalWorkSize[1] = groupsY;
    localWorkSize[0] = localThresholdGroupWidth;
    localWorkSize[1] = 1;
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, method == LocalThresholdMethod::Niblack ? "niblack" : "sauvola", &error);
        checkCL(error, "clCreateKernel");
        const float factor = static_cast<float>(k);
        const float stddevRange = static_cast<float>(sauvolaStddevRange);
        const int bandHeight = localThresholdBandHeight;
        const std::size_t localBytes = (outputColumns + 2 * radius) * sizeof(cl_uint);
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(width), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(height), &height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(radius), &radius), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(factor), &factor), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 6, sizeof(stddevRange), &stddevRange), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 7, sizeof(bandHeight), &bandHeight), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 8, localBytes, nullptr), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 9, localBytes, nullptr), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void LocalThresholdGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

LocalThresholdGPU::~LocalThresholdGPU() {
    release();
}

void LocalThresholdGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (image.width != width || image.height != height || image.pixels.size() != bufferSize ||
        output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}

void bernsenRows(const Image &image, Image &result, int radius, int minContrast, int firstRow, int endRow) {
    for (int y=firstRow; y<endRow; y++) {
        for (int x=0; x<image.width; x++) {
            int left = std::max(x- radius, 0);
            int right = std::min(image.width-1, x+radius);
            int top = std::max(y-radius, 0);
            int bottom = std::min(image.height-1,y+radius);
            int Jmin  = 255; // because we're going to be searching by std::min and update the value
            int Jmax  = 0;
            for (int ny=top; ny<=bottom; ny++) {
                for (int nx=left; nx<=right; nx++) {
                    std::size_t index = ny*image.width + nx;
                    int value = image.pixels[index];
                    Jmin = std::min(Jmin, value);
                    Jmax = std::max(Jmax, value);
                }
            }
            int contrast = Jmax - Jmin;
            if (contrast < minContrast) {
                result.pixels[y*image.width + x] = 0;
            }
            else {
                int threshold = (Jmax + Jmin)/2;
                int pixel = image.pixels[y*image.width + x];
                //result.pixels[y*image.width + x] = pixel < threshold ? 0 : 255; // Andrea Latina speaking....
                if (pixel < threshold) {
                    result.pixels[y*image.width + x] = 0;
                }

                else {
                    result.pixels[y*image.width + x] = 255;
                }
            }
        }
    }
}

void bernsenThreshold(const Image &image, Image &result, int radius, int minContrast, int threads) {
    // we need to divide the rows for threads and for each range call bernsenRows

    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        bernsenRows(image, result, radius, minContrast, static_cast<int>(begin), static_cast<int>(end));
    });
}

BernsenGPU::BernsenGPU(const Image& image, int radius, int minContrast)
    : cl("kernels/bernsen.cl"),
      bufferSize(image.pixels.size()),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "bernsen", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(radius), &radius), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(minContrast), &minContrast), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void BernsenGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

BernsenGPU::~BernsenGPU() {
    release();
}

void BernsenGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(image.width) != globalWorkSize[0] || static_cast<std::size_t>(image.height) != globalWorkSize[1] ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel");
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}
