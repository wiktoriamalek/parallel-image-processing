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
constexpr std::size_t bernsenRowOutputs = 2048;
constexpr std::size_t bernsenTileColumns = 32;
constexpr std::size_t bernsenTileRows = 64;
constexpr std::size_t bernsenTileLanes = 8;

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

namespace {
struct MinMaxRows {
    std::vector<unsigned char> low, high;

    void resize(std::size_t size) {
        low.resize(size);
        high.resize(size);
    }
};

struct BernsenScratch {
    MinMaxRows padded, prefix, suffix;
    MinMaxRows current, next, nextPrefix;
};

void rowMinMax(const unsigned char* row, int width, int radius, unsigned char* minimum, unsigned char* maximum,
               BernsenScratch& scratch) {
    const int window = 2 * radius + 1;
    const int length = width + 2 * radius;
    scratch.padded.resize(length);
    scratch.prefix.resize(length);
    scratch.suffix.resize(length);
    unsigned char* low = scratch.padded.low.data();
    unsigned char* high = scratch.padded.high.data();
    std::fill(low, low + length, 255);
    std::fill(high, high + length, 0);
    std::copy(row, row + width, low + radius);
    std::copy(row, row + width, high + radius);
    unsigned char* prefixLow = scratch.prefix.low.data();
    unsigned char* prefixHigh = scratch.prefix.high.data();
    unsigned char* suffixLow = scratch.suffix.low.data();
    unsigned char* suffixHigh = scratch.suffix.high.data();

    for (int start = 0; start < length; start += window) {
        const int stop = std::min(start + window, length);
        prefixLow[start] = low[start];
        prefixHigh[start] = high[start];
        for (int i = start + 1; i < stop; ++i) {
            prefixLow[i] = std::min(prefixLow[i - 1], low[i]);
            prefixHigh[i] = std::max(prefixHigh[i - 1], high[i]);
        }
        suffixLow[stop - 1] = low[stop - 1];
        suffixHigh[stop - 1] = high[stop - 1];
        for (int i = stop - 2; i >= start; --i) {
            suffixLow[i] = std::min(suffixLow[i + 1], low[i]);
            suffixHigh[i] = std::max(suffixHigh[i + 1], high[i]);
        }
    }
    for (int x = 0; x < width; ++x) {
        minimum[x] = std::min(suffixLow[x], prefixLow[x + 2 * radius]);
        maximum[x] = std::max(suffixHigh[x], prefixHigh[x + 2 * radius]);
    }
}

void loadBlock(const Image& image, int radius, int blockStart, MinMaxRows& block, BernsenScratch& scratch) {
    const int width = image.width;
    const int window = 2 * radius + 1;
    block.resize(static_cast<std::size_t>(window) * width);
    for (int i = 0; i < window; ++i) {
        const int y = blockStart + i - radius;
        unsigned char* low = block.low.data() + static_cast<std::size_t>(i) * width;
        unsigned char* high = block.high.data() + static_cast<std::size_t>(i) * width;
        if (y < 0 || y >= image.height) {
            std::fill(low, low + width, 255);
            std::fill(high, high + width, 0);
        } else {
            rowMinMax(&image.pixels[static_cast<std::size_t>(y) * width], width, radius, low, high, scratch);
        }
    }
}

void suffixInPlace(MinMaxRows& block, int window, int width) {
    for (int i = window - 2; i >= 0; --i) {
        unsigned char* low = block.low.data() + static_cast<std::size_t>(i) * width;
        unsigned char* high = block.high.data() + static_cast<std::size_t>(i) * width;
        for (int x = 0; x < width; ++x) {
            low[x] = std::min(low[x], low[x + width]);
            high[x] = std::max(high[x], high[x + width]);
        }
    }
}

void prefixOf(const MinMaxRows& block, MinMaxRows& prefix, int window, int width) {
    prefix.resize(block.low.size());
    std::copy(block.low.begin(), block.low.begin() + width, prefix.low.begin());
    std::copy(block.high.begin(), block.high.begin() + width, prefix.high.begin());
    for (int i = 1; i < window; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int x = 0; x < width; ++x) {
            prefix.low[row + x] = std::min(prefix.low[row - width + x], block.low[row + x]);
            prefix.high[row + x] = std::max(prefix.high[row - width + x], block.high[row + x]);
        }
    }
}

void bernsenRowRange(const Image& image, Image& result, int radius, int minContrast, int firstRow, int endRow,
                     BernsenScratch& scratch) {
    const int width = image.width;
    const int window = 2 * radius + 1;
    loadBlock(image, radius, firstRow, scratch.current, scratch);
    suffixInPlace(scratch.current, window, width);
    for (int blockStart = firstRow; blockStart < endRow; blockStart += window) {
        loadBlock(image, radius, blockStart + window, scratch.next, scratch);
        prefixOf(scratch.next, scratch.nextPrefix, window, width);
        for (int i = 0; i < window && blockStart + i < endRow; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) * width;
            const std::size_t previousRow = static_cast<std::size_t>(std::max(i - 1, 0)) * width;
            const unsigned char* suffixLow = scratch.current.low.data() + row;
            const unsigned char* suffixHigh = scratch.current.high.data() + row;
            const unsigned char* prefixLow = (i == 0 ? scratch.current.low.data() : scratch.nextPrefix.low.data()) + previousRow;
            const unsigned char* prefixHigh = (i == 0 ? scratch.current.high.data() : scratch.nextPrefix.high.data()) + previousRow;
            const std::size_t offset = static_cast<std::size_t>(blockStart + i) * width;
            const unsigned char* pixels = &image.pixels[offset];
            unsigned char* output = &result.pixels[offset];
            for (int x = 0; x < width; ++x) {
                const int Jmin = std::min(suffixLow[x], prefixLow[x]);
                const int Jmax = std::max(suffixHigh[x], prefixHigh[x]);
                const int threshold = (Jmax + Jmin) / 2;
                output[x] = Jmax - Jmin < minContrast || pixels[x] < threshold ? 0 : 255;
            }
        }
        suffixInPlace(scratch.next, window, width);
        std::swap(scratch.current, scratch.next);
    }
}
}

void bernsenThreshold(const Image &image, Image &result, int radius, int minContrast, int threads) {
    if (radius < 0) throw std::invalid_argument("Window radius must be non-negative");
    if (result.pixels.size() != image.pixels.size()) throw std::runtime_error("Output image size does not match the input");
    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        BernsenScratch scratch;
        bernsenRowRange(image, result, radius, minContrast, static_cast<int>(begin), static_cast<int>(end), scratch);
    });
}

BernsenGPU::BernsenGPU(const Image& image, int radius, int minContrast)
    : cl("kernels/bernsen.cl"),
      bufferSize(image.pixels.size()),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    if (radius < 0) throw std::invalid_argument("Window radius must be non-negative");
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        rowMinMaxBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, 2 * bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        cl_ulong localMemory = 0;
        checkCL(clGetDeviceInfo(cl.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemory), &localMemory, nullptr), "clGetDeviceInfo");
        const std::size_t rowBytes = 4 * (bernsenRowOutputs + 2 * static_cast<std::size_t>(radius));
        const std::size_t columnBytes = 2 * (bernsenTileRows + 2 * static_cast<std::size_t>(radius)) * bernsenTileColumns * 2;
        rowLocal = rowBytes <= localMemory;
        columnLocal = columnBytes <= localMemory;
        rowKernel = clCreateKernel(cl.program, rowLocal ? "bernsenRowMinMaxLocal" : "bernsenRowMinMax", &error);
        checkCL(error, "clCreateKernel");
        columnKernel = clCreateKernel(cl.program, columnLocal ? "bernsenColumnsLocal" : "bernsenColumns", &error);
        checkCL(error, "clCreateKernel");
        if (rowLocal) checkCL(clSetKernelArg(rowKernel, 5, rowBytes, nullptr), "clSetKernelArg");
        if (columnLocal) checkCL(clSetKernelArg(columnKernel, 7, columnBytes, nullptr), "clSetKernelArg");
        checkCL(clSetKernelArg(rowKernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(rowKernel, 1, sizeof(rowMinMaxBuffer), &rowMinMaxBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(rowKernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(rowKernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(rowKernel, 4, sizeof(radius), &radius), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 1, sizeof(rowMinMaxBuffer), &rowMinMaxBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 2, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 3, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 4, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 5, sizeof(radius), &radius), "clSetKernelArg");
        checkCL(clSetKernelArg(columnKernel, 6, sizeof(minContrast), &minContrast), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void BernsenGPU::release() {
    if (rowKernel) clReleaseKernel(rowKernel);
    if (columnKernel) clReleaseKernel(columnKernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (rowMinMaxBuffer) clReleaseMemObject(rowMinMaxBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

BernsenGPU::~BernsenGPU() {
    release();
}

void BernsenGPU::run(const Image& image, Image& output, cl_event* rowEvent, cl_event* columnEvent) {
    if (static_cast<std::size_t>(image.width) != globalWorkSize[0] || static_cast<std::size_t>(image.height) != globalWorkSize[1] ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    const std::size_t rowGlobal[2] = {(globalWorkSize[0] + bernsenRowOutputs - 1) / bernsenRowOutputs * 256, globalWorkSize[1]};
    const std::size_t rowGroup[2] = {256, 1};
    const std::size_t columnGlobal[2] = {(globalWorkSize[0] + bernsenTileColumns - 1) / bernsenTileColumns * bernsenTileColumns,
                                         (globalWorkSize[1] + bernsenTileRows - 1) / bernsenTileRows * bernsenTileLanes};
    const std::size_t columnGroup[2] = {bernsenTileColumns, bernsenTileLanes};
    checkCL(clEnqueueNDRangeKernel(cl.queue, rowKernel, 2, nullptr, rowLocal ? rowGlobal : globalWorkSize,
                                   rowLocal ? rowGroup : nullptr, 0, nullptr, rowEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueNDRangeKernel(cl.queue, columnKernel, 2, nullptr, columnLocal ? columnGlobal : globalWorkSize,
                                   columnLocal ? columnGroup : nullptr, 0, nullptr, columnEvent), "clEnqueueNDRangeKernel");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}
