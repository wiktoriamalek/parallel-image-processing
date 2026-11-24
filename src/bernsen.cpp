#include "bernsen.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
constexpr int sauvolaStddevRange = 128;
constexpr std::int64_t sauvolaMaxWindowPixels = 1000000;
constexpr std::int64_t niblackMaxWindowPixels = 10000000;
constexpr std::int64_t niblackMaxSmallWindowPixels = 160000;
constexpr int localThresholdGroupWidth = 256;
constexpr int localThresholdColumnsPerItem = 8;
constexpr int localThresholdMinBandHeight = 16;
constexpr int localThresholdMaxBandHeight = 64;
constexpr std::size_t bernsenRowOutputs = 2048;
constexpr std::size_t bernsenTileColumns = 32;
constexpr std::size_t bernsenTileRows = 64;
constexpr std::size_t bernsenTileLanes = 8;

int localThresholdBandHeight(int radius) {
    int band = localThresholdMinBandHeight;
    while (band < radius && band < localThresholdMaxBandHeight) band *= 2;
    return band;
}

template<class Classify, class Exact>
void localThresholdRows(const Image& image, Image& result, int radius, int firstRow, int endRow,
                        Classify classify, Exact exact) {
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
            const std::int64_t count = static_cast<std::int64_t>(right - left + 1) * (bottom - top + 1);
            const auto index = static_cast<std::size_t>(y) * width + x;
            result.pixels[index] = classify(count, static_cast<std::int64_t>(sum), static_cast<std::int64_t>(sumOfSquares),
                                            image.pixels[index]);
        }

        if constexpr (std::is_same_v<Exact, std::nullptr_t>) continue;
        unsigned char* row = &result.pixels[static_cast<std::size_t>(y) * width];
        for (auto* unsure = static_cast<unsigned char*>(std::memchr(row, 1, width)); unsure != nullptr;
             unsure = static_cast<unsigned char*>(std::memchr(unsure + 1, 1, static_cast<std::size_t>(row + width - unsure - 1)))) {
            const int x = static_cast<int>(unsure - row);
            const int left = std::max(0, x - radius);
            const int right = std::min(width - 1, x + radius);
            std::int64_t windowSum = 0;
            std::int64_t windowSquares = 0;
            for (int column = left; column <= right; ++column) {
                windowSum += columnSum[column];
                windowSquares += columnSquares[column];
            }
            const std::int64_t count = static_cast<std::int64_t>(right - left + 1) * (bottom - top + 1);
            if constexpr (!std::is_same_v<Exact, std::nullptr_t>) {
                *unsure = exact(count, windowSum, windowSquares, image.pixels[static_cast<std::size_t>(y) * width + x]) ? 0 : 255;
            }
        }
    }
}

template<class Classify, class Exact>
void localThreshold(const Image& image, Image& result, int radius, int threads, Classify classify, Exact exact) {
    if (radius < 0) throw std::invalid_argument("Window radius must be non-negative");
    if (result.pixels.size() != image.pixels.size()) throw std::runtime_error("Output image size does not match the input");
    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        localThresholdRows(image, result, radius, static_cast<int>(begin), static_cast<int>(end), classify, exact);
    });
}
}

int localThresholdHundredths(double k) {
    const double scaled = k * 100.0;
    const double rounded = std::round(scaled);
    if (!std::isfinite(scaled) || rounded < 0.0 || rounded > 100.0 || std::abs(scaled - rounded) > 1e-6) {
        throw std::invalid_argument("k must be between 0 and 1 with at most two decimal places");
    }
    return static_cast<int>(rounded);
}

static inline bool niblackSmallWindow(std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel,
                                      std::uint64_t factor) {
    const std::int64_t difference = 100 * (sum - count * pixel);
    const std::uint64_t magnitude = static_cast<std::uint64_t>(difference);
    const std::uint64_t variance = static_cast<std::uint64_t>(count * squares - sum * sum);
    return (difference > 0) & (magnitude * magnitude > factor * variance);
}

bool niblackIsBlack(std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel, int hundredthsK) {
    const std::int64_t factor = static_cast<std::int64_t>(hundredthsK) * hundredthsK;
    if (count <= niblackMaxSmallWindowPixels) {
        return niblackSmallWindow(count, sum, squares, pixel, static_cast<std::uint64_t>(factor));
    }
    const std::int64_t difference = 100 * (sum - count * pixel);
    const std::int64_t variance = count * squares - sum * sum;
    return difference > 0 && static_cast<__int128>(difference) * difference > static_cast<__int128>(factor) * variance;
}

static inline unsigned char sauvolaClassify(std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel,
                                            int hundredthsK) {
    const std::int64_t variance = count * squares - sum * sum;
    const double n = static_cast<double>(count);
    const double s = static_cast<double>(sum);
    const double left = n * sauvolaStddevRange * (100.0 * n * pixel - (100 - hundredthsK) * s);
    const double right = s * hundredthsK;
    const double leftSquared = left * left;
    const double rightSquared = right * right * static_cast<double>(variance);
    const double difference = leftSquared - rightSquared;
    const bool unsure = (left >= 0.0) & (std::abs(difference) <= 1e-9 * (leftSquared + rightSquared));
    const bool black = (left < 0.0) | (difference < 0.0);
    return unsure ? 1 : (black ? 0 : 255);
}

bool sauvolaIsBlack(std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel, int hundredthsK) {
    const unsigned char code = sauvolaClassify(count, sum, squares, pixel, hundredthsK);
    if (code != 1) return code == 0;
    const std::int64_t left = count * sauvolaStddevRange * (100 * count * pixel - (100 - hundredthsK) * sum);
    const std::int64_t variance = count * squares - sum * sum;
    const std::int64_t right = sum * hundredthsK;
    return left < 0 || static_cast<__int128>(left) * left < static_cast<__int128>(right) * right * variance;
}

static std::int64_t windowPixels(const Image& image, int radius) {
    const std::int64_t side = 2 * static_cast<std::int64_t>(radius) + 1;
    return std::min<std::int64_t>(side, image.width) * std::min<std::int64_t>(side, image.height);
}

void niblackThreshold(const Image& image, Image& result, int radius, double k, int threads) {
    const int hundredthsK = localThresholdHundredths(k);
    const std::int64_t pixels = radius >= 0 ? windowPixels(image, radius) : 0;
    if (pixels > niblackMaxWindowPixels) {
        throw std::invalid_argument("Niblack window has too many pixels for exact thresholding");
    }
    if (pixels <= niblackMaxSmallWindowPixels) {
        const auto factor = static_cast<std::uint64_t>(hundredthsK) * static_cast<std::uint64_t>(hundredthsK);
        localThreshold(image, result, radius, threads, [factor](std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel) {
            return static_cast<unsigned char>(niblackSmallWindow(count, sum, squares, pixel, factor) ? 0 : 255);
        }, nullptr);
        return;
    }
    localThreshold(image, result, radius, threads, [hundredthsK](std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel) {
        return static_cast<unsigned char>(niblackIsBlack(count, sum, squares, pixel, hundredthsK) ? 0 : 255);
    }, nullptr);
}

void sauvolaThreshold(const Image& image, Image& result, int radius, double k, int threads) {
    const int hundredthsK = localThresholdHundredths(k);
    if (radius >= 0 && windowPixels(image, radius) > sauvolaMaxWindowPixels) {
        throw std::invalid_argument("Sauvola window has too many pixels for exact thresholding");
    }
    localThreshold(image, result, radius, threads, [hundredthsK](std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel) {
        return sauvolaClassify(count, sum, squares, pixel, hundredthsK);
    }, [hundredthsK](std::int64_t count, std::int64_t sum, std::int64_t squares, int pixel) {
        return sauvolaIsBlack(count, sum, squares, pixel, hundredthsK);
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
    const int bandHeight = localThresholdBandHeight(radius);
    const std::size_t groupsY = (static_cast<std::size_t>(height) + bandHeight - 1) / bandHeight;
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
        const float factor = static_cast<float>(localThresholdHundredths(k)) / 100.0f;
        const float stddevRange = static_cast<float>(sauvolaStddevRange);
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
        const std::size_t blockBytes = (outputColumns + 2 * radius + localThresholdColumnsPerItem - 1) /
                                       localThresholdColumnsPerItem * sizeof(cl_uint);
        checkCL(clSetKernelArg(kernel, 10, blockBytes, nullptr), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 11, blockBytes, nullptr), "clSetKernelArg");
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
            int sum = Jmax + Jmin;
            if (contrast < minContrast) {
                result.pixels[y*image.width + x] = sum < 256 ? 0 : 255;
            }
            else {
                int pixel = image.pixels[y*image.width + x];
                //result.pixels[y*image.width + x] = pixel < threshold ? 0 : 255; // Andrea Latina speaking....
                if (2 * pixel < sum) {
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
                const int sum = Jmax + Jmin;
                output[x] = (Jmax - Jmin < minContrast ? sum < 256 : 2 * pixels[x] < sum) ? 0 : 255;
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
    if (threads <= 0) throw std::invalid_argument("Thread count must be positive");
    std::vector<BernsenScratch> scratch(threads);
    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int thread) {
        bernsenRowRange(image, result, radius, minContrast, static_cast<int>(begin), static_cast<int>(end), scratch[thread]);
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
    if (rowLocal) {
        checkCL(clEnqueueNDRangeKernel(cl.queue, rowKernel, 2, nullptr, rowGlobal, rowGroup, 0, nullptr, rowEvent), "clEnqueueNDRangeKernel");
    } else {
        enqueueTiles(cl.queue, rowKernel, globalWorkSize[0], globalWorkSize[1], rowEvent);
    }
    if (columnLocal) {
        checkCL(clEnqueueNDRangeKernel(cl.queue, columnKernel, 2, nullptr, columnGlobal, columnGroup, 0, nullptr, columnEvent), "clEnqueueNDRangeKernel");
    } else {
        enqueueTiles(cl.queue, columnKernel, globalWorkSize[0], globalWorkSize[1], columnEvent);
    }
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}
