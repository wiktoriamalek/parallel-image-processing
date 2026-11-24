#include "canny.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr unsigned char weakEdge = 128;
constexpr unsigned char strongEdge = 255;
constexpr int tanOf22_5Degrees = 27146;
constexpr int fixedPointOne = 65536;
constexpr int largestSquaredMagnitude = 2 * 1020 * 1020;
constexpr std::size_t trackTile = 64;
constexpr std::size_t trackGroup = 16;

void checkThresholds(float lowThreshold, float highThreshold) {
    if (!(lowThreshold >= 0.0f) || !(highThreshold >= lowThreshold)) {
        throw std::invalid_argument("Canny thresholds must satisfy 0 <= low <= high");
    }
}

int squaredThreshold(float threshold) {
    const double squared = std::ceil(static_cast<double>(threshold) * threshold);
    return static_cast<int>(std::min(squared, static_cast<double>(largestSquaredMagnitude) + 1.0));
}

void gaussRow(const Image& image, int y, unsigned char* blurred) {
    const int width = image.width;
    const unsigned char* row = image.pixels.data() + static_cast<std::size_t>(y) * width;
    if (y == 0 || y == image.height - 1 || width < 3) {
        std::copy_n(row, width, blurred);
        return;
    }
    const unsigned char* up = row - width;
    const unsigned char* down = row + width;
    blurred[0] = row[0];
    blurred[width - 1] = row[width - 1];
    for (int x = 1; x < width - 1; ++x) {
        const int sum = up[x - 1] + 2 * up[x] + up[x + 1] + 2 * row[x - 1] + 4 * row[x] + 2 * row[x + 1] +
                        down[x - 1] + 2 * down[x] + down[x + 1];
        blurred[x] = static_cast<unsigned char>((sum + 8) >> 4);
    }
}

void gaussRows(const Image& image, Image& blurred, int firstRow, int endRow) {
    for (int y = firstRow; y < endRow; ++y) gaussRow(image, y, blurred.pixels.data() + static_cast<std::size_t>(y) * image.width);
}

void gradientRow(const unsigned char* up, const unsigned char* row, const unsigned char* down, int width, int* magnitude,
                 unsigned char* direction) {
    magnitude[0] = 0;
    direction[0] = 0;
    magnitude[width - 1] = 0;
    direction[width - 1] = 0;
    for (int x = 1; x < width - 1; ++x) {
        const int gx = (up[x + 1] + 2 * row[x + 1] + down[x + 1]) - (up[x - 1] + 2 * row[x - 1] + down[x - 1]);
        const int gy = (down[x - 1] + 2 * down[x] + down[x + 1]) - (up[x - 1] + 2 * up[x] + up[x + 1]);
        const int ax = std::abs(gx);
        const int ay = std::abs(gy);
        const unsigned char diagonal = (gx > 0) == (gy > 0) ? 1 : 3;
        const unsigned char steep = ax * fixedPointOne <= tanOf22_5Degrees * ay ? 2 : diagonal;
        magnitude[x] = gx * gx + gy * gy;
        direction[x] = ay * fixedPointOne < tanOf22_5Degrees * ax ? 0 : steep;
    }
}

void suppressRow(const int* up, const int* row, const int* down, const unsigned char* direction, int width, int lowSquared,
                 int highSquared, unsigned char* state) {
    state[0] = 0;
    state[width - 1] = 0;
    for (int x = 1; x < width - 1; ++x) {
        const int candidate = row[x];
        if (candidate < lowSquared) {
            state[x] = 0;
            continue;
        }
        const unsigned char sector = direction[x];
        const int before = sector == 0 ? row[x - 1] : sector == 1 ? up[x - 1] : sector == 2 ? up[x] : up[x + 1];
        const int after = sector == 0 ? row[x + 1] : sector == 1 ? down[x + 1] : sector == 2 ? down[x] : down[x - 1];
        const unsigned char edge = candidate >= highSquared ? strongEdge : weakEdge;
        state[x] = candidate > before && candidate >= after ? edge : 0;
    }
}

void cannyBand(const Image& image, std::vector<unsigned char>& state, int lowSquared, int highSquared, int firstRow, int endRow) {
    const int width = image.width;
    const int height = image.height;
    for (int y = firstRow; y < endRow; ++y) {
        if (y == 0 || y == height - 1) std::fill_n(state.data() + static_cast<std::size_t>(y) * width, width, 0);
    }
    const int first = std::max(firstRow, 1);
    const int last = std::min(endRow, height - 1);
    if (first >= last) return;
    std::vector<unsigned char> blurred(3 * static_cast<std::size_t>(width));
    std::vector<int> magnitude(3 * static_cast<std::size_t>(width));
    std::vector<unsigned char> direction(3 * static_cast<std::size_t>(width));
    auto slot = [width](int row) { return static_cast<std::size_t>(row % 3) * width; };
    int nextBlurred = std::max(first - 2, 0);
    for (int row = first - 1; row <= last; ++row) {
        if (row == 0 || row == height - 1) {
            std::fill_n(magnitude.data() + slot(row), width, 0);
            std::fill_n(direction.data() + slot(row), width, 0);
        } else {
            for (; nextBlurred <= row + 1; ++nextBlurred) gaussRow(image, nextBlurred, blurred.data() + slot(nextBlurred));
            gradientRow(blurred.data() + slot(row - 1), blurred.data() + slot(row), blurred.data() + slot(row + 1), width,
                        magnitude.data() + slot(row), direction.data() + slot(row));
        }
        const int y = row - 1;
        if (y >= first && y < last) {
            suppressRow(magnitude.data() + slot(y - 1), magnitude.data() + slot(y), magnitude.data() + slot(y + 1),
                        direction.data() + slot(y), width, lowSquared, highSquared, state.data() + static_cast<std::size_t>(y) * width);
        }
    }
}

void trackRows(std::vector<unsigned char>& state, int width, int firstRow, int endRow, std::vector<std::size_t>& stack) {
    while (!stack.empty()) {
        const std::size_t frontIndex = stack.back();
        stack.pop_back();
        const int xCoord = static_cast<int>(frontIndex % width);
        const int yCoord = static_cast<int>(frontIndex / width);
        for (int dy = -1; dy <= 1; dy++) {
            const int neighbourYCoord = yCoord + dy;
            if (neighbourYCoord < firstRow || neighbourYCoord >= endRow) continue;
            for (int dx = -1; dx <= 1; dx++) {
                const int neighbourXCoord = xCoord + dx;
                if ((dx == 0 && dy == 0) || neighbourXCoord < 0 || neighbourXCoord >= width) continue;
                const std::size_t neighbourIndex = static_cast<std::size_t>(neighbourYCoord) * width + neighbourXCoord;
                if (state[neighbourIndex] == weakEdge) {
                    state[neighbourIndex] = strongEdge;
                    stack.push_back(neighbourIndex);
                }
            }
        }
    }
}

void trackEdges(std::vector<unsigned char>& state, int width, int height, int threads) {
    const int bands = std::max(1, std::min(threads, height));
    auto bandStart = [&](std::size_t band) {
        return static_cast<int>(static_cast<long long>(height) * static_cast<long long>(band) / bands);
    };
    parallelRanges(bands, threads, [&](std::size_t begin, std::size_t end, int) {
        std::vector<std::size_t> stack;
        for (std::size_t band = begin; band < end; ++band) {
            const std::size_t first = static_cast<std::size_t>(bandStart(band)) * width;
            const std::size_t last = static_cast<std::size_t>(bandStart(band + 1)) * width;
            const auto begin = state.begin() + static_cast<std::ptrdiff_t>(first);
            const auto end = state.begin() + static_cast<std::ptrdiff_t>(last);
            for (auto edge = std::find(begin, end, strongEdge); edge != end; edge = std::find(edge + 1, end, strongEdge)) {
                stack.push_back(static_cast<std::size_t>(edge - state.begin()));
            }
            trackRows(state, width, bandStart(band), bandStart(band + 1), stack);
        }
    });
    if (bands == 1) return;

    std::vector<unsigned char> boundary(static_cast<std::size_t>(bands) * 2 * width);
    for (;;) {
        for (int band = 0; band < bands; ++band) {
            const std::size_t first = static_cast<std::size_t>(bandStart(band)) * width;
            const std::size_t last = static_cast<std::size_t>(bandStart(band + 1) - 1) * width;
            std::copy_n(&state[first], width, &boundary[static_cast<std::size_t>(2 * band) * width]);
            std::copy_n(&state[last], width, &boundary[static_cast<std::size_t>(2 * band + 1) * width]);
        }
        std::vector<char> changed(bands, 0);
        parallelRanges(bands, threads, [&](std::size_t begin, std::size_t end, int) {
            std::vector<std::size_t> stack;
            for (std::size_t band = begin; band < end; ++band) {
                auto seedFrom = [&](int row, const unsigned char* neighbourRow) {
                    for (int x = 0; x < width; ++x) {
                        const std::size_t index = static_cast<std::size_t>(row) * width + x;
                        if (state[index] != weakEdge) continue;
                        for (int nx = std::max(0, x - 1); nx <= std::min(width - 1, x + 1); ++nx) {
                            if (neighbourRow[nx] == strongEdge) {
                                state[index] = strongEdge;
                                stack.push_back(index);
                                changed[band] = 1;
                                break;
                            }
                        }
                    }
                };
                if (band > 0) seedFrom(bandStart(band), &boundary[(2 * band - 1) * width]);
                if (band + 1 < static_cast<std::size_t>(bands)) seedFrom(bandStart(band + 1) - 1, &boundary[(2 * band + 2) * width]);
                trackRows(state, width, bandStart(band), bandStart(band + 1), stack);
            }
        });
        if (std::none_of(changed.begin(), changed.end(), [](char value) { return value != 0; })) break;
    }
}
}

Image gauss_function(const Image& image)
{
    Image blurred = image;
    gaussRows(image, blurred, 0, image.height);
    return blurred;
}

Image canny(const Image& image, float lowThreshold, float highThreshold)
{
    Image result = image;
    canny(image, result, lowThreshold, highThreshold, 1);
    return result;
}

void canny(const Image& image, Image& result, float lowThreshold, float highThreshold, int threads)
{
    checkThresholds(lowThreshold, highThreshold);
    if (threads <= 0) throw std::invalid_argument("Thread count must be positive");
    if (result.pixels.size() != image.pixels.size()) throw std::runtime_error("Output image size does not match the input");
    result.width = image.width;
    result.height = image.height;
    const int width = image.width;
    const int height = image.height;
    const int lowSquared = squaredThreshold(lowThreshold);
    const int highSquared = squaredThreshold(highThreshold);
    std::vector<unsigned char>& state = result.pixels;
    auto rows = [&](auto step) {
        parallelRanges(height, threads, [&](std::size_t begin, std::size_t end, int) {
            step(static_cast<int>(begin), static_cast<int>(end));
        });
    };
    rows([&](int first, int end) { cannyBand(image, state, lowSquared, highSquared, first, end); });
    trackEdges(state, width, height, threads);
    rows([&](int first, int end) {
        unsigned char* pixels = state.data();
        const std::size_t stop = static_cast<std::size_t>(end) * width;
        for (std::size_t i = static_cast<std::size_t>(first) * width; i < stop; ++i) pixels[i] = pixels[i] == weakEdge ? 0 : pixels[i];
    });
}

CannyGPU::CannyGPU(const Image& image, float lowThreshold, float highThreshold)
    : cl("kernels/canny.cl"), bufferSize(image.pixels.size()), width(image.width), height(image.height),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    checkThresholds(lowThreshold, highThreshold);
    tilesX = static_cast<int>((static_cast<std::size_t>(width) + trackTile - 1) / trackTile);
    tilesY = static_cast<int>((static_cast<std::size_t>(height) + trackTile - 1) / trackTile);
    tileGlobalWorkSize[0] = static_cast<std::size_t>(tilesX) * trackGroup;
    tileGlobalWorkSize[1] = static_cast<std::size_t>(tilesY) * trackGroup;
    const int lowSquared = squaredThreshold(lowThreshold);
    const int highSquared = squaredThreshold(highThreshold);
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        blurredBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        magnitudeBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize * sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        sectorBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        stateBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        changedBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");
        const std::size_t activeBytes = static_cast<std::size_t>(tilesX) * tilesY * sizeof(cl_int);
        activeBufferA = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, activeBytes, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        activeBufferB = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, activeBytes, nullptr, &error);
        checkCL(error, "clCreateBuffer");

        gaussKernel = clCreateKernel(cl.program, "canny_gauss", &error);
        checkCL(error, "clCreateKernel");
        gradientKernel = clCreateKernel(cl.program, "canny_gradient", &error);
        checkCL(error, "clCreateKernel");
        suppressKernel = clCreateKernel(cl.program, "canny_suppress", &error);
        checkCL(error, "clCreateKernel");
        trackKernel = clCreateKernel(cl.program, "canny_track", &error);
        checkCL(error, "clCreateKernel");
        finishKernel = clCreateKernel(cl.program, "canny_finish", &error);
        checkCL(error, "clCreateKernel");

        checkCL(clSetKernelArg(gaussKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(gaussKernel, 1, sizeof(cl_mem), &blurredBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(gaussKernel, 2, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(gaussKernel, 3, sizeof(int), &height), "clSetKernelArg");

        checkCL(clSetKernelArg(gradientKernel, 0, sizeof(cl_mem), &blurredBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(gradientKernel, 1, sizeof(cl_mem), &magnitudeBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(gradientKernel, 2, sizeof(cl_mem), &sectorBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(gradientKernel, 3, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(gradientKernel, 4, sizeof(int), &height), "clSetKernelArg");

        checkCL(clSetKernelArg(suppressKernel, 0, sizeof(cl_mem), &magnitudeBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 1, sizeof(cl_mem), &sectorBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 2, sizeof(cl_mem), &stateBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 3, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 4, sizeof(int), &height), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 5, sizeof(int), &lowSquared), "clSetKernelArg");
        checkCL(clSetKernelArg(suppressKernel, 6, sizeof(int), &highSquared), "clSetKernelArg");

        checkCL(clSetKernelArg(trackKernel, 0, sizeof(cl_mem), &stateBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 1, sizeof(cl_mem), &changedBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 4, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 5, sizeof(int), &height), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 6, sizeof(int), &tilesX), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 7, sizeof(int), &tilesY), "clSetKernelArg");

        checkCL(clSetKernelArg(finishKernel, 0, sizeof(cl_mem), &stateBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(finishKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(finishKernel, 2, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(finishKernel, 3, sizeof(int), &height), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void CannyGPU::release() {
    for (cl_event event : kernelEvents) clReleaseEvent(event);
    kernelEvents.clear();
    for (cl_kernel kernel : {gaussKernel, gradientKernel, suppressKernel, trackKernel, finishKernel}) {
        if (kernel) clReleaseKernel(kernel);
    }
    for (cl_mem buffer : {inputBuffer, blurredBuffer, magnitudeBuffer, sectorBuffer, stateBuffer, outputBuffer, changedBuffer,
                          activeBufferA, activeBufferB}) {
        if (buffer) clReleaseMemObject(buffer);
    }
}

CannyGPU::~CannyGPU() {
    release();
}

void CannyGPU::launch(cl_kernel kernel, const std::size_t* global, const std::size_t* local, double* kernelUs) {
    cl_event event = nullptr;
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, global, local, 0, nullptr, kernelUs ? &event : nullptr),
            "clEnqueueNDRangeKernel");
    if (kernelUs) kernelEvents.push_back(event);
}

void CannyGPU::addKernelTimes(double* kernelUs) {
    for (cl_event event : kernelEvents) {
        *kernelUs += kernelTimeUs(event);
        checkCL(clReleaseEvent(event), "clReleaseEvent");
    }
    kernelEvents.clear();
}

void CannyGPU::run(const Image& image, Image& output, double* kernelUs) {
    if (image.width != width || image.height != height || image.pixels.size() != bufferSize ||
        output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    if (kernelUs) *kernelUs = 0.0;
    const std::size_t tileLocalWorkSize[2] = {trackGroup, trackGroup};
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_FALSE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer");
    const std::size_t pixelGlobal[2] = {roundUp(globalWorkSize[0], tileGroupSize), roundUp(globalWorkSize[1], tileGroupSize)};
    const std::size_t pixelLocal[2] = {tileGroupSize, tileGroupSize};
    launch(gaussKernel, pixelGlobal, pixelLocal, kernelUs);
    launch(gradientKernel, pixelGlobal, pixelLocal, kernelUs);
    launch(suppressKernel, pixelGlobal, pixelLocal, kernelUs);
    const cl_int one = 1;
    checkCL(clEnqueueFillBuffer(cl.queue, activeBufferA, &one, sizeof(one), 0,
                                static_cast<std::size_t>(tilesX) * tilesY * sizeof(cl_int), 0, nullptr, nullptr),
            "clEnqueueFillBuffer");
    cl_mem previousActive = activeBufferA;
    cl_mem nextActive = activeBufferB;
    iterationCount = 0;
    for (cl_int changed = 1; changed != 0;) {
        changed = 0;
        checkCL(clEnqueueWriteBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr),
                "clEnqueueWriteBuffer");
        checkCL(clSetKernelArg(trackKernel, 2, sizeof(cl_mem), &previousActive), "clSetKernelArg");
        checkCL(clSetKernelArg(trackKernel, 3, sizeof(cl_mem), &nextActive), "clSetKernelArg");
        launch(trackKernel, tileGlobalWorkSize, tileLocalWorkSize, kernelUs);
        std::swap(previousActive, nextActive);
        checkCL(clEnqueueReadBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr),
                "clEnqueueReadBuffer");
        ++iterationCount;
        if (kernelUs) addKernelTimes(kernelUs);
    }
    launch(finishKernel, pixelGlobal, pixelLocal, kernelUs);
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer");
    if (kernelUs) addKernelTimes(kernelUs);
    output.width = width;
    output.height = height;
}
