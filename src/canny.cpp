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

void gaussRows(const Image& image, Image& blurred, int firstRow, int endRow) {
    const int width = image.width;
    const int height = image.height;
    for (int y = firstRow; y < endRow; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * width;
        if (y == 0 || y == height - 1 || width < 3) {
            std::copy_n(&image.pixels[rowStart], width, &blurred.pixels[rowStart]);
            continue;
        }
        blurred.pixels[rowStart] = image.pixels[rowStart];
        blurred.pixels[rowStart + width - 1] = image.pixels[rowStart + width - 1];
        for (int x = 1; x < width - 1; ++x) {
            const unsigned char* p = &image.pixels[rowStart + x];
            const int sum = p[-width - 1] + 2 * p[-width] + p[-width + 1] +
                            2 * p[-1] + 4 * p[0] + 2 * p[1] +
                            p[width - 1] + 2 * p[width] + p[width + 1];
            blurred.pixels[rowStart + x] = static_cast<unsigned char>((sum + 8) / 16);
        }
    }
}

unsigned char gradientSector(int gx, int gy) {
    const int ax = std::abs(gx);
    const int ay = std::abs(gy);
    if (ay * fixedPointOne < tanOf22_5Degrees * ax) return 0;
    if (ax * fixedPointOne <= tanOf22_5Degrees * ay) return 2;
    return (gx > 0) == (gy > 0) ? 1 : 3;
}

void gradientRows(const Image& blurred, std::vector<int>& magnitude, std::vector<unsigned char>& direction,
                  int firstRow, int endRow) {
    const int width = blurred.width;
    const int height = blurred.height;
    for (int y = firstRow; y < endRow; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const std::size_t pix_idx = rowStart + x;
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                magnitude[pix_idx] = 0;
                direction[pix_idx] = 0;
                continue;
            }
            const unsigned char* p = &blurred.pixels[pix_idx];
            // intensity change in both axes
            const int gx = (p[-width + 1] + 2 * p[1] + p[width + 1]) - (p[-width - 1] + 2 * p[-1] + p[width - 1]);
            const int gy = (p[width - 1] + 2 * p[width] + p[width + 1]) - (p[-width - 1] + 2 * p[-width] + p[-width + 1]);
            magnitude[pix_idx] = gx * gx + gy * gy;
            direction[pix_idx] = gradientSector(gx, gy);
        }
    }
}

void suppressRows(const std::vector<int>& magnitude, const std::vector<unsigned char>& direction,
                  std::vector<unsigned char>& state, int width, int height, int lowSquared, int highSquared,
                  int firstRow, int endRow) {
    const std::ptrdiff_t offsets[4] = {1, width + 1, width, width - 1};
    for (int y = firstRow; y < endRow; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const std::size_t pix_idx = rowStart + x;
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                state[pix_idx] = 0;
                continue;
            }
            // non-maximum suppression - choosing locally strongest response diagonally
            const std::ptrdiff_t offset = offsets[direction[pix_idx]];
            const int candidate = magnitude[pix_idx];
            unsigned char value = 0;
            if (candidate > magnitude[pix_idx - offset] && candidate >= magnitude[pix_idx + offset]) {
                if (candidate >= highSquared) value = strongEdge;
                else if (candidate >= lowSquared) value = weakEdge;
            }
            state[pix_idx] = value;
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
            for (std::size_t i = first; i < last; ++i) {
                if (state[i] == strongEdge) stack.push_back(i);
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
    Image blurred_image = image;
    std::vector<int> magnitude(image.pixels.size());
    std::vector<unsigned char> direction(image.pixels.size());
    std::vector<unsigned char>& state = result.pixels;
    auto rows = [&](auto step) {
        parallelRanges(height, threads, [&](std::size_t begin, std::size_t end, int) {
            step(static_cast<int>(begin), static_cast<int>(end));
        });
    };
    rows([&](int first, int end) { gaussRows(image, blurred_image, first, end); });
    rows([&](int first, int end) { gradientRows(blurred_image, magnitude, direction, first, end); });
    rows([&](int first, int end) {
        suppressRows(magnitude, direction, state, width, height, lowSquared, highSquared, first, end);
    });
    trackEdges(state, width, height, threads);
    rows([&](int first, int end) {
        for (std::size_t i = static_cast<std::size_t>(first) * width; i < static_cast<std::size_t>(end) * width; ++i) {
            if (state[i] == weakEdge) state[i] = 0;
        }
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
    launch(gaussKernel, globalWorkSize, nullptr, kernelUs);
    launch(gradientKernel, globalWorkSize, nullptr, kernelUs);
    launch(suppressKernel, globalWorkSize, nullptr, kernelUs);
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
    launch(finishKernel, globalWorkSize, nullptr, kernelUs);
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer");
    if (kernelUs) addKernelTimes(kernelUs);
    output.width = width;
    output.height = height;
}
