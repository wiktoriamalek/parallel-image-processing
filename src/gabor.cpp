#include "gabor.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static_assert(gaborGroupSize == tileGroupSize, "The Gabor kernels assume 16 x 16 work-groups");

namespace {
unsigned char gaborLevel(float squared) {
    if (squared >= 65025.0f) return 255;
    int level = static_cast<int>(std::sqrt(squared));
    if (static_cast<float>(level * level) > squared) --level;
    else if (static_cast<float>((level + 1) * (level + 1)) <= squared) ++level;
    return static_cast<unsigned char>(level);
}

void normalize(std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) total += std::abs(value);
    if (total < 1e-12) {
        std::fill(values.begin(), values.end(), 0.0);
        return;
    }
    for (double& value : values) value *= 2.0 / total;
}

void checkBank(const GaborBank& bank) {
    if (bank.radius < 1 || bank.radius > gaborMaxRadius || bank.orientations < 1 ||
        bank.orientations > gaborMaxOrientations ||
        bank.weights.size() != 2 * static_cast<std::size_t>(bank.taps()) * bank.orientations) {
        throw std::invalid_argument("Invalid Gabor filter bank");
    }
}

void checkImage(const Image& image) {
    if (image.width < 0 || image.height < 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height) {
        throw std::invalid_argument("Image dimensions do not match its pixels");
    }
}
}

int gaborRadius(float sigma, float gamma) {
    if (!std::isfinite(sigma) || !std::isfinite(gamma) || sigma <= 0.0f || gamma <= 0.0f) {
        throw std::invalid_argument("Gabor sigma and gamma must be positive and finite");
    }
    const double extent = 3.0 * sigma / std::min(1.0, static_cast<double>(gamma));
    return std::max(1, static_cast<int>(std::min(std::ceil(extent), 1e6)));
}

GaborBank gaborBank(float sigma, float wavelength, float gamma, int orientations) {
    const int radius = gaborRadius(sigma, gamma);
    if (!std::isfinite(wavelength) || wavelength < 2.0f) {
        throw std::invalid_argument("Gabor wavelength must be at least 2 pixels");
    }
    if (orientations < 1 || orientations > gaborMaxOrientations) {
        throw std::invalid_argument("Gabor needs 1 to " + std::to_string(gaborMaxOrientations) + " orientations");
    }
    if (radius > gaborMaxRadius) {
        throw std::invalid_argument("Gabor window radius " + std::to_string(radius) + " exceeds " +
                                    std::to_string(gaborMaxRadius) + "; lower sigma or raise gamma");
    }
    GaborBank bank;
    bank.radius = radius;
    bank.orientations = orientations;
    const int taps = bank.taps();
    bank.weights.reserve(2 * static_cast<std::size_t>(taps) * orientations);
    const double pi = std::acos(-1.0);
    const double gammaSquared = static_cast<double>(gamma) * gamma;
    const double spread = 2.0 * static_cast<double>(sigma) * sigma;
    for (int orientation = 0; orientation < orientations; ++orientation) {
        const double theta = pi * orientation / orientations;
        const double cosine = std::cos(theta);
        const double sine = std::sin(theta);
        std::vector<double> envelope(taps);
        std::vector<double> even(taps);
        std::vector<double> odd(taps);
        double evenSum = 0.0;
        double envelopeSum = 0.0;
        int tap = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx, ++tap) {
                const double along = kx * cosine + ky * sine;
                const double across = -kx * sine + ky * cosine;
                envelope[tap] = std::exp(-(along * along + gammaSquared * across * across) / spread);
                const double phase = 2.0 * pi * along / wavelength;
                even[tap] = envelope[tap] * std::cos(phase);
                odd[tap] = envelope[tap] * std::sin(phase);
                evenSum += even[tap];
                envelopeSum += envelope[tap];
            }
        }
        for (int i = 0; i < taps; ++i) even[i] -= envelope[i] * evenSum / envelopeSum;
        normalize(even);
        normalize(odd);
        for (int i = 0; i < taps; ++i) {
            bank.weights.push_back(static_cast<float>(even[i]));
            bank.weights.push_back(static_cast<float>(odd[i]));
        }
    }
    return bank;
}

std::size_t gaborTileBytes(int radius) {
    return static_cast<std::size_t>(gaborGroupSize * gaborPixelsPerItem + 2 * radius) * (gaborGroupSize + 2 * radius);
}

Image gaborFilter(const Image& image, const GaborBank& bank) {
    Image result;
    gaborFilter(image, bank, result);
    return result;
}

void gaborFilter(const Image& image, const GaborBank& bank, Image& result, int threads) {
    checkBank(bank);
    checkImage(image);
    if (&image == &result) throw std::invalid_argument("Gabor filtering cannot run in place");
    result.width = image.width;
    result.height = image.height;
    result.pixels.resize(image.pixels.size());
    parallelRanges(static_cast<std::size_t>(image.height), threads, [&](std::size_t begin, std::size_t end, int) {
        gaborRows(image, bank, result, static_cast<int>(begin), static_cast<int>(end));
    });
}

void gaborRows(const Image& image, const GaborBank& bank, Image& result, int firstRow, int endRow) {
    if (firstRow >= endRow || image.width == 0) return;
    const int width = image.width;
    const int height = image.height;
    const int radius = bank.radius;
    const int size = bank.size();
    const int padded = width + 2 * radius;
    const unsigned char* pixels = image.pixels.data();
    unsigned char* output = result.pixels.data();
    std::vector<float> rows(static_cast<std::size_t>(size) * padded);
    const int block = std::min(width, gaborColumnBlock);
    std::vector<float> evenSums(block);
    std::vector<float> oddSums(block);
    std::vector<float> strongest(block);
    float* even = evenSums.data();
    float* odd = oddSums.data();
    float* best = strongest.data();

    auto load = [&](int row) {
        const unsigned char* line = pixels + static_cast<std::size_t>(std::clamp(row, 0, height - 1)) * width;
        float* target = rows.data() + static_cast<std::size_t>((row + radius) % size) * padded;
        for (int x = 0; x < padded; ++x) target[x] = line[std::clamp(x - radius, 0, width - 1)];
    };

    for (int row = firstRow - radius; row < firstRow + radius; ++row) load(row);
    for (int y = firstRow; y < endRow; ++y) {
        load(y + radius);
        for (int firstX = 0; firstX < width; firstX += block) {
            const int count = std::min(block, width - firstX);
            const float* weight = bank.weights.data();
            for (int orientation = 0; orientation < bank.orientations; ++orientation) {
                std::fill(even, even + count, 0.0f);
                std::fill(odd, odd + count, 0.0f);
                for (int ky = 0; ky < size; ++ky) {
                    const float* line = rows.data() + static_cast<std::size_t>((y + ky) % size) * padded + firstX;
                    for (int kx = 0; kx < size; ++kx) {
                        const float evenWeight = weight[0];
                        const float oddWeight = weight[1];
                        weight += 2;
                        const float* source = line + kx;
                        for (int x = 0; x < count; ++x) {
                            even[x] += source[x] * evenWeight;
                            odd[x] += source[x] * oddWeight;
                        }
                    }
                }
                for (int x = 0; x < count; ++x) {
                    float magnitude = even[x] * even[x];
                    magnitude += odd[x] * odd[x];
                    best[x] = orientation == 0 ? magnitude : std::max(best[x], magnitude);
                }
            }
            unsigned char* target = output + static_cast<std::size_t>(y) * width + firstX;
            for (int x = 0; x < count; ++x) target[x] = gaborLevel(best[x]);
        }
    }
}

GaborGPU::GaborGPU(const Image& image, const GaborBank& bank)
    : cl("kernels/gabor.cl"),
      bufferSize(image.pixels.size()),
      globalWorkSize{static_cast<std::size_t>(image.width + gaborPixelsPerItem - 1) / gaborPixelsPerItem,
                     static_cast<std::size_t>(image.height)} {
    checkBank(bank);
    checkImage(image);
    if (image.pixels.empty()) throw std::invalid_argument("The GPU Gabor filter needs a nonempty image");
    try {
        cl_ulong localMemory = 0;
        checkCL(clGetDeviceInfo(cl.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemory), &localMemory, nullptr), "clGetDeviceInfo");
        const std::size_t tileBytes = gaborTileBytes(bank.radius);
        if (tileBytes > localMemory) throw std::runtime_error("Insufficient OpenCL local memory for the Gabor window");
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        weightsBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bank.weights.size() * sizeof(float),
                                       const_cast<float*>(bank.weights.data()), &error);
        checkCL(error, "clCreateBuffer");
        kernel = clCreateKernel(cl.program, "gabor", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(inputBuffer), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(outputBuffer), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(image.width), &image.width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(image.height), &image.height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(weightsBuffer), &weightsBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 5, sizeof(bank.radius), &bank.radius), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 6, sizeof(bank.orientations), &bank.orientations), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 7, tileBytes, nullptr), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

void GaborGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
    if (weightsBuffer) clReleaseMemObject(weightsBuffer);
}

GaborGPU::~GaborGPU() {
    release();
}

void GaborGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (static_cast<std::size_t>(image.width + gaborPixelsPerItem - 1) / gaborPixelsPerItem != globalWorkSize[0] ||
        static_cast<std::size_t>(image.height) != globalWorkSize[1] || image.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    output.width = image.width;
    output.height = image.height;
    output.pixels.resize(bufferSize);
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    enqueueTiles(cl.queue, kernel, globalWorkSize[0], globalWorkSize[1], kernelEvent);
    checkCL(clFinish(cl.queue), "clFinish");
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
