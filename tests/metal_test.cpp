#include "metal_algorithms.hpp"
#include "metal_utils.hpp"
#include "metal_cases.hpp"
#include <iostream>
#include <random>
#include <limits>

int main() {
    try {
        std::cout << "Metal GPU: " << metalDeviceName() << std::endl;
        std::mt19937 random(20261120);
        std::size_t checked = 0;
        for (auto [width, height] : {std::pair{1, 1}, {3, 5}, {65, 67}, {257, 129}}) {
            Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height)};
            for (auto& pixel : image.pixels) pixel = static_cast<unsigned char>(random());
            for (MetalCase test : metalCases(true)) {
                test.parameters.seedX = width / 2;
                test.parameters.seedY = height / 2;
                if (test.algorithm == Algorithm::MultiOtsu) {
                    const auto histogram = calculateHistogram(image);
                    const auto distinct = std::count_if(histogram.begin(), histogram.end(), [](auto count) { return count != 0; });
                    if (distinct < test.parameters.classes) continue;
                }
                std::cout << width << 'x' << height << ' ' << test.name << std::endl;
                const Image reference = metalCPUReference(test, image);
                MetalProcessor gpu(test.algorithm, image, test.parameters);
                Image output;
                gpu.run(image, output);
                const ImageDifference difference = metalDifference(reference, output);
                const bool floating = test.algorithm == Algorithm::Contrast || test.algorithm == Algorithm::Convolution ||
                                      test.algorithm == Algorithm::Resize || test.algorithm == Algorithm::Distortion;
                const bool histogram = test.algorithm == Algorithm::Otsu || test.algorithm == Algorithm::MultiOtsu;
                if (isLocalThreshold(test)) checkMetalLocalThreshold(test, image, reference, output, "Metal");
                else if (difference.maximum > (floating ? 1 : 0) && !histogram) {
                    throw std::runtime_error(test.name + " differs from CPU on " + std::to_string(difference.pixels) +
                                             " pixels, max difference " + std::to_string(difference.maximum));
                }
                if (histogram) checkMetalSegmentation(test, image, output, gpu.thresholds());
                const double gpuTime = gpu.kernelTimeUs();
                if (gpuTime < 0) throw std::runtime_error("GPU timestamp profiling is unavailable");
                if (gpu.kernelTimeUs() != gpuTime) throw std::runtime_error("Reading GPU timestamps changed the reported time");
                Image second = image;
                for (auto& pixel : second.pixels) pixel = static_cast<unsigned char>(random());
                if (test.algorithm == Algorithm::MultiOtsu) {
                    const auto histogram = calculateHistogram(second);
                    const auto distinct = std::count_if(histogram.begin(), histogram.end(), [](auto count) { return count != 0; });
                    if (distinct < test.parameters.classes) second = image;
                }
                gpu.run(second, output);
                const Image secondReference = metalCPUReference(test, second);
                const ImageDifference nextDifference = metalDifference(secondReference, output);
                if (isLocalThreshold(test)) checkMetalLocalThreshold(test, second, secondReference, output, "Metal");
                else if (nextDifference.maximum > (floating ? 1 : 0) && !histogram) {
                    throw std::runtime_error(test.name + " failed when reusing Metal buffers with a changed image");
                }
                if (histogram) checkMetalSegmentation(test, second, output, gpu.thresholds());
                checked += 2;
            }
        }
        Image connected{131, 130, std::vector<unsigned char>(131 * 130, 100)};
        for (bool tiles : {false, true}) {
            Parameters p;
            p.seedX = 0;
            p.seedY = connected.height - 1;
            p.tolerance = 0;
            p.localTiles = tiles;
            MetalCase test{"connected_region", Algorithm::RegionGrowing, p};
            MetalProcessor gpu(test.algorithm, connected, p);
            Image output;
            gpu.run(connected, output);
            if (metalDifference(metalCPUReference(test, connected), output).pixels != 0) {
                throw std::runtime_error("Region propagation failed across tile boundaries");
            }
            for (int y = 0; y < connected.height; ++y) {
                if (y != connected.height / 2) connected.pixels[static_cast<std::size_t>(y) * connected.width + 64] = 255;
            }
            gpu.run(connected, output);
            if (metalDifference(metalCPUReference(test, connected), output).pixels != 0) {
                throw std::runtime_error("Region propagation failed through a narrow passage");
            }
            std::fill(connected.pixels.begin(), connected.pixels.end(), 100);
            checked += 2;
        }
        Image edges{131, 130, std::vector<unsigned char>(131 * 130)};
        for (int y = 0; y < edges.height; ++y) {
            for (int x = 0; x < edges.width; ++x) {
                edges.pixels[static_cast<std::size_t>(y) * edges.width + x] =
                    static_cast<unsigned char>((x > y / 2 ? 160 : 20) + (y % 17));
            }
        }
        for (auto [low, high] : {std::pair{0.0f, 0.0f}, {20.0f, 80.0f}, {100.0f, 200.0f}}) {
            Parameters p;
            p.cannyLow = low;
            p.cannyHigh = high;
            MetalCase test{"connected_canny", Algorithm::Canny, p};
            MetalProcessor gpu(test.algorithm, edges, p);
            Image output;
            gpu.run(edges, output);
            if (metalDifference(metalCPUReference(test, edges), output).pixels != 0) {
                throw std::runtime_error("Canny tracking failed across tile boundaries");
            }
            ++checked;
        }
        for (int strip : {1, 3, 16, 32}) {
            for (int radius : {1, 7}) {
                Parameters p;
                p.medianRadius = radius;
                p.medianSort = static_cast<int>(MedianSort::Shared);
                p.medianStrip = strip;
                MetalCase test{"median_strip", Algorithm::MedianFilter, p};
                MetalProcessor gpu(test.algorithm, edges, p);
                Image output;
                gpu.run(edges, output);
                if (metalDifference(metalCPUReference(test, edges), output).pixels != 0) {
                    throw std::runtime_error("Shared median failed for strip " + std::to_string(strip) + " and radius " + std::to_string(radius));
                }
                ++checked;
            }
        }
        for (float scale : {0.6f, 2.0f, 66.5f / 131.0f, 100.5f / 131.0f}) {
            Parameters p;
            p.scale = scale;
            MetalCase test{"inplace_resize", Algorithm::Resize, p};
            MetalProcessor gpu(test.algorithm, edges, p);
            Image inplace = edges;
            gpu.run(inplace, inplace);
            if (metalDifference(metalCPUReference(test, edges), inplace).maximum > 1) {
                throw std::runtime_error("In-place resize failed");
            }
            ++checked;
        }
        auto expectInvalid = [](auto action) {
            try {
                action();
            } catch (const std::invalid_argument&) {
                return;
            }
            throw std::runtime_error("Invalid Metal input was accepted");
        };
        expectInvalid([] { MetalProcessor gpu(Algorithm::Negative, Image{}, Parameters{}); });
        Parameters invalid;
        invalid.radius = 128;
        expectInvalid([&] { MetalProcessor gpu(Algorithm::Niblack, edges, invalid); });
        invalid = Parameters{};
        invalid.medianRadius = 0;
        expectInvalid([&] { MetalProcessor gpu(Algorithm::MedianFilter, edges, invalid); });
        invalid = Parameters{};
        invalid.maxClasses = 1;
        expectInvalid([&] { MetalProcessor gpu(Algorithm::PeaksValleys, edges, invalid); });
        invalid = Parameters{};
        invalid.scale = 0;
        expectInvalid([&] { MetalProcessor gpu(Algorithm::Resize, edges, invalid); });
        invalid = Parameters{};
        invalid.cannyLow = std::numeric_limits<float>::quiet_NaN();
        expectInvalid([&] { MetalProcessor gpu(Algorithm::Canny, edges, invalid); });
        Parameters multiParameters;
        MetalProcessor multi(Algorithm::MultiOtsu, connected, multiParameters);
        Image multiOutput;
        expectInvalid([&] { multi.run(connected, multiOutput); });
        multi.run(edges, multiOutput);
        checkMetalSegmentation({"multi_retry", Algorithm::MultiOtsu, multiParameters}, edges, multiOutput, multi.thresholds());
        ++checked;
        std::cout << "Verified " << checked << " Metal executions" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
