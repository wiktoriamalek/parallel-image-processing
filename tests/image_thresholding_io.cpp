#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <cstddef>
#include <stdexcept>
#include "parallel_utils.hpp"

Image thresholding(const Image& image, int threshold)
{
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("Image is empty");
    }

    std::size_t expectedCount = static_cast<std::size_t>(image.width) * image.height;
    if (image.pixels.size() != expectedCount) {
        throw std::runtime_error("Image size mismatch");
    }

    Image result = image; // we are modifying the copy
    ThresholdRange(result, threshold, 0, result.pixels.size());
    return result;
}

void ThresholdRange(Image& image, int threshold, std::size_t min, std::size_t max)
{
    for (std::size_t i = min; i < max; ++i) {
        if (image.pixels[i] < threshold) {
            image.pixels[i] = 0;
        }
        else {
            image.pixels[i] = 255;
        }
    }
}

void ThresholdParallel(Image& image, int threshold, int numThreads) {
    parallelRanges(image.pixels.size(), numThreads, [&](std::size_t begin, std::size_t end, int) {
        ThresholdRange(image, threshold, begin, end);
    });
}
