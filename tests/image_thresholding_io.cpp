#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <cstddef>
#include <stdexcept>

Image thresholding(const Image& image)
{
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("Image is empty");
    }

    std::size_t expectedCount = static_cast<std::size_t>(image.width) * image.height;
    if (image.pixels.size() != expectedCount) {
        throw std::runtime_error("Image size mismatch");
    }

    int threshold = 128;
    Image result = image; // we are modifying the copy

    for (std::size_t i = 0; i < result.pixels.size(); ++i) {
        if (result.pixels[i] < threshold) {
            result.pixels[i] = 0;
        }
        else {
            result.pixels[i] = 255;
        }
    }
    return result;
}

