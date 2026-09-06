// test of save and load

// for libraries of stb_image and stb_image_write

#define STB_IMAGE_IMPLEMENTATION // generates the implementation of stb_image functions in this .cpp file
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <cstddef> // for size_t
#include <stdexcept> // standard library for exceptions

Image loadImage(const std::string& path)
{
    Image image;
    int channels = 0; // number of channels in the source image, grayscale=1, RGB=3, RGBA=4 etc.
    unsigned char* data = stbi_load(path.c_str(), &image.width, &image.height, &channels, 1);

    if (data == nullptr) {
        throw std::runtime_error("Could not load image");
    }

    // size_t is for sizes and element counts, for bigger ones
    // static cast will take a value and cast it into another type - from int to size_t
    std::size_t count  = static_cast<std::size_t>(image.width) * image.height;
    image.pixels.assign(data, data + count); // copy to pixels every value from data[0] to data[count - 1]
    stbi_image_free(data);
    return image;
}

void saveImage(const std::string& path, const Image& image)
{
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("Image is empty");
    }

    std::size_t expectedCount = static_cast<std::size_t>(image.width) * image.height;
    if (image.pixels.size() != expectedCount) {
        throw std::runtime_error("Image size mismatch");
    }

    int result = stbi_write_png(path.c_str(), image.width, image.height, 1, image.pixels.data(), image.width);

    if (result == 0) {
        throw std::runtime_error("Could not write png");
    }




}

