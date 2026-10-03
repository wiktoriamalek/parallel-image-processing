#include "image_io.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        Image image = loadImage("images/clic_image.png");
        Image single = thresholding(image);
        Image divided = image;
        ThresholdParallel(divided, 128, 2);

        if (single.pixels == divided.pixels) {
            std::cout << " Image are identical!" << std::endl;
        }

        std::cout << "Single pixels: " <<single.pixels.size() << std::endl;
        std::cout << "Divided: " << divided.pixels.size() << std::endl;
        std::cout << "Width:" << image.width << " " << " Height:" <<image.height;
        if (!image.pixels.empty()) {
            std::cout << " First pixel:" << static_cast<int>(image.pixels.front()) << " ";
        };
        saveImage("results/clic_threshold.png", single);

        Image copy = loadImage("results/clic_copy.png");



        if (image.width != copy.width || image.height != copy.height || image.pixels != copy.pixels) {
            throw std::runtime_error("Image are different");
        }
        else {
            std::cout << " Images are identical!" << std::endl;
        }

        std::cout << "Image.pixels[0]: " << static_cast<int>(image.pixels[0]) << " " <<  "result.pixels[0]: " << static_cast<int>(single.pixels[0]) << " ";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error when writing to image: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}





















