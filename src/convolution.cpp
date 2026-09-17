#include "image_io.hpp"
#include <array>
#include <algorithm>
#include <cmath>

Image convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue = false) {
    Image result = image;
    convolution(image, kernel, absoluteValue, result);
    return result;
}

void convolution(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result) {
    convolutionRows(image, kernel, absoluteValue, result, 1, image.height - 1);
}

void convolutionRows(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, bool absoluteValue, Image& result, int firstRow, int endRow) {

    // dont start from edges
    // y = 1...height-2
    // x = 1 ... width -2

    for (int y=firstRow; y<endRow; y++) {
        for (int x=1; x<image.width-1; x++) {
            float sum = 0.0f;
            // even though images are 2D, pixels are stored as one big 1D array
            // 0 1 2 3 4
            // 5 6 7 8 9 1
            // 0 11 12 13 14 - pixels, 3 rows 5 numbers each
            // index = y * width + x
            // index = 1 * 5 + 2 = 7, so pixel(2, 1) = 7
            // kx = -1, ky = 0 => 1 pixel left, the same row
            for (int ky=-1; ky<=1; ky++) {
                for (int kx=-1; kx<=1; kx++) {
                    int neighourX = x + kx;
                    int neighourY = y + ky;
                    std::size_t neighbourIndex = neighourY * image.width + neighourX;
                    // neighbour value * kernel weight
                    float neighbourValue = image.pixels[neighbourIndex];
                    sum += neighbourValue * kernel[ky+1][kx+1];
                }
            }
            std::size_t resultIndex = y * image.width + x;
            if (absoluteValue) {
                sum = std::abs(sum);
            }

            sum = std::clamp(sum, 0.0f, 255.0f);
            result.pixels[resultIndex] = static_cast<unsigned char>(sum);
        }
    }

}
