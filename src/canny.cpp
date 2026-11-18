#include "canny.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "convolution.hpp"
#include <numbers>

Image gauss_function(const Image& image)
{
    const ConvolutionFilter& selected = convolutionFilter("gaussian_blur");
    return convolution(image, selected.kernel, selected.absoluteValue);
}

float sobel_function(const Image& image, const std::array<std::array<float, 3>, 3>& kernel, int x, int y) {
    float sum = 0.0f;
    for (int ky=-1; ky<=1; ky++) {
        for (int kx=-1; kx<=1; kx++) {
            int neighourX = x + kx;
            int neighourY = y + ky;
            std::size_t neighbourIndex = neighourY * image.width + neighourX;
            float neighbourValue = image.pixels[neighbourIndex];
            sum += neighbourValue * kernel[ky+1][kx+1];
        }
    }
    return sum;

}

void canny(const Image& image)
{
    Image blurred_image = gauss_function(image);
    const ConvolutionFilter& sobel_v = convolutionFilter("sobel_v");
    const ConvolutionFilter& sobel_h = convolutionFilter("sobel_h");
    std::vector<float> magnitude (blurred_image.pixels.size(),0.0f);
    std::vector<float> direction (blurred_image.pixels.size(),0.0f);

    for (int y=1; y<image.height-1; y++)
    {
        for (int x=1; x<image.width-1; x++)
        {
            // intensity change in both axes
            float gx = sobel_function(blurred_image, sobel_v.kernel, x, y);
            float gy = sobel_function(blurred_image, sobel_h.kernel, x, y);
            float grad_module = sqrt(gx*gx + gy*gy);
            std::size_t pix_idx = y*image.width + x;
            magnitude[pix_idx]=grad_module;
            float radians = std::atan2(gy, gx); // rad
            float degrees = radians * 180.0f/M_PI;
            if (degrees < 0.0f) degrees += 180.0f;
            if (degrees >= 180.0f) degrees -= 180.0f; // so the range is 0-180 degrees
            direction[pix_idx]=degrees;
        }
    }
}


