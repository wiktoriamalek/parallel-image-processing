#include "canny.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "convolution.hpp"
#include <numbers>
#include <queue>

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

void canny(const Image& image, float lowThreshold, float highThreshold)
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

    // non-maximum suppression - choosing locally strongest response diagonally
    std::vector<float> suppressed (blurred_image.pixels.size(),0.0f);
    for (int y=1; y<image.height-1; y++)
    {
        for (int x=1; x<image.width-1; x++)
        {
            std::size_t pix_idx = y*image.width + x;
            float current_degree = direction[pix_idx];
            float mag1 = 0.0f;
            float mag2 = 0.0f;

            if (current_degree < 22.5f || current_degree>=157.5f)
            {
                mag1 = magnitude[pix_idx - 1];
                mag2 = magnitude[pix_idx + 1];
            }
            else if (current_degree >= 22.5f && current_degree<67.5f) // top left and bottom right
            {
                mag1 = magnitude[pix_idx - image.width-1];
                mag2 = magnitude[pix_idx + image.width+1];
            }

            else if (current_degree >= 67.5f && current_degree<112.5f) // top and bottom
            {
                mag1 = magnitude[pix_idx - image.width];
                mag2 = magnitude[pix_idx + image.width];
            }

            else if (current_degree >= 112.5f && current_degree<157.5f) // top right and bottom left
            {
                mag1 = magnitude[pix_idx - image.width+1];
                mag2 = magnitude[pix_idx + image.width-1];
            }

            if (magnitude[pix_idx] >= mag1 && magnitude[pix_idx] >= mag2)
            {
                suppressed[pix_idx] = magnitude[pix_idx];
            }
        }
    }
    std::vector<unsigned char> state (blurred_image.pixels.size(),0.0f);
    for (std::size_t i=0; i<suppressed.size(); i++)
    {
        float candidate = suppressed[i];
        if (candidate >= highThreshold) {candidate = 255;}
        else if (candidate >= lowThreshold) {candidate = 128;}
        else if (candidate <lowThreshold) {candidate = 0;}
        state[i] = candidate;
    }

    std::queue<std::size_t> queue;
    for (std::size_t i=0; i<state.size(); i++)
    {
        if (state[i] == 255)
        {
            queue.push(i);
        }
    }
    while (!queue.empty())
    {
        int frontIndex = queue.front();
        queue.pop();
        int xCoord = frontIndex%image.width;
        int yCoord = frontIndex/image.width;

        for (int dy=-1; dy<=1; dy++)
        {
            for (int dx=-1; dx<=1; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;
                int neighbourXCoord = xCoord + dx;
                int neighbourYCoord = yCoord + dy;
                if (neighbourXCoord<0 || neighbourXCoord>=image.width || neighbourYCoord<0 || neighbourYCoord>=image.height)
                    continue;
                int neighbourIndex = neighbourYCoord*image.width + neighbourXCoord;
                if (state[neighbourIndex] ==128)
                {
                    state[neighbourIndex] = 255;
                    queue.push(neighbourIndex);
                }

            }
        }
    }
    for (int value: state)
    {
        if (value == 128){value = 0;}
    }
}


