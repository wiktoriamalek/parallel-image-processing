#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.hpp"
#include <stdexcept>
#include <thread>
#include <chrono>
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

int main() {
    try {
        std::array<std::array<float, 3>, 3> identity = {
            {
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 0},
            }};

        std::array<std::array<float, 3>, 3> boxBlur = {
            {
                // box blur 3x3 - it takes for each pixel, 9 values from the surrounding pixels and calculates mean
                {1.0f/9, 1.0f/9, 1.0f/9},
                {1.0f/9, 1.0f/9, 1.0f/9},
                {1.0f/9, 1.0f/9, 1.0f/9},
            }};

        std::array<std::array<float, 3>, 3> edgeDetectionH = {
            {
                {-1, -1, -1},
                {0,   0,  0},
                {1,   1,  1},
            }};

        std::array<std::array<float, 3>, 3> edgeDetectionV = {
            {
                {-1,  0,  1},
                {-1,  0,  1},
                {-1,  0,  1},
            }};

        std::array<std::array<float, 3>, 3> sharpen = {
            {
                {0,  -1,  0},
                {-1,  5, -1},
                {0,  -1,  0},
            }};


        Image image = loadImage("images/cms.png");
        Image identical = convolution(image, identity, false);

        if (identical.pixels == image.pixels) {
            std::cout << "Identity convolution works!\n";
        }
        saveImage("results/cms_identity.png", identical);

        Image blurred = convolution(image, boxBlur, false);
        saveImage("results/cms_blurred.png", blurred);

        Image sharpened = convolution(image, sharpen, false);
        saveImage("results/cms_sharpened.png", sharpened);

        Image edgeDetectedH = convolution(image, edgeDetectionH, true);
        saveImage("results/cms_edgeDetectionH.png", edgeDetectedH);

        Image edgeDetectedV = convolution(image, edgeDetectionV, true);
        saveImage("results/cms_edgeDetectionV.png", edgeDetectedV);

        return 0;
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}