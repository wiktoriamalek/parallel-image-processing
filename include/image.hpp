// size of an image - width height and an array of pixels

#pragma once // include this only once in a code

#include <vector>

struct Image {
    int width=0; // number of pixel columns
    int height=0; // number of pixel rows
    std::vector<unsigned char> pixels; // intensity (0 - 255)
    // (x,y) = y * image.width + x
};


