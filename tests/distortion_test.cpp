#include "distortion.hpp"
#include <iostream>
#include "image_io.hpp"

int main() {
    Image image = loadImage("images/mitsubishi.png");
    Image distorted = distortRadial(image, 0.75f, 0.75f);
    saveImage("images/mitsubishi_distorted/mitusbishi_distorted_075_075.png", distorted);
    Image corrected = correctRadialDistortion(distorted, 0.75f,0.75f);
    // if (corrected.pixels == image.pixels) {
    //     std::cout<<"Corrected and image are the same"<<std::endl;
    // }
    saveImage("results/distortion/mitsubishi_distortion_075_075.png", corrected);
    return 0;
}