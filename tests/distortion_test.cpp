#include "image_io.hpp"
#define CL_SILENCE_DEPRECATION
#include <string>
#include "distortion.hpp"

int main() {
    // Real image
    Image image = loadImage("images/mitsubishi_resolution_change/mitsubishi_16384.png");

    // Create distorted image
    Image distorted = distortRadial(image, 0.15f, 0.0f);
    //saveImage("images/mitsubishi_distorted/mitusbishi_distorted_075_075.png", distorted);

    // CPU1
    Image correctedCPU1 = correctRadialDistortionCPU1(distorted, 0.15f, 0.0f);
    //saveImage("results/distortion/CPU_1T/mitsubishi_distortion_075_075.png", correctedCPU1);

    // CPU2
    Image correctedCPU2 = correctRadialDistortionCPU2(distorted, 0.15f, 0.0f);
    //saveImage("results/distortion/CPU2T/mitsubishi_distortion_075_075.png", correctedCPU2);

    // GPU
    Image correctedGPU = correctRadialDistortionGPU(distorted, 0.15f, 0.0f);
    //saveImage("results/distortion/GPU/mitsubishi_distortion_GPU_075_075.png", correctedGPU);

    return 0;
}
