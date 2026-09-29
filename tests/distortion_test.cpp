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
#include "distortion.hpp"

int main() {
    // Real image
    Image image = loadImage("images/mitsubishi.png");

    // Create distorted image
    Image distorted = distortRadial(image, 0.75f, 0.75f);
    //saveImage("images/mitsubishi_distorted/mitusbishi_distorted_075_075.png", distorted);

    // CPU1
    Image correctedCPU1 = correctRadialDistortionCPU1(distorted, 0.75f,0.75f);
    saveImage("results/distortion/CPU_1T/mitsubishi_distortion_075_075.png", correctedCPU1);

    // CPU2
    Image correctedCPU2 = correctRadialDistortionCPU2(distorted, 0.75f,0.75f);
    saveImage("results/distortion/CPU2T/mitsubishi_distortion_075_075.png", correctedCPU2);

    // GPU
    Image correctedGPU = correctRadialDistortionGPU(distorted, 0.75f,0.75f);
    saveImage("results/distortion/GPU/mitsubishi_distortion_GPU_075_075.png", correctedGPU);

    return 0;
}
