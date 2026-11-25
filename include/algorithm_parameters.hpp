#pragma once

enum class Algorithm { Threshold, Negative, Contrast, Otsu, MultiOtsu, PeaksValleys, Bernsen, Niblack, Sauvola,
                       Convolution, MedianFilter, Distortion, Resize, RegionGrowing, Canny, Gabor };
struct Parameters {
    int threshold = 128;
    float contrastFactor = 2.0f;
    int classes = 3;
    int maxClasses = 3;
    int radius = 7;
    int minContrast = 15;
    double localThresholdK = 0.2;
    int filter = 0;
    int medianSort = 0;
    int medianRadius = 1;
    int medianStrip = 8;
    bool medianSwitching = false;
    float k1 = 0.15f;
    float k2 = 0.0f;
    int distortionModel = 0;
    float k3 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
    bool bilinear = true;
    float scale = 2.0f;
    int seedX = 0;
    int seedY = 0;
    int tolerance = 20;
    bool localTiles = true;
    bool convolutionLocalTile = false;
    float cannyLow = 50.0f;
    float cannyHigh = 100.0f;
    float gaborSigma = 2.0f;
    float gaborWavelength = 6.0f;
    float gaborGamma = 0.5f;
    int gaborOrientations = 4;
};

