#include "distortion.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <thread>
#include <functional>

Image distortRadial(const Image& image, float k1, float k2) {
    Image output = image;
    float centerX = (image.width - 1)/2.0f;
    float centerY = (image.height - 1)/2.0f;
    float scale = std::max(image.width, image.height) / 2.0f;

    for (int pixelX = 0; pixelX < image.width; pixelX++) {
        for (int pixelY = 0; pixelY < image.height; pixelY++) {
            float normalizedX = (pixelX - centerX) / scale;
            float normalizedY = (pixelY - centerY) / scale;
            float distortedRadius = std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY); // rd
            float lowerRadius = 0.0f;
            float upperRadius = distortedRadius;
            for (int iteration = 0; iteration < 25; iteration++) {
                float middleRadius = (lowerRadius + upperRadius)/2.0f;
                float ru = middleRadius;
                float r2 = middleRadius*middleRadius; // ru^2
                float calculatedRadius = ru * (1 + k1 * r2 + k2 * r2 * r2);
                if (calculatedRadius > distortedRadius) {
                    upperRadius = middleRadius;
                }
                else if (calculatedRadius < distortedRadius) {
                    lowerRadius = middleRadius;
                }

            }
            float undistortedRadius = (lowerRadius + upperRadius)/2.0f;
            float radiusRatio;
            if (distortedRadius == 0.0f) {
                radiusRatio = 1.0f;
            }
            else {
                radiusRatio = undistortedRadius/distortedRadius;
            }

            float sourceNormalizedX = normalizedX * radiusRatio;
            float sourceNormalizedY = normalizedY * radiusRatio;

            float sourceRelativeX = sourceNormalizedX * scale; // original picture, undistorted, relative to the center of the image
            float sourceRelativeY = sourceNormalizedY * scale;

            float sourcePixelX = sourceRelativeX + centerX;
            float sourcePixelY = sourceRelativeY + centerY;

            int sourceX = std::lround(sourcePixelX);
            int sourceY = std::lround(sourcePixelY);

            int destinationIndex = pixelY * image.width + pixelX;
            if (sourceX < 0 || image.width <= sourceX || sourceY < 0 || image.height <= sourceY) {
                output.pixels[destinationIndex] = 0;
            }

            else {
                int sourceIndex = sourceY * image.width + sourceX;
                output.pixels[destinationIndex] = image.pixels[sourceIndex];
            }
        }
    }
    return output;
}

Image correctRadialDistortion(const Image& image, float k1, float k2) {
    Image output = image;
    int middleY = image.height/2;

    std::thread thread1(correctRadialDistortionRange, std::cref(image), std::ref(output), k1, k2, 0, middleY);
    std::thread thread2(correctRadialDistortionRange, std::cref(image), std::ref(output), k1, k2, middleY, image.height);

    thread1.join();
    thread2.join();
    return output;
}

void correctRadialDistortionRange(const Image &image, Image &output, float k1, float k2, int startY, int endY) {
    float centerX = (image.width - 1)/2.0f;
    float centerY = (image.height - 1)/2.0f;

    float scale = std::max(image.width, image.height) / 2.0f;

    for (int pixelY = startY; pixelY < endY; pixelY++) {
        for (int pixelX = 0; pixelX < image.width; pixelX++) {
            float relativeX = pixelX - centerX;
            float relativeY = pixelY - centerY;

            float normalizedX = relativeX / scale;
            float normalizedY = relativeY / scale;

            float r2 = normalizedX*normalizedX + normalizedY*normalizedY; // squared distance from the center

            float factor = 1.0f + k1 * r2 + k2 * r2 * r2;

            float distortedX = normalizedX * factor; // radial distortion
            float distortedY = normalizedY * factor; // radial distortion

            float distortedRelativeX = distortedX * scale ;
            float distortedRelativeY = distortedY * scale ;

            float distortedPixelX = distortedRelativeX + centerX;
            float distortedPixelY = distortedRelativeY + centerY;

            int sourceX = std::lround(distortedPixelX);
            int sourceY = std::lround(distortedPixelY);

            int destinationIndex = pixelY * image.width + pixelX;

            if (sourceX < 0 || image.width <= sourceX || sourceY < 0 || image.height <= sourceY) {
                output.pixels[destinationIndex] = 0;
            }

            else {
                int sourceIndex = sourceY * image.width + sourceX;
                output.pixels[destinationIndex] = image.pixels[sourceIndex];
            }
        }
    }
}


Image correctRadialDistortion(const Image& image, float k1, float k2) {
    Image output = image;
    float centerX = (image.width - 1)/2.0f;
    float centerY = (image.height - 1)/2.0f;

    // int pixelX = 700;
    // int pixelY = 400;
    float scale = std::max(image.width, image.height) / 2.0f;

    for (int pixelX = 0; pixelX < image.width; pixelX++) {
        for (int pixelY = 0; pixelY < image.height; pixelY++) {
            float relativeX = pixelX - centerX;
            float relativeY = pixelY - centerY;

            float normalizedX = relativeX / scale;
            float normalizedY = relativeY / scale;

            float r2 = normalizedX*normalizedX + normalizedY*normalizedY; // squared distance from the center

            float factor = 1.0f + k1 * r2 + k2 * r2 * r2;

            float distortedX = normalizedX * factor; // radial distortion
            float distortedY = normalizedY * factor; // radial distortion

            float distortedRelativeX = distortedX * scale ;
            float distortedRelativeY = distortedY * scale ;

            float distortedPixelX = distortedRelativeX + centerX;
            float distortedPixelY = distortedRelativeY + centerY;

            int sourceX = std::lround(distortedPixelX);
            int sourceY = std::lround(distortedPixelY);

            int destinationIndex = pixelY * image.width + pixelX;
            if (sourceX < 0 || image.width <= sourceX || sourceY < 0 || image.height <= sourceY) {
                output.pixels[destinationIndex] = 0;
            }

            else {
                int sourceIndex = sourceY * image.width + sourceX;
                output.pixels[destinationIndex] = image.pixels[sourceIndex];
            }
        }
    }
    return output;
}