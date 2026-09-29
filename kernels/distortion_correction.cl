__kernel void distortion_correction(__global const unsigned char* input, __global unsigned char* output, int height, int width, float k1, float k2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > width-1 || y > height-1){
    return;
    }

    float centerX = (width - 1)/2.0f;
    float centerY = (height - 1)/2.0f;

    float scale = max(width, height) / 2.0f;

    float relativeX = x - centerX;
    float relativeY = y - centerY;

    float normalizedX = relativeX / scale;
    float normalizedY = relativeY / scale;

    float r2 = normalizedX*normalizedX + normalizedY*normalizedY; // squared distance from the center

    float factor = 1.0f + k1 * r2 + k2 * r2 * r2;

    float distortedX = normalizedX * factor; // radial distortion
    float distortedY = normalizedY * factor; // radial distortion

    float distortedRelativeX = distortedX * scale ;
    float distortedRelativeY = distortedY * scale ;

    float distortedx = distortedRelativeX + centerX;
    float distortedy = distortedRelativeY + centerY;

    int sourceX = (int)round(distortedx);
    int sourceY = (int)round(distortedy);

    int destinationIndex = y * width + x;

    if (sourceX < 0 || width <= sourceX || sourceY < 0 || height <= sourceY) {
        output[destinationIndex] = 0;
    }

    else {
        int sourceIndex = sourceY * width + sourceX;
        output[destinationIndex] = input[sourceIndex];
    }
}