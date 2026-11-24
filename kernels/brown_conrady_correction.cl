__kernel void brown_conrady_correction(__global const unsigned char* input, __global unsigned char* output, int width, int height,
                                       float k1, float k2, float k3, float p1, float p2)
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

    float r2 = normalizedX*normalizedX + normalizedY*normalizedY;
    float factor = 1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    float tangentialX = 2.0f * p1 * normalizedX * normalizedY + p2 * (r2 + 2.0f * normalizedX * normalizedX);
    float tangentialY = p1 * (r2 + 2.0f * normalizedY * normalizedY) + 2.0f * p2 * normalizedX * normalizedY;

    float distortedX = normalizedX * factor + tangentialX;
    float distortedY = normalizedY * factor + tangentialY;

    float distortedRelativeX = distortedX * scale;
    float distortedRelativeY = distortedY * scale;

    float distortedPixelX = distortedRelativeX + centerX;
    float distortedPixelY = distortedRelativeY + centerY;

    int destinationIndex = y * width + x;

    if (!(distortedPixelX >= 0 && distortedPixelX <= width - 1 && distortedPixelY >= 0 && distortedPixelY <= height - 1)) {
        output[destinationIndex] = 0;
        return;
    }

    int x0 = floor(distortedPixelX);
    int y0 = floor(distortedPixelY);
    int x1 = min(width - 1, x0 + 1);
    int y1 = min(height - 1, y0 + 1);
    float fx = distortedPixelX - x0;
    float fy = distortedPixelY - y0;
    float p00 = input[y0*width+x0];
    float p10 = input[y0*width+x1];
    float p01 = input[y1*width+x0];
    float p11 = input[y1*width+x1];
    float top = (1-fx)*p00+fx*p10;
    float bottom = (1-fx)*p01+fx*p11;
    float value = (1-fy)*top+fy*bottom;
    output[destinationIndex] = (unsigned char)round(value);
}
