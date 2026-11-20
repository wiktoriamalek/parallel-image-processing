#include <metal_stdlib>

using namespace metal;

kernel void brown_conrady_correction(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant float& k1 [[buffer(4)]],
    constant float& k2 [[buffer(5)]],
    constant float& k3 [[buffer(6)]],
    constant float& p1 [[buffer(7)]],
    constant float& p2 [[buffer(8)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;

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

    float distortedPixelX = distortedX * scale + centerX;
    float distortedPixelY = distortedY * scale + centerY;

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
