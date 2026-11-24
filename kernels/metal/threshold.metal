#include <metal_stdlib>

using namespace metal;

kernel void threshold_kernel(
    device unsigned char* pixels [[buffer(0)]],
    constant int& threshold [[buffer(1)]],
    constant int& pixelCount [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{                              
    size_t i = gridPosition.x;
    if (i >= (size_t)pixelCount) {
        return;
    }
    if (pixels[i] < threshold) {
        pixels[i]=0;
     }
    else
    {
        pixels[i] = 255;
    }
}
