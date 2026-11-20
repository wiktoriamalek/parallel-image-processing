#include <metal_stdlib>

using namespace metal;

kernel void threshold_kernel(
    device unsigned char* pixels [[buffer(0)]],
    constant int& threshold [[buffer(1)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
) 
{                              
    size_t i = gridPosition.x;
    if (pixels[i] < threshold) {
        pixels[i]=0;
     }
    else
    {
        pixels[i] = 255;
    }
}
