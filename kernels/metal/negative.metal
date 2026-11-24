#include <metal_stdlib>

using namespace metal;

kernel void negative_gpu(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& pixelCount [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    size_t x = gridPosition.x;
    if (x >= (size_t)pixelCount) {
        return;
    }
    unsigned char intensity = input[x];
    output[x]=255-intensity;
}
