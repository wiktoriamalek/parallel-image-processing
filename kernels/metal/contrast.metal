#include <metal_stdlib>

using namespace metal;

kernel void contrast_gpu(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant float& factor [[buffer(2)]],
    constant int& pixelCount [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    size_t x = gridPosition.x;
    if (x >= (size_t)pixelCount) {
        return;
    }
    float scaled = (input[x] - 128) * factor;
    float value = scaled + 128.0f;
    output[x] = (unsigned char)round(clamp(value, 0.0f, 255.0f));
}
