#include <metal_stdlib>

using namespace metal;

kernel void contrast_gpu(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant float& factor [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    size_t x = gridPosition.x;
    float scaled = (input[x] - 128) * factor;
    float value = scaled + 128.0f;
    output[x] = (unsigned char)round(clamp(value, 0.0f, 255.0f));
}
