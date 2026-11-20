#include <metal_stdlib>

using namespace metal;

kernel void negative_gpu(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    size_t x = gridPosition.x;
    unsigned char intensity = input[x];
    output[x]=255-intensity;
}
