#include <metal_stdlib>

using namespace metal;

kernel void convolution3x3(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    device const float* filter [[buffer(4)]],
    constant int& absoluteValue [[buffer(5)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;

    int index = y * width + x;

    if (x==0 || y==0 || x==width-1 || y==height-1)
    {
    output[index] = input[index];
    return;
    }

    float sum = 0.0f;

    for (int ky = -1; ky<=1; ky++) 

                                   
    {
        for (int kx = -1; kx<=1; kx++)
        {
            int neighbourX = x + kx;
            int neighbourY = y + ky;

            int neighbourIndex = neighbourY * width + neighbourX;
            int kernelIndex = (ky + 1) * 3 + (kx + 1);

            sum+= input[neighbourIndex] * filter[kernelIndex];
        }
    }

        if (absoluteValue) {
            sum = fabs(sum);
        }

    sum = clamp(sum, 0.0f, 255.0f);
    output[index] = (unsigned char)sum;

}
