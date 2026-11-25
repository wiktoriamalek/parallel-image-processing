#include <metal_stdlib>

using namespace metal;

kernel void convolution3x3(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    device const float* filter [[buffer(4)]],
    constant int& absoluteValue [[buffer(5)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }

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
    output[index] = (unsigned char)round(sum);

}


constant int convolutionTile = 16;

kernel void convolution3x3_local(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    device const float* filter [[buffer(4)]],
    constant int& absoluteValue [[buffer(5)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupSize [[threads_per_threadgroup]]
)
{
    threadgroup unsigned char tile[convolutionTile + 2][convolutionTile + 2];
    int localX = localPosition.x;
    int localY = localPosition.y;
    int originX = int(gridPosition.x) - localX - 1;
    int originY = int(gridPosition.y) - localY - 1;
    int threads = int(groupSize.x * groupSize.y);
    for (int i = localY * int(groupSize.x) + localX; i < (convolutionTile + 2) * (convolutionTile + 2); i += threads) {
        int tileX = i % (convolutionTile + 2);
        int tileY = i / (convolutionTile + 2);
        int sourceX = clamp(originX + tileX, 0, width - 1);
        int sourceY = clamp(originY + tileY, 0, height - 1);
        tile[tileY][tileX] = input[sourceY * width + sourceX];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int x = gridPosition.x;
    int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }

    int index = y * width + x;

    if (x==0 || y==0 || x==width-1 || y==height-1)
    {
    output[index] = tile[localY + 1][localX + 1];
    return;
    }

    float sum = 0.0f;

    for (int ky = -1; ky<=1; ky++)
    {
        for (int kx = -1; kx<=1; kx++)
        {
            int kernelIndex = (ky + 1) * 3 + (kx + 1);

            sum+= tile[localY + 1 + ky][localX + 1 + kx] * filter[kernelIndex];
        }
    }

        if (absoluteValue) {
            sum = fabs(sum);
        }

    sum = clamp(sum, 0.0f, 255.0f);
    output[index] = (unsigned char)round(sum);
}
