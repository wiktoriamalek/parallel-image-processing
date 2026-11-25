__kernel void convolution3x3(__global const unsigned char* input, __global unsigned char* output, int width, int height, __global const float* filter, int absoluteValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    int index = y * width + x;

    // border pixels
    if (x==0 || y==0 || x==width-1 || y==height-1)
    {
    output[index] = input[index];
    return;
    }

    float sum = 0.0f;

    for (int ky = -1; ky<=1; ky++) // ky = -1 is upper row, kx = -1 is left column
                                   // ky = 0 is middle row, kx = 0 is middle column
                                   // ky = 1 is lower row, kx = 1 is right column
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

    // code above is able to find 9 neighbouring pixels around a certain pixel

//     0 1 2
//   0 x x x
//   1 x x x
//   2 x x x
        if (absoluteValue) {
            sum = fabs(sum);
        }


    sum = clamp(sum, 0.0f, 255.0f);
    output[index] = (unsigned char)round(sum);


}

#define CONVOLUTION_TILE 16

__kernel void convolution3x3_local(__global const unsigned char* input, __global unsigned char* output, int width, int height, __global const float* filter, int absoluteValue)
{
    __local unsigned char tile[CONVOLUTION_TILE + 2][CONVOLUTION_TILE + 2];
    int localX = get_local_id(0);
    int localY = get_local_id(1);
    int originX = get_group_id(0) * CONVOLUTION_TILE - 1;
    int originY = get_group_id(1) * CONVOLUTION_TILE - 1;
    for (int i = localY * CONVOLUTION_TILE + localX; i < (CONVOLUTION_TILE + 2) * (CONVOLUTION_TILE + 2); i += CONVOLUTION_TILE * CONVOLUTION_TILE) {
        int tileX = i % (CONVOLUTION_TILE + 2);
        int tileY = i / (CONVOLUTION_TILE + 2);
        int sourceX = clamp(originX + tileX, 0, width - 1);
        int sourceY = clamp(originY + tileY, 0, height - 1);
        tile[tileY][tileX] = input[sourceY * width + sourceX];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = get_global_id(0);
    int y = get_global_id(1);
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
