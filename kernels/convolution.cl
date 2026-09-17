__kernel void convolution3x3(__global const unsigned char* input, __global unsigned char* output, int width, int height, __global const float* filter, int absoluteValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

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
    output[index] = (unsigned char)sum;


}