__kernel void median_filter3x3(__global const unsigned char* input, __global unsigned char* output, int width, int height)
{
    // __global is a pointer to a global memory of a device

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x>=width || y>=height){
        return;
    }

    int pixelIndex = y * width + x;

    if (x==0 || x==width-1 || y==0 || y==height-1 )
    {
     output[pixelIndex] = input[pixelIndex];
     return;
    }

    unsigned char neighbours[9];
    int index = 0;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int neighbourX = x + kx;
            int neighbourY = y + ky;
            int neighbourIndex = neighbourY * width + neighbourX;
            neighbours[index] = input[neighbourIndex];
            index++;
        }
    }

    for (int i = 1; i < 9; i++) {
        unsigned char value = neighbours[i];
        int j = i - 1;
        while (j >= 0 && neighbours[j] > value) {
            neighbours[j + 1] = neighbours[j];
            j--;
            }
        neighbours[j + 1] = value;
    }

    output[pixelIndex] = neighbours[4];






















}