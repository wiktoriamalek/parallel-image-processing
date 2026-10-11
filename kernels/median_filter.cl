// the smaller value goes to a, the bigger one to b - min and max instead of if, so every work-item runs the same instructions
void compareSwap(unsigned char* a, unsigned char* b)
{
    unsigned char low = min(*a, *b);
    *b = max(*a, *b);
    *a = low;
}

__kernel void median_filter3x3(__global const unsigned char* input, __global unsigned char* output, int width, int height, int sortingNetwork)
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

    if (sortingNetwork) {
        // the same 19 comparisons as medianOf9 in src/median_filter.cpp, the median ends up in neighbours[4]
        compareSwap(&neighbours[1], &neighbours[2]);
        compareSwap(&neighbours[4], &neighbours[5]);
        compareSwap(&neighbours[7], &neighbours[8]);

        compareSwap(&neighbours[0], &neighbours[1]);
        compareSwap(&neighbours[3], &neighbours[4]);
        compareSwap(&neighbours[6], &neighbours[7]);

        compareSwap(&neighbours[1], &neighbours[2]);
        compareSwap(&neighbours[4], &neighbours[5]);
        compareSwap(&neighbours[7], &neighbours[8]);

        compareSwap(&neighbours[0], &neighbours[3]);
        compareSwap(&neighbours[5], &neighbours[8]);
        compareSwap(&neighbours[4], &neighbours[7]);
        compareSwap(&neighbours[3], &neighbours[6]);

        compareSwap(&neighbours[1], &neighbours[4]);
        compareSwap(&neighbours[2], &neighbours[5]);
        compareSwap(&neighbours[4], &neighbours[7]);

        compareSwap(&neighbours[4], &neighbours[2]);
        compareSwap(&neighbours[6], &neighbours[4]);
        compareSwap(&neighbours[4], &neighbours[2]);
    }
    else {
        for (int i = 1; i < 9; i++) {
            unsigned char value = neighbours[i];
            int j = i - 1;
            while (j >= 0 && neighbours[j] > value) {
                neighbours[j + 1] = neighbours[j];
                j--;
                }
            neighbours[j + 1] = value;
        }
    }

    output[pixelIndex] = neighbours[4];






















}