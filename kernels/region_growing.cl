__kernel void region_init(__global unsigned char* mask, int width, int height, int seedX, int seedY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x>=width || y>= height){return;}
    int index = y*width+x;
    if (x==seedX && y== seedY){mask[index] = 255;}
    else {mask[index] = 0;}
}

__kernel void region_step(__global const unsigned char* input, __global const unsigned char* previousMask, __global unsigned char* nextMask, __global int* changed, int width, int height, int seedValue, int tolerance)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x>=width || y>= height){return;}
    int index = y*width+x;
    nextMask[index] = previousMask[index];
    if (previousMask[index] != 0){return;}
    int pixelValue = input[index];
    int difference = abs(pixelValue - seedValue);
    if (difference>tolerance){return;}
    bool touchesRegion = x>0 && previousMask[index-1]!=0 || x<width-1 && previousMask[index+1]!=0 || y>0&& previousMask[index-width]!=0 || y<height-1 && previousMask[index+width]!=0 ;
    if (touchesRegion)
    {
        nextMask[index] = 255;
        atomic_or(changed, 1);
    }


}