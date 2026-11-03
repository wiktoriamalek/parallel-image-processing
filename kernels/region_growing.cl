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
        *changed = 1;
    }


}

#define TILE 64
#define GROUP 16
#define PIXELS_PER_ITEM (TILE / GROUP)

__kernel void region_active_init(__global int* active, int tileCount, int seedTile)
{
    int tile = get_global_id(0);
    if (tile >= tileCount) {return;}
    active[tile] = tile == seedTile;
}

__kernel void region_tile_step(__global const unsigned char* input, __global unsigned char* mask, __global int* changed,
                               __global const int* previousActive, __global int* nextActive,
                               int width, int height, int seedValue, int tolerance, int tilesX, int tilesY)
{
    __local unsigned char tile[TILE + 2][TILE + 2];
    __local int tileChanged;
    int tileX = get_group_id(0);
    int tileY = get_group_id(1);
    int tileIndex = tileY * tilesX + tileX;
    int localX = get_local_id(0);
    int localY = get_local_id(1);
    bool firstItem = localX == 0 && localY == 0;
    bool active = previousActive[tileIndex] ||
                  (tileX > 0 && previousActive[tileIndex - 1]) || (tileX < tilesX - 1 && previousActive[tileIndex + 1]) ||
                  (tileY > 0 && previousActive[tileIndex - tilesX]) || (tileY < tilesY - 1 && previousActive[tileIndex + tilesX]);
    if (!active)
    {
        if (firstItem) {nextActive[tileIndex] = 0;}
        return;
    }

    int startX = tileX * TILE;
    int startY = tileY * TILE;
    bool candidate[PIXELS_PER_ITEM][PIXELS_PER_ITEM];
    bool grown[PIXELS_PER_ITEM][PIXELS_PER_ITEM];
    for (int j = 0; j < PIXELS_PER_ITEM; j++)
    {
        for (int i = 0; i < PIXELS_PER_ITEM; i++)
        {
            int column = localX * PIXELS_PER_ITEM + i;
            int row = localY * PIXELS_PER_ITEM + j;
            int x = startX + column;
            int y = startY + row;
            bool inside = x < width && y < height;
            int index = y * width + x;
            candidate[j][i] = inside && abs(input[index] - seedValue) <= tolerance;
            grown[j][i] = false;
            tile[row + 1][column + 1] = inside ? mask[index] : 0;
        }
    }
    for (int k = 0; k < PIXELS_PER_ITEM; k++)
    {
        int column = localX * PIXELS_PER_ITEM + k;
        int row = localY * PIXELS_PER_ITEM + k;
        int x = startX + column;
        int y = startY + row;
        if (localX == 0) {tile[row + 1][0] = (startX > 0 && y < height) ? mask[y * width + startX - 1] : 0;}
        if (localX == GROUP - 1) {tile[row + 1][TILE + 1] = (startX + TILE < width && y < height) ? mask[y * width + startX + TILE] : 0;}
        if (localY == 0) {tile[0][column + 1] = (startY > 0 && x < width) ? mask[(startY - 1) * width + x] : 0;}
        if (localY == GROUP - 1) {tile[TILE + 1][column + 1] = (startY + TILE < height && x < width) ? mask[(startY + TILE) * width + x] : 0;}
    }

    bool tileGrew = false;
    while (true)
    {
        if (firstItem) {tileChanged = 0;}
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < PIXELS_PER_ITEM; j++)
        {
            for (int i = 0; i < PIXELS_PER_ITEM; i++)
            {
                int column = localX * PIXELS_PER_ITEM + i + 1;
                int row = localY * PIXELS_PER_ITEM + j + 1;
                if (candidate[j][i] && tile[row][column] == 0 &&
                    (tile[row][column - 1] | tile[row][column + 1] | tile[row - 1][column] | tile[row + 1][column]) != 0)
                {
                    tile[row][column] = 255;
                    grown[j][i] = true;
                    tileChanged = 1;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tileChanged == 0) {break;}
        tileGrew = true;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int j = 0; j < PIXELS_PER_ITEM; j++)
    {
        for (int i = 0; i < PIXELS_PER_ITEM; i++)
        {
            if (grown[j][i]) {mask[(startY + localY * PIXELS_PER_ITEM + j) * width + startX + localX * PIXELS_PER_ITEM + i] = 255;}
        }
    }
    if (firstItem)
    {
        nextActive[tileIndex] = tileGrew;
        if (tileGrew) {*changed = 1;}
    }
}
