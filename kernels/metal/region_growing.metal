#include <metal_stdlib>

using namespace metal;

kernel void region_init(
    device unsigned char* mask [[buffer(0)]],
    constant int& width [[buffer(1)]],
    constant int& height [[buffer(2)]],
    constant int& seedX [[buffer(3)]],
    constant int& seedY [[buffer(4)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;

    if (x>=width || y>= height){return;}
    int index = y*width+x;
    if (x==seedX && y== seedY){mask[index] = 255;}
    else {mask[index] = 0;}
}

kernel void region_step(
    device const unsigned char* input [[buffer(0)]],
    device const unsigned char* previousMask [[buffer(1)]],
    device unsigned char* nextMask [[buffer(2)]],
    device int* changed [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& seedValue [[buffer(6)]],
    constant int& tolerance [[buffer(7)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;

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

kernel void region_active_init(
    device int* active [[buffer(0)]],
    constant int& tileCount [[buffer(1)]],
    constant int& seedTile [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int tile = gridPosition.x;
    if (tile >= tileCount) {return;}
    active[tile] = tile == seedTile;
}

kernel void region_tile_step(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* mask [[buffer(1)]],
    device int* changed [[buffer(2)]],
    device const int* previousActive [[buffer(3)]],
    device int* nextActive [[buffer(4)]],
    constant int& width [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& seedValue [[buffer(7)]],
    constant int& tolerance [[buffer(8)]],
    constant int& tilesX [[buffer(9)]],
    constant int& tilesY [[buffer(10)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    threadgroup unsigned char tile[TILE + 2][TILE + 2];
    threadgroup int tileChanged;
    int tileX = groupPosition.x;
    int tileY = groupPosition.y;
    int tileIndex = tileY * tilesX + tileX;
    int localX = localPosition.x;
    int localY = localPosition.y;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tileChanged == 0) {break;}
        tileGrew = true;
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
