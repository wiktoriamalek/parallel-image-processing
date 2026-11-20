#define WEAK_EDGE 128
#define STRONG_EDGE 255
#define TAN_22_5_DEGREES 27146
#define FIXED_POINT_ONE 65536
#define TRACK_TILE 64
#define TRACK_GROUP 16
#define TRACK_PIXELS (TRACK_TILE / TRACK_GROUP)

__kernel void canny_gauss(__global const uchar* input, __global uchar* blurred, int width, int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        blurred[index] = input[index];
        return;
    }
    __global const uchar* p = input + index;
    const int sum = p[-width - 1] + 2 * p[-width] + p[-width + 1] +
                    2 * p[-1] + 4 * p[0] + 2 * p[1] +
                    p[width - 1] + 2 * p[width] + p[width + 1];
    blurred[index] = (uchar)((sum + 8) / 16);
}

__kernel void canny_gradient(__global const uchar* blurred, __global int* magnitude, __global uchar* sector,
                             int width, int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        magnitude[index] = 0;
        sector[index] = 0;
        return;
    }
    __global const uchar* p = blurred + index;
    const int gx = (p[-width + 1] + 2 * p[1] + p[width + 1]) - (p[-width - 1] + 2 * p[-1] + p[width - 1]);
    const int gy = (p[width - 1] + 2 * p[width] + p[width + 1]) - (p[-width - 1] + 2 * p[-width] + p[-width + 1]);
    magnitude[index] = gx * gx + gy * gy;
    const int ax = (int)abs(gx);
    const int ay = (int)abs(gy);
    uchar direction;
    if (ay * FIXED_POINT_ONE < TAN_22_5_DEGREES * ax) {
        direction = 0;
    } else if (ax * FIXED_POINT_ONE <= TAN_22_5_DEGREES * ay) {
        direction = 2;
    } else {
        direction = (gx > 0) == (gy > 0) ? 1 : 3;
    }
    sector[index] = direction;
}

__kernel void canny_suppress(__global const int* magnitude, __global const uchar* sector, __global uchar* state,
                             int width, int height, int lowSquared, int highSquared)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        state[index] = 0;
        return;
    }
    const int direction = sector[index];
    const int offset = direction == 0 ? 1 : direction == 1 ? width + 1 : direction == 2 ? width : width - 1;
    const int candidate = magnitude[index];
    uchar value = 0;
    if (candidate > magnitude[index - offset] && candidate >= magnitude[index + offset]) {
        value = candidate >= highSquared ? STRONG_EDGE : candidate >= lowSquared ? WEAK_EDGE : 0;
    }
    state[index] = value;
}

__kernel void canny_track(__global uchar* state, __global int* changed, __global const int* previousActive,
                          __global int* nextActive, int width, int height, int tilesX, int tilesY)
{
    __local uchar tile[TRACK_TILE + 2][TRACK_TILE + 2];
    __local int passChanged[3];
    __local int tileGrew;
    const int tileX = get_group_id(0);
    const int tileY = get_group_id(1);
    const int tileIndex = tileY * tilesX + tileX;
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);
    const int item = localY * TRACK_GROUP + localX;

    bool active = false;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int neighbourX = tileX + dx;
            const int neighbourY = tileY + dy;
            if (neighbourX >= 0 && neighbourX < tilesX && neighbourY >= 0 && neighbourY < tilesY &&
                previousActive[neighbourY * tilesX + neighbourX] != 0) {
                active = true;
            }
        }
    }
    if (!active) {
        if (item == 0) {
            nextActive[tileIndex] = 0;
        }
        return;
    }

    const int startX = tileX * TRACK_TILE;
    const int startY = tileY * TRACK_TILE;
    for (int i = item; i < (TRACK_TILE + 2) * (TRACK_TILE + 2); i += TRACK_GROUP * TRACK_GROUP) {
        const int row = i / (TRACK_TILE + 2);
        const int column = i % (TRACK_TILE + 2);
        const int x = startX + column - 1;
        const int y = startY + row - 1;
        tile[row][column] = x >= 0 && x < width && y >= 0 && y < height ? state[y * width + x] : 0;
    }
    bool grown[TRACK_PIXELS][TRACK_PIXELS];
    for (int j = 0; j < TRACK_PIXELS; j++) {
        for (int i = 0; i < TRACK_PIXELS; i++) {
            grown[j][i] = false;
        }
    }
    if (item == 0) {
        passChanged[0] = 0;
        passChanged[1] = 0;
        passChanged[2] = 0;
        tileGrew = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool anyGrown = false;
    for (int pass = 0;; pass++) {
        bool changedHere = false;
        for (bool again = true; again;) {
            again = false;
            for (int j = 0; j < TRACK_PIXELS; j++) {
                for (int i = 0; i < TRACK_PIXELS; i++) {
                    const int row = localY * TRACK_PIXELS + j + 1;
                    const int column = localX * TRACK_PIXELS + i + 1;
                    if (tile[row][column] == WEAK_EDGE &&
                        (tile[row - 1][column - 1] == STRONG_EDGE || tile[row - 1][column] == STRONG_EDGE ||
                         tile[row - 1][column + 1] == STRONG_EDGE || tile[row][column - 1] == STRONG_EDGE ||
                         tile[row][column + 1] == STRONG_EDGE || tile[row + 1][column - 1] == STRONG_EDGE ||
                         tile[row + 1][column] == STRONG_EDGE || tile[row + 1][column + 1] == STRONG_EDGE)) {
                        tile[row][column] = STRONG_EDGE;
                        grown[j][i] = true;
                        again = true;
                        changedHere = true;
                        anyGrown = true;
                    }
                }
            }
        }
        if (changedHere) {
            passChanged[pass % 3] = 1;
        }
        if (item == 0) {
            passChanged[(pass + 1) % 3] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (passChanged[pass % 3] == 0) {
            break;
        }
    }

    for (int j = 0; j < TRACK_PIXELS; j++) {
        for (int i = 0; i < TRACK_PIXELS; i++) {
            if (grown[j][i]) {
                const int x = startX + localX * TRACK_PIXELS + i;
                const int y = startY + localY * TRACK_PIXELS + j;
                state[y * width + x] = STRONG_EDGE;
            }
        }
    }
    if (anyGrown) {
        tileGrew = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item == 0) {
        nextActive[tileIndex] = tileGrew;
        if (tileGrew != 0) {
            *changed = 1;
        }
    }
}

__kernel void canny_finish(__global const uchar* state, __global uchar* output, int width, int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    output[index] = state[index] == STRONG_EDGE ? STRONG_EDGE : 0;
}
