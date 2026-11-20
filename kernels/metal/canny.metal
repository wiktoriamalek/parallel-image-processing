#include <metal_stdlib>

using namespace metal;

#define WEAK_EDGE 128
#define STRONG_EDGE 255
#define TAN_22_5_DEGREES 27146
#define FIXED_POINT_ONE 65536
#define TRACK_TILE 64
#define TRACK_GROUP 16
#define TRACK_PIXELS (TRACK_TILE / TRACK_GROUP)

kernel void canny_gauss(
    device const uchar* input [[buffer(0)]],
    device uchar* blurred [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    const int x = gridPosition.x;
    const int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        blurred[index] = input[index];
        return;
    }
    device const uchar* p = input + index;
    const int sum = p[-width - 1] + 2 * p[-width] + p[-width + 1] +
                    2 * p[-1] + 4 * p[0] + 2 * p[1] +
                    p[width - 1] + 2 * p[width] + p[width + 1];
    blurred[index] = (uchar)((sum + 8) / 16);
}

kernel void canny_gradient(
    device const uchar* blurred [[buffer(0)]],
    device int* magnitude [[buffer(1)]],
    device uchar* sector [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant int& height [[buffer(4)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    const int x = gridPosition.x;
    const int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        magnitude[index] = 0;
        sector[index] = 0;
        return;
    }
    device const uchar* p = blurred + index;
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

kernel void canny_suppress(
    device const int* magnitude [[buffer(0)]],
    device const uchar* sector [[buffer(1)]],
    device uchar* state [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& lowSquared [[buffer(5)]],
    constant int& highSquared [[buffer(6)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    const int x = gridPosition.x;
    const int y = gridPosition.y;
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

kernel void canny_track(
    device uchar* state [[buffer(0)]],
    device int* changed [[buffer(1)]],
    device const int* previousActive [[buffer(2)]],
    device int* nextActive [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& tilesX [[buffer(6)]],
    constant int& tilesY [[buffer(7)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    threadgroup uchar tile[TRACK_TILE + 2][TRACK_TILE + 2];
    threadgroup int passChanged[3];
    threadgroup int tileGrew;
    const int tileX = groupPosition.x;
    const int tileY = groupPosition.y;
    const int tileIndex = tileY * tilesX + tileX;
    const int localX = localPosition.x;
    const int localY = localPosition.y;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (item == 0) {
        nextActive[tileIndex] = tileGrew;
        if (tileGrew != 0) {
            *changed = 1;
        }
    }
}

kernel void canny_finish(
    device const uchar* state [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    const int x = gridPosition.x;
    const int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    output[index] = state[index] == STRONG_EDGE ? STRONG_EDGE : 0;
}
