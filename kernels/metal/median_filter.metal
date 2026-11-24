#include <metal_stdlib>

using namespace metal;

#ifdef MEDIAN_RADIUS
#define MEDIAN_SIZE (2 * MEDIAN_RADIUS + 1)
#define MEDIAN_AREA (MEDIAN_SIZE * MEDIAN_SIZE)

kernel void median_insertion(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& switching [[buffer(4)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    const int x = gridPosition.x;
    const int y = gridPosition.y;
    if (x >= width || y >= height) {
        return;
    }
    const size_t index = (size_t)y * width + x;
    const uchar pixel = input[index];
    if (x < MEDIAN_RADIUS || x >= width - MEDIAN_RADIUS || y < MEDIAN_RADIUS || y >= height - MEDIAN_RADIUS ||
        (switching && pixel != 0 && pixel != 255)) {
        output[index] = pixel;
        return;
    }
    uchar neighbours[MEDIAN_AREA];
    int count = 0;
    for (int ky = -MEDIAN_RADIUS; ky <= MEDIAN_RADIUS; ky++) {
        for (int kx = -MEDIAN_RADIUS; kx <= MEDIAN_RADIUS; kx++) {
            neighbours[count++] = input[(size_t)(y + ky) * width + x + kx];
        }
    }
    for (int i = 1; i < MEDIAN_AREA; i++) {
        const uchar value = neighbours[i];
        int j = i - 1;
        while (j >= 0 && neighbours[j] > value) {
            neighbours[j + 1] = neighbours[j];
            j--;
        }
        neighbours[j + 1] = value;
    }
    output[index] = neighbours[MEDIAN_AREA / 2];
}
#endif

#ifdef MEDIAN_NETWORK
#define LD(w, row, column) v[w] = input[(size_t)clamp(y0 - MEDIAN_RADIUS + (row), 0, height - 1) * width + x - MEDIAN_RADIUS + (column)];
#define CS(a, b) { const uchar low = min(v[a], v[b]); v[b] = max(v[a], v[b]); v[a] = low; }
#define LO(a, b) v[a] = min(v[a], v[b]);
#define HI(a, b) v[b] = max(v[a], v[b]);
#define CP(a, b) v[a] = v[b];

kernel void median_network(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& switching [[buffer(4)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    const int x = gridPosition.x;
    const int y0 = gridPosition.y * MEDIAN_STRIP;
    if (x >= width || y0 >= height) {
        return;
    }
    const bool insideColumn = x >= MEDIAN_RADIUS && x < width - MEDIAN_RADIUS;
    uchar result[MEDIAN_STRIP];
    if (insideColumn) {
        uchar v[MEDIAN_WIRES];
        MEDIAN_NETWORK
        MEDIAN_OUTPUTS
    }
    for (int j = 0; j < MEDIAN_STRIP; j++) {
        const int y = y0 + j;
        if (y >= height) {
            break;
        }
        const size_t index = (size_t)y * width + x;
        const uchar pixel = input[index];
        const bool inside = insideColumn && y >= MEDIAN_RADIUS && y < height - MEDIAN_RADIUS;
        output[index] = inside && (!switching || pixel == 0 || pixel == 255) ? result[j] : pixel;
    }
}
#endif
