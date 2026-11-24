#include <metal_stdlib>

using namespace metal;

constant int gaborGroup = 16;
constant int gaborPixels = 8;

static int gaborLevel(float squared)
{
    if (squared >= 65025.0f) {
        return 255;
    }
    int level = (int)sqrt(squared);
    if ((float)(level * level) > squared) {
        level--;
    } else if ((float)((level + 1) * (level + 1)) <= squared) {
        level++;
    }
    return level;
}

kernel void gabor(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    device const float2* weights [[buffer(4)]],
    constant int& radius [[buffer(5)]],
    constant int& orientations [[buffer(6)]],
    threadgroup unsigned char* tile [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
)
{
    const int localX = local.x;
    const int localY = local.y;
    const int tileWidth = gaborGroup * gaborPixels + 2 * radius;
    const int tileHeight = gaborGroup + 2 * radius;
    const int originX = int(group.x) * gaborGroup * gaborPixels - radius;
    const int originY = int(group.y) * gaborGroup - radius;
    for (int i = localY * gaborGroup + localX; i < tileWidth * tileHeight; i += gaborGroup * gaborGroup) {
        const int sourceX = clamp(originX + i % tileWidth, 0, width - 1);
        const int sourceY = clamp(originY + i / tileWidth, 0, height - 1);
        tile[i] = input[sourceY * width + sourceX];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int firstX = (int(group.x) * gaborGroup + localX) * gaborPixels;
    const int y = int(group.y) * gaborGroup + localY;
    if (firstX >= width || y >= height) {
        return;
    }

    const int size = 2 * radius + 1;
    device const float2* weight = weights;
    float best[gaborPixels];
    for (int orientation = 0; orientation < orientations; orientation++) {
        float even[gaborPixels];
        float odd[gaborPixels];
        for (int i = 0; i < gaborPixels; i++) {
            even[i] = 0.0f;
            odd[i] = 0.0f;
        }
        for (int ky = 0; ky < size; ky++) {
            threadgroup const unsigned char* row = tile + (localY + ky) * tileWidth + localX * gaborPixels;
            float values[gaborPixels];
            for (int i = 0; i < gaborPixels - 1; i++) {
                values[i] = row[i];
            }
            for (int kx = 0; kx < size; kx++) {
                values[gaborPixels - 1] = row[kx + gaborPixels - 1];
                const float2 pair = weight[kx];
                for (int i = 0; i < gaborPixels; i++) {
                    even[i] += values[i] * pair.x;
                    odd[i] += values[i] * pair.y;
                }
                for (int i = 0; i < gaborPixels - 1; i++) {
                    values[i] = values[i + 1];
                }
            }
            weight += size;
        }
        for (int i = 0; i < gaborPixels; i++) {
            float magnitude = even[i] * even[i];
            magnitude += odd[i] * odd[i];
            best[i] = orientation == 0 ? magnitude : max(best[i], magnitude);
        }
    }
    for (int i = 0; i < gaborPixels && firstX + i < width; i++) {
        output[y * width + firstX + i] = (unsigned char)gaborLevel(best[i]);
    }
}
