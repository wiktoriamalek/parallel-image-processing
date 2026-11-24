#define GABOR_GROUP 16
#define GABOR_PIXELS 8

int gaborLevel(float squared)
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

__kernel void gabor(__global const unsigned char* input, __global unsigned char* output, int width, int height,
                    __global const float2* weights, int radius, int orientations, __local unsigned char* tile)
{
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);
    const int tileWidth = GABOR_GROUP * GABOR_PIXELS + 2 * radius;
    const int tileHeight = GABOR_GROUP + 2 * radius;
    const int originX = get_group_id(0) * GABOR_GROUP * GABOR_PIXELS - radius;
    const int originY = get_group_id(1) * GABOR_GROUP - radius;
    for (int i = localY * GABOR_GROUP + localX; i < tileWidth * tileHeight; i += GABOR_GROUP * GABOR_GROUP) {
        const int sourceX = clamp(originX + i % tileWidth, 0, width - 1);
        const int sourceY = clamp(originY + i / tileWidth, 0, height - 1);
        tile[i] = input[sourceY * width + sourceX];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int firstX = get_global_id(0) * GABOR_PIXELS;
    const int y = get_global_id(1);
    if (firstX >= width || y >= height) {
        return;
    }

    const int size = 2 * radius + 1;
    __global const float2* weight = weights;
    float best[GABOR_PIXELS];
    for (int orientation = 0; orientation < orientations; orientation++) {
        float even[GABOR_PIXELS];
        float odd[GABOR_PIXELS];
        for (int i = 0; i < GABOR_PIXELS; i++) {
            even[i] = 0.0f;
            odd[i] = 0.0f;
        }
        for (int ky = 0; ky < size; ky++) {
            __local const unsigned char* row = tile + (localY + ky) * tileWidth + localX * GABOR_PIXELS;
            float values[GABOR_PIXELS];
            for (int i = 0; i < GABOR_PIXELS - 1; i++) {
                values[i] = row[i];
            }
            for (int kx = 0; kx < size; kx++) {
                values[GABOR_PIXELS - 1] = row[kx + GABOR_PIXELS - 1];
                const float2 pair = weight[kx];
                for (int i = 0; i < GABOR_PIXELS; i++) {
                    even[i] += values[i] * pair.x;
                    odd[i] += values[i] * pair.y;
                }
                for (int i = 0; i < GABOR_PIXELS - 1; i++) {
                    values[i] = values[i + 1];
                }
            }
            weight += size;
        }
        for (int i = 0; i < GABOR_PIXELS; i++) {
            float magnitude = even[i] * even[i];
            magnitude += odd[i] * odd[i];
            best[i] = orientation == 0 ? magnitude : max(best[i], magnitude);
        }
    }
    for (int i = 0; i < GABOR_PIXELS && firstX + i < width; i++) {
        output[y * width + firstX + i] = (unsigned char)gaborLevel(best[i]);
    }
}
