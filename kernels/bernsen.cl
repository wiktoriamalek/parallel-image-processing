__kernel void bernsenRowMinMax(__global const uchar* input, __global uchar2* rowMinMax, int width, int height, int radius)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    __global const uchar* row = input + (size_t)y * width;
    uchar low = 255;
    uchar high = 0;
    for (int nx = max(x - radius, 0); nx <= min(width - 1, x + radius); nx++) {
        low = min(low, row[nx]);
        high = max(high, row[nx]);
    }
    rowMinMax[(size_t)y * width + x] = (uchar2)(low, high);
}

__kernel void bernsenColumns(__global const uchar* input, __global const uchar2* rowMinMax, __global uchar* output,
                             int width, int height, int radius, int minContrast)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    uchar low = 255;
    uchar high = 0;
    for (int ny = max(y - radius, 0); ny <= min(height - 1, y + radius); ny++) {
        const uchar2 value = rowMinMax[(size_t)ny * width + x];
        low = min(low, value.x);
        high = max(high, value.y);
    }
    const int Jmin = low;
    const int Jmax = high;
    const int threshold = (Jmax + Jmin) / 2;
    const size_t index = (size_t)y * width + x;
    output[index] = Jmax - Jmin < minContrast || input[index] < threshold ? 0 : 255;
}

#define BERNSEN_ROW_OUTPUTS 2048
#define BERNSEN_TILE_COLUMNS 32
#define BERNSEN_TILE_ROWS 64

__kernel void bernsenRowMinMaxLocal(__global const uchar* input, __global uchar2* rowMinMax, int width, int height,
                                    int radius, __local uchar* scratch)
{
    const int id = get_local_id(0);
    const int items = get_local_size(0);
    const int y = get_group_id(1);
    const int x0 = get_group_id(0) * BERNSEN_ROW_OUTPUTS;
    const int length = BERNSEN_ROW_OUTPUTS + 2 * radius;
    __local uchar* low[2] = {scratch, scratch + length};
    __local uchar* high[2] = {scratch + 2 * length, scratch + 3 * length};
    __global const uchar* row = input + (size_t)y * width;
    for (int i = id; i < length; i += items) {
        const int x = x0 - radius + i;
        const bool inside = x >= 0 && x < width;
        const uchar value = inside ? row[x] : 0;
        low[0][i] = inside ? value : 255;
        high[0][i] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int window = 2 * radius + 1;
    int span = 1;
    int buffer = 0;
    while (2 * span <= window) {
        for (int i = id; i + span < length; i += items) {
            low[1 - buffer][i] = min(low[buffer][i], low[buffer][i + span]);
            high[1 - buffer][i] = max(high[buffer][i], high[buffer][i + span]);
        }
        buffer = 1 - buffer;
        span *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int o = id; o < BERNSEN_ROW_OUTPUTS && x0 + o < width; o += items) {
        const int other = o + window - span;
        rowMinMax[(size_t)y * width + x0 + o] = (uchar2)(min(low[buffer][o], low[buffer][other]),
                                                         max(high[buffer][o], high[buffer][other]));
    }
}

__kernel void bernsenColumnsLocal(__global const uchar* input, __global const uchar2* rowMinMax, __global uchar* output,
                                  int width, int height, int radius, int minContrast, __local uchar2* scratch)
{
    const int column = get_local_id(0);
    const int lane = get_local_id(1);
    const int lanes = get_local_size(1);
    const int x = get_group_id(0) * BERNSEN_TILE_COLUMNS + column;
    const int y0 = get_group_id(1) * BERNSEN_TILE_ROWS;
    const int length = BERNSEN_TILE_ROWS + 2 * radius;
    __local uchar2* values[2] = {scratch, scratch + length * BERNSEN_TILE_COLUMNS};
    for (int t = lane; t < length; t += lanes) {
        const int y = y0 - radius + t;
        const bool inside = x < width && y >= 0 && y < height;
        values[0][t * BERNSEN_TILE_COLUMNS + column] = inside ? rowMinMax[(size_t)y * width + x] : (uchar2)(255, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int window = 2 * radius + 1;
    int span = 1;
    int buffer = 0;
    while (2 * span <= window) {
        for (int t = lane; t + span < length; t += lanes) {
            const uchar2 a = values[buffer][t * BERNSEN_TILE_COLUMNS + column];
            const uchar2 b = values[buffer][(t + span) * BERNSEN_TILE_COLUMNS + column];
            values[1 - buffer][t * BERNSEN_TILE_COLUMNS + column] = (uchar2)(min(a.x, b.x), max(a.y, b.y));
        }
        buffer = 1 - buffer;
        span *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x >= width) {
        return;
    }
    for (int o = lane; o < BERNSEN_TILE_ROWS && y0 + o < height; o += lanes) {
        const uchar2 a = values[buffer][o * BERNSEN_TILE_COLUMNS + column];
        const uchar2 b = values[buffer][(o + window - span) * BERNSEN_TILE_COLUMNS + column];
        const int Jmin = min(a.x, b.x);
        const int Jmax = max(a.y, b.y);
        const int threshold = (Jmax + Jmin) / 2;
        const size_t index = (size_t)(y0 + o) * width + x;
        output[index] = Jmax - Jmin < minContrast || input[index] < threshold ? 0 : 255;
    }
}

#define LOCAL_THRESHOLD_COLUMNS 8

uchar localThresholdValue(long count, uint sum, uint squares, long pixel, float k, float stddevRange, bool sauvola)
{
    const float deviation = sqrt((float)(count * squares - (ulong)sum * sum));
    if (sauvola) {
        const float threshold = sum * (1.0f - k) + sum * k * deviation / (count * stddevRange);
        return (float)(count * pixel) < threshold ? 0 : 255;
    }
    return (float)((long)sum - count * pixel) > k * deviation ? 0 : 255;
}

void localThresholdBand(__global const uchar* input, __global uchar* output, int width, int height, int radius,
                        float k, float stddevRange, int bandHeight, __local uint* localSum,
                        __local uint* localSquares, bool sauvola)
{
    const int id = get_local_id(0);
    const int items = get_local_size(0);
    const int stripOutput = items * LOCAL_THRESHOLD_COLUMNS;
    const int stripWidth = stripOutput + 2 * radius;
    const int stripStart = get_group_id(0) * stripOutput - radius;
    const int firstRow = get_group_id(1) * bandHeight;
    const int endRow = min(firstRow + bandHeight, height);

    uint columnSum[LOCAL_THRESHOLD_COLUMNS + 1];
    uint columnSquares[LOCAL_THRESHOLD_COLUMNS + 1];
    for (int j = 0; j <= LOCAL_THRESHOLD_COLUMNS; j++) {
        columnSum[j] = 0;
        columnSquares[j] = 0;
        const int s = id + j * items;
        const int x = stripStart + s;
        if (s < stripWidth && x >= 0 && x < width) {
            for (int row = max(0, firstRow - radius); row <= min(height - 1, firstRow + radius); row++) {
                const uint value = input[(size_t)row * width + x];
                columnSum[j] += value;
                columnSquares[j] += value * value;
            }
        }
    }

    for (int y = firstRow; y < endRow; y++) {
        for (int j = 0; j <= LOCAL_THRESHOLD_COLUMNS; j++) {
            const int s = id + j * items;
            const int x = stripStart + s;
            if (s >= stripWidth) {
                break;
            }
            if (y > firstRow && x >= 0 && x < width) {
                if (y + radius < height) {
                    const uint value = input[(size_t)(y + radius) * width + x];
                    columnSum[j] += value;
                    columnSquares[j] += value * value;
                }
                if (y - radius - 1 >= 0) {
                    const uint value = input[(size_t)(y - radius - 1) * width + x];
                    columnSum[j] -= value;
                    columnSquares[j] -= value * value;
                }
            }
            localSum[s] = columnSum[j];
            localSquares[s] = columnSquares[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int first = id * LOCAL_THRESHOLD_COLUMNS;
        uint sum = 0;
        uint squares = 0;
        for (int s = first; s <= first + 2 * radius; s++) {
            sum += localSum[s];
            squares += localSquares[s];
        }
        const int rows = min(height - 1, y + radius) - max(0, y - radius) + 1;
        for (int j = 0; j < LOCAL_THRESHOLD_COLUMNS; j++) {
            if (j > 0) {
                sum += localSum[first + j + 2 * radius] - localSum[first + j - 1];
                squares += localSquares[first + j + 2 * radius] - localSquares[first + j - 1];
            }
            const int x = stripStart + radius + first + j;
            if (x < width) {
                const long count = (long)(min(width - 1, x + radius) - max(0, x - radius) + 1) * rows;
                const size_t index = (size_t)y * width + x;
                output[index] = localThresholdValue(count, sum, squares, input[index], k, stddevRange, sauvola);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void niblack(__global const uchar* input, __global uchar* output, int width, int height, int radius,
                      float k, float stddevRange, int bandHeight, __local uint* localSum, __local uint* localSquares)
{
    localThresholdBand(input, output, width, height, radius, k, stddevRange, bandHeight, localSum, localSquares, false);
}

__kernel void sauvola(__global const uchar* input, __global uchar* output, int width, int height, int radius,
                      float k, float stddevRange, int bandHeight, __local uint* localSum, __local uint* localSquares)
{
    localThresholdBand(input, output, width, height, radius, k, stddevRange, bandHeight, localSum, localSquares, true);
}
