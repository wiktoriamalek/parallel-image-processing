__kernel void bernsen(__global const unsigned char* input, __global unsigned char* output, int width, int height, int radius, int minContrast)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    int left = max(x- radius, 0);
    int right = min(width-1, x+radius);
    int top = max(y-radius, 0);
    int bottom = min(height-1,y+radius);
    int Jmin  = 255; // because we're going to be searching by min and update the value
    int Jmax  = 0;
    for (int ny=top; ny<=bottom; ny++) {
        for (int nx=left; nx<=right; nx++) {
            size_t index = ny*width + nx;
            int value = input[index];
            Jmin = min(Jmin, value);
            Jmax = max(Jmax, value);
        }
    }
    int contrast = Jmax - Jmin;
    if (contrast < minContrast) {
        output[y*width + x] = 0;
    }
    else {
        int threshold = (Jmax + Jmin)/2;
        int pixel = input[y*width + x];
        if (pixel < threshold) {
           output[y*width + x] = 0;
        }

        else {
            output[y*width + x] = 255;
        }
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
