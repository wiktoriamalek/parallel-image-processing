#define BINS 256
#define FIXED_POINT_SHIFT 30
#define BLOCK 16

int plateauEnd(__local const ulong* histogram, int start)
{
    int end = start;
    while (end < BINS && histogram[end + 1] == histogram[start]) {
        end++;
    }
    return end;
}

bool isPeakStart(__local const ulong* histogram, int i)
{
    return histogram[i] > histogram[i - 1] && histogram[plateauEnd(histogram, i) + 1] < histogram[i];
}

ulong smoothedValue(__local const ulong* histogram, int i)
{
    return (histogram[i - 1] + 2 * histogram[i] + histogram[i + 1]) >> 2;
}

int countPeaks(__local const ulong* histogram, __local int* peakCounts, int* counter)
{
    const int id = get_local_id(0);
    int peaks = 0;
    for (int i = id + 1; i <= BINS; i += get_local_size(0)) {
        peaks += isPeakStart(histogram, i);
    }
    if (peaks > 0) {
        atomic_add(&peakCounts[*counter], peaks);
    }
    const int nextCounter = *counter == 2 ? 0 : *counter + 1;
    if (id == 0) {
        peakCounts[nextCounter] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int count = peakCounts[*counter];
    *counter = nextCounter;
    return count;
}

__kernel void peaksValleysThresholdsKernel(__global const unsigned int* histogram, int maxClasses,
                                           __global int* result, __global uchar* lookup)
{
    __local ulong smoothed[2][BINS + 2];
    __local ulong blockStart[BINS + 2];
    __local int peakCounts[3];
    __local int valleys[BINS];
    __local int valleyCount;
    const int id = get_local_id(0);
    const int size = get_local_size(0);

    for (int i = id + 1; i <= BINS; i += size) {
        smoothed[0][i] = (ulong)histogram[i - 1] << FIXED_POINT_SHIFT;
    }
    if (id == 0) {
        smoothed[0][0] = 0;
        smoothed[0][BINS + 1] = 0;
        smoothed[1][0] = 0;
        smoothed[1][BINS + 1] = 0;
        peakCounts[0] = 0;
        valleyCount = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int current = 0;
    int counter = 0;
    int iterations = 0;
    while (countPeaks(smoothed[current], peakCounts, &counter) > maxClasses) {
        for (int i = id + 1; i <= BINS; i += size) {
            blockStart[i] = smoothed[current][i];
        }
        for (int step = 0; step < BLOCK; step++) {
            for (int i = id + 1; i <= BINS; i += size) {
                smoothed[1 - current][i] = smoothedValue(smoothed[current], i);
            }
            current = 1 - current;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        iterations += BLOCK;
    }

    if (iterations > 0) {
        for (int i = id + 1; i <= BINS; i += size) {
            smoothed[current][i] = blockStart[i];
        }
        iterations -= BLOCK;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (;;) {
            for (int i = id + 1; i <= BINS; i += size) {
                smoothed[1 - current][i] = smoothedValue(smoothed[current], i);
            }
            if (countPeaks(smoothed[current], peakCounts, &counter) <= maxClasses) {
                break;
            }
            current = 1 - current;
            iterations++;
        }
    }

    __local const ulong* h = smoothed[current];
    for (int i = id + 1; i <= BINS; i += size) {
        if (h[i] < h[i - 1]) {
            const int end = plateauEnd(h, i);
            if (h[end + 1] > h[i]) {
                valleys[atomic_inc(&valleyCount)] = (i + end) / 2 - 1;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int count = valleyCount;
    for (int v = id; v < count; v += size) {
        int rank = 0;
        for (int j = 0; j < count; j++) {
            rank += valleys[j] < valleys[v];
        }
        result[2 + rank] = valleys[v];
    }
    if (id == 0) {
        result[0] = iterations;
        result[1] = count;
    }
    for (int level = id; level < BINS; level += size) {
        int k = 0;
        for (int j = 0; j < count; j++) {
            k += level > valleys[j];
        }
        lookup[level] = (uchar)(255 * k / max(1, count));
    }
}
