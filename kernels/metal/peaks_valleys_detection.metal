#include <metal_stdlib>

using namespace metal;

inline uint atomic_inc(device uint* target) {
    return atomic_fetch_add_explicit((device atomic_uint*)target, (uint)1, memory_order_relaxed);
}

inline uint atomic_add(device uint* target, uint value) {
    return atomic_fetch_add_explicit((device atomic_uint*)target, value, memory_order_relaxed);
}

inline uint atomic_inc(threadgroup uint* target) {
    return atomic_fetch_add_explicit((threadgroup atomic_uint*)target, (uint)1, memory_order_relaxed);
}

inline uint atomic_add(threadgroup uint* target, uint value) {
    return atomic_fetch_add_explicit((threadgroup atomic_uint*)target, value, memory_order_relaxed);
}

inline int atomic_inc(threadgroup int* target) {
    return atomic_fetch_add_explicit((threadgroup atomic_int*)target, (int)1, memory_order_relaxed);
}

inline int atomic_add(threadgroup int* target, int value) {
    return atomic_fetch_add_explicit((threadgroup atomic_int*)target, value, memory_order_relaxed);
}

#define BINS 256
#define FIXED_POINT_SHIFT 30
#define BLOCK 16

int plateauEnd(threadgroup const ulong* histogram, int start)
{
    int end = start;
    while (end < BINS && histogram[end + 1] == histogram[start]) {
        end++;
    }
    return end;
}

bool isPeakStart(threadgroup const ulong* histogram, int i)
{
    return histogram[i] > histogram[i - 1] && histogram[plateauEnd(histogram, i) + 1] < histogram[i];
}

ulong smoothedValue(threadgroup const ulong* histogram, int i)
{
    return (histogram[i - 1] + 2 * histogram[i] + histogram[i + 1]) >> 2;
}

int countPeaks(threadgroup const ulong* histogram, threadgroup int* peakCounts, thread int* counter, uint3 localPosition, uint3 groupSize)
{
    const int id = localPosition.x;
    int peaks = 0;
    for (int i = id + 1; i <= BINS; i += groupSize.x) {
        peaks += isPeakStart(histogram, i);
    }
    if (peaks > 0) {
        atomic_add(&peakCounts[*counter], peaks);
    }
    const int nextCounter = *counter == 2 ? 0 : *counter + 1;
    if (id == 0) {
        peakCounts[nextCounter] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int count = peakCounts[*counter];
    *counter = nextCounter;
    return count;
}

kernel void peaksValleysThresholdsKernel(
    device const unsigned int* histogram [[buffer(0)]],
    constant int& maxClasses [[buffer(1)]],
    device int* result [[buffer(2)]],
    device uchar* lookup [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    threadgroup ulong smoothed[2][BINS + 2];
    threadgroup ulong blockStart[BINS + 2];
    threadgroup int peakCounts[3];
    threadgroup int valleys[BINS];
    threadgroup int valleyCount;
    const int id = localPosition.x;
    const int size = groupSize.x;

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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int current = 0;
    int counter = 0;
    int iterations = 0;
    while (countPeaks(smoothed[current], peakCounts, &counter, localPosition, groupSize) > maxClasses) {
        for (int i = id + 1; i <= BINS; i += size) {
            blockStart[i] = smoothed[current][i];
        }
        for (int step = 0; step < BLOCK; step++) {
            for (int i = id + 1; i <= BINS; i += size) {
                smoothed[1 - current][i] = smoothedValue(smoothed[current], i);
            }
            current = 1 - current;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        iterations += BLOCK;
    }

    if (iterations > 0) {
        for (int i = id + 1; i <= BINS; i += size) {
            smoothed[current][i] = blockStart[i];
        }
        iterations -= BLOCK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (;;) {
            for (int i = id + 1; i <= BINS; i += size) {
                smoothed[1 - current][i] = smoothedValue(smoothed[current], i);
            }
            if (countPeaks(smoothed[current], peakCounts, &counter, localPosition, groupSize) <= maxClasses) {
                break;
            }
            current = 1 - current;
            iterations++;
        }
    }

    threadgroup const ulong* h = smoothed[current];
    for (int i = id + 1; i <= BINS; i += size) {
        if (h[i] < h[i - 1]) {
            const int end = plateauEnd(h, i);
            if (h[end + 1] > h[i]) {
                valleys[atomic_inc(&valleyCount)] = (i + end) / 2 - 1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
