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

kernel void histogramKernel(
    device const unsigned char* input [[buffer(0)]],
    device unsigned int* histogram [[buffer(1)]],
    constant int& Npixels [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    int i = gridPosition.x; 

    if (i >= Npixels){
        return;
    }
    unsigned char pixel = input[i];
    atomic_inc(&histogram[pixel]);
}

kernel void thresholdKernel(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& Npixels [[buffer(2)]],
    device const int* threshold [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    int i = gridPosition.x;
    if (i >= Npixels) {return;}

    if (input[i] <= threshold[0])
    {
        output[i] = 0;
    }

    else
    {
        output[i] = 255;
    }
}

kernel void histogramLocalKernel(
    device const unsigned char* input [[buffer(0)]],
    device unsigned int* histogram [[buffer(1)]],
    constant int& Npixels [[buffer(2)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    threadgroup unsigned int localHistogram[256]; 
    int bin = localPosition.x; 
    localHistogram[bin] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t i = gridPosition.x; i < (size_t)Npixels; i += gridSize.x)
    {
        unsigned char intensity = input[i];
        atomic_inc(&localHistogram[intensity]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_add(&histogram[bin], localHistogram[bin]);
}

kernel void clearHistogramKernel(
    device unsigned int* histogram [[buffer(0)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    histogram[gridPosition.x] = 0;
}

kernel void otsuThresholdKernel(
    device const unsigned int* histogram [[buffer(0)]],
    device int* threshold [[buffer(1)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    int t = gridPosition.x;
    threadgroup float variances[256];
    ulong totalPixels = 0;
    ulong totalSum = 0;
    ulong backgroundPixels = 0;
    ulong backgroundSum = 0;

    for (int i = 0; i < 256; i++)
    {
        ulong count = histogram[i];
        totalPixels += count;
        totalSum += i * count;
        if (i<=t)
        {
            backgroundPixels += count;
            backgroundSum += i * count;
        }
    }
    ulong foregroundPixels = totalPixels - backgroundPixels;
    ulong foregroundSum = totalSum - backgroundSum;
    variances[t]=0;
    if (backgroundPixels>0 && foregroundPixels>0)
    {
        float backgroundAverage = (float)backgroundSum/backgroundPixels;
        float foregroundAverage = (float)foregroundSum/foregroundPixels;
        float backgroundWeight = (float)backgroundPixels/totalPixels;
        float foregroundWeight = (float)foregroundPixels/totalPixels;

        float difference = backgroundAverage - foregroundAverage;
        variances[t] = backgroundWeight * foregroundWeight * difference * difference;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t==0)
    {
        float maxVariance = 0;
        int bestThreshold = 0;
        for (int i = 0; i < 256; i++)
        {
            if (variances[i] > maxVariance)
            {
                maxVariance = variances[i];
                bestThreshold = i;
            }
        }
    threshold[0] = bestThreshold;

    }
}

kernel void multiOtsuApplyLookupKernel(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant int& pixelCount [[buffer(2)]],
    device const uchar* lookup [[buffer(3)]],
    uint3 gridPosition [[thread_position_in_grid]]
)
{
    const size_t i = gridPosition.x;
    if (i >= (size_t)pixelCount) {
        return;
    }

    output[i] = lookup[input[i]];
}

float multiOtsuGain(ulong count, long centeredSum, float totalPixels)
{
    if (count == 0) {
        return -INFINITY;
    }
    const float deviation = (float)centeredSum / (float)count;
    return (float)count / totalPixels * deviation * deviation;
}

kernel void multiOtsuPrefixKernel(
    device const unsigned int* histogram [[buffer(0)]],
    device ulong* prefixCount [[buffer(1)]],
    device long* prefixCenteredSum [[buffer(2)]],
    device float* layers [[buffer(3)]],
    uint3 localPosition [[thread_position_in_threadgroup]]
)
{
    threadgroup unsigned int localHistogram[256];
    threadgroup ulong totals[2];
    const int id = localPosition.x;
    const int end = id + 1;
    localHistogram[id] = histogram[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong count = 0;
    ulong sum = 0;
    for (int i = 0; i < end; i++) {
        count += localHistogram[i];
        sum += (ulong)i * localHistogram[i];
    }
    if (end == 256) {
        totals[0] = count;
        totals[1] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong totalCount = totals[0];
    const ulong center = (totals[1] + totalCount / 2) / totalCount;
    const long centeredSum = (long)sum - (long)(center * count);
    prefixCount[end] = count;
    prefixCenteredSum[end] = centeredSum;
    layers[end] = multiOtsuGain(count, centeredSum, (float)totalCount);
    if (id == 0) {
        prefixCount[0] = 0;
        prefixCenteredSum[0] = 0;
        layers[0] = -INFINITY;
    }
}

kernel void multiOtsuLayerKernel(
    device const ulong* prefixCount [[buffer(0)]],
    device const long* prefixCenteredSum [[buffer(1)]],
    device float* layers [[buffer(2)]],
    device ushort* split [[buffer(3)]],
    constant int& k [[buffer(4)]],
    constant int& firstEnd [[buffer(5)]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]]
)
{
    threadgroup float values[256];
    threadgroup int begins[256];
    const int begin = localPosition.x;
    const int end = firstEnd + groupPosition.x;
    device const float* previous = layers + (k - 2) * 257;

    float value = -INFINITY;
    if (begin >= k - 1 && begin < end) {
        value = previous[begin] + multiOtsuGain(prefixCount[end] - prefixCount[begin],
                                                prefixCenteredSum[end] - prefixCenteredSum[begin],
                                                (float)prefixCount[256]);
    }
    values[begin] = value;
    begins[begin] = begin;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int stride = 128; stride > 0; stride /= 2) {
        if (begin < stride) {
            const float other = values[begin + stride];
            const int otherBegin = begins[begin + stride];
            if (other > values[begin] || (other == values[begin] && otherBegin < begins[begin])) {
                values[begin] = other;
                begins[begin] = otherBegin;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (begin == 0) {
        layers[(k - 1) * 257 + end] = values[0];
        split[(k - 2) * 256 + end - 1] = (ushort)begins[0];
    }
}

kernel void multiOtsuLookupTableKernel(
    device const float* layers [[buffer(0)]],
    device const ushort* split [[buffer(1)]],
    constant int& classes [[buffer(2)]],
    device int* thresholds [[buffer(3)]],
    device uchar* lookup [[buffer(4)]],
    uint3 localPosition [[thread_position_in_threadgroup]]
)
{
    threadgroup int localThresholds[255];
    const int id = localPosition.x;
    if (id == 0) {
        const int valid = layers[(classes - 1) * 257 + 256] != -INFINITY;
        int last = 256;
        for (int k = classes; k >= 2; k--) {
            const int begin = valid ? split[(k - 2) * 256 + last - 1] : 0;
            localThresholds[k - 2] = begin - 1;
            thresholds[k - 2] = begin - 1;
            last = begin;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int k = 0;
    while (k < classes - 1 && id > localThresholds[k]) {
        k++;
    }
    lookup[id] = (uchar)(255 * k / (classes - 1));
}
