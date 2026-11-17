__kernel void histogramKernel(__global const unsigned char* input, __global unsigned int* histogram, int Npixels)
{
    int i = get_global_id(0); // it replaces the loop for int i in range something - each pixel gets its own work-item

    if (i >= Npixels){
        return;
    }
    unsigned char pixel = input[i];
    atomic_inc(&histogram[pixel]);
}
// threshold points at int numbers that are in global GPU memory, and kernel can read them
__kernel void thresholdKernel(__global const unsigned char* input, __global unsigned char* output, int Npixels, __global const int* threshold) // global means data is in global GPU memory and each work-item can access it
{
    int i = get_global_id(0);
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

__kernel void histogramLocalKernel(__global const unsigned char* input, __global unsigned int* histogram, int Npixels)
{
    __local unsigned int localHistogram[256]; // local indicates a memory shared with all work items of one working group, each group has its own localHistogram
    int bin = get_local_id(0); // tells about the thread number in a local group
    localHistogram[bin] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // counting pixels to a local histogram

    for (int i = get_global_id(0); i < Npixels; i+=get_global_size(0))
    {
        unsigned char intensity = input[i];
        atomic_inc(&localHistogram[intensity]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&histogram[bin], localHistogram[bin]);
}

// zeroes the histogram before every run on the GPU itself; clEnqueueFillBuffer would be simpler,
// but on Apple M3 Pro (OpenCL 1.2) it writes garbage instead of zeros
__kernel void clearHistogramKernel(__global unsigned int* histogram)
{
    histogram[get_global_id(0)] = 0;
}

// kernel only reads, so const         // kernel writes; without const
__kernel void otsuThresholdKernel(__global const unsigned int* histogram, __global int* threshold)
{
    int t = get_global_id(0);
    __local float variances[256];
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
    barrier(CLK_LOCAL_MEM_FENCE);

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

__kernel void multiOtsuApplyLookupKernel(__global const uchar* input, __global uchar* output, int pixelCount, __global const uchar* lookup)
{
    const size_t i = get_global_id(0);
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

__kernel void multiOtsuPrefixKernel(__global const unsigned int* histogram, __global ulong* prefixCount,
                                    __global long* prefixCenteredSum, __global float* layers)
{
    __local unsigned int localHistogram[256];
    __local ulong totals[2];
    const int id = get_local_id(0);
    const int end = id + 1;
    localHistogram[id] = histogram[id];
    barrier(CLK_LOCAL_MEM_FENCE);

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
    barrier(CLK_LOCAL_MEM_FENCE);

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

__kernel void multiOtsuLayerKernel(__global const ulong* prefixCount, __global const long* prefixCenteredSum,
                                   __global float* layers, __global ushort* split, int k, int firstEnd)
{
    __local float values[256];
    __local int begins[256];
    const int begin = get_local_id(0);
    const int end = firstEnd + get_group_id(0);
    __global const float* previous = layers + (k - 2) * 257;

    float value = -INFINITY;
    if (begin >= k - 1 && begin < end) {
        value = previous[begin] + multiOtsuGain(prefixCount[end] - prefixCount[begin],
                                                prefixCenteredSum[end] - prefixCenteredSum[begin],
                                                (float)prefixCount[256]);
    }
    values[begin] = value;
    begins[begin] = begin;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 128; stride > 0; stride /= 2) {
        if (begin < stride) {
            const float other = values[begin + stride];
            const int otherBegin = begins[begin + stride];
            if (other > values[begin] || (other == values[begin] && otherBegin < begins[begin])) {
                values[begin] = other;
                begins[begin] = otherBegin;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (begin == 0) {
        layers[(k - 1) * 257 + end] = values[0];
        split[(k - 2) * 256 + end - 1] = (ushort)begins[0];
    }
}

__kernel void multiOtsuLookupTableKernel(__global const float* layers, __global const ushort* split, int classes,
                                         __global int* thresholds, __global uchar* lookup)
{
    __local int localThresholds[255];
    const int id = get_local_id(0);
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
    barrier(CLK_LOCAL_MEM_FENCE);

    int k = 0;
    while (k < classes - 1 && id > localThresholds[k]) {
        k++;
    }
    lookup[id] = (uchar)(255 * k / (classes - 1));
}
