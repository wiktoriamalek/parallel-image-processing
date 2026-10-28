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