__kernel void histogramKernel(__global const unsigned char* input, __global unsigned int* histogram, int Npixels)
{
    int i = get_global_id(0); // it replaces the loop for int i in range something - each pixel gets its own work-item

    if (i >= Npixels){
        return;
}
    unsigned char pixel = input[i];
    atomic_inc(&histogram[pixel]);
}

__kernel void thresholdKernel(__global const unsigned char* input, __global unsigned char* output, int Npixels, int threshold)
{
    int i = get_global_id(0);
    if (i >= Npixels) {return;}

    if (input[i] <= threshold)
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