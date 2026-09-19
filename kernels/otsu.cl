__kernel void histogramKernel(__global const unsigned char* input, __global unsigned int* histogram, int Npixels)
{
    int i = get_global_id(0); // it replaces the loop for int i in range something - each pixel gets its own work-item

    if (i >= Npixels){
        return;
}
    unsigned char pixel = input[i];
    atomic_inc(&histogram[pixel]);


}