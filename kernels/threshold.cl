__kernel void threshold_kernel(__global unsigned char* pixels, int threshold) // kernel function can be run by the opencl on GPU
{                              // pixels points to data in a global memory that can by accessed by GPU work-items
    size_t i = get_global_id(0);
    if (pixels[i] < threshold) {
        pixels[i]=0;
     }
    else
    {
        pixels[i] = 255;
    }
}