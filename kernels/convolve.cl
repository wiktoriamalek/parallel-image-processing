__kernel void threshold_kernel(__global unsigned char* pixels, int threshold)
    size_t i = get_global_id(0);
    if (pixels[i] < threshold) {
        pixels[i]=0;
     }
    else
    {
        pixels[i] = 255;
    }
}