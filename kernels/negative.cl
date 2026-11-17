__kernel void negative_gpu(__global const unsigned char* input, __global unsigned char* output)
{
    size_t x = get_global_id(0);
    unsigned char intensity = input[x];
    output[x]=255-intensity;
}