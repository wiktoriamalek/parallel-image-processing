__kernel void contrast_gpu(__global const unsigned char* input, __global unsigned char* output, float factor)
{
    size_t x = get_global_id(0);
    float scaled = (input[x] - 128) * factor;
    float value = scaled + 128.0f;
    output[x] = (unsigned char)round(clamp(value, 0.0f, 255.0f));
}
