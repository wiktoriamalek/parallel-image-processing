__kernel void histRGB(__global const uchar4* img, const uint pixels,
                      __global uint* R, __global uint* G, __global uint* B)
{
    uint gid = get_global_id(0); //numer watku
    uint stride = get_global_size(0); //liczba watkow

    for (uint i = gid; i < pixels; i += stride)
    {
        uchar4 p = img[i];

        atomic_add(&R[p.x], 1u); //zapobieganie race condition
        atomic_add(&G[p.y], 1u);
        atomic_add(&B[p.z], 1u);

//        if (gid == 0 && i < 3)
   //         printf("pix[%u] = (%u,%u,%u)\n", i, p.x, p.y, p.z);
    }
}
