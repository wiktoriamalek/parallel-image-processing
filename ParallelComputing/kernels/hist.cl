__kernel void histRGB(__global const uchar4* img,const uint pixels,__global uint* R,__global uint* G,__global uint* B)
{
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);

    for(uint i = gid; i < pixels; i += stride)
    {
        uchar4 p = img[i];
        atomic_inc(&R[p.x]);
        atomic_inc(&G[p.y]);
        atomic_inc(&B[p.z]);
    }
}
