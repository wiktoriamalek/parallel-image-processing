
__kernel void invertRGBA(__global uchar4* img, const uint pixels)
{
    uint gid = get_global_id(0);
    if (gid >= pixels) return;

    uchar4 pix = img[gid];
    pix.x = (uchar)(255 - (int)pix.x);
    pix.y = (uchar)(255 - (int)pix.y);
    pix.z = (uchar)(255 - (int)pix.z);

    // if (gid == 0) printf("pix0: %u %u %u\n", pix.x, pix.y, pix.z);

    img[gid] = pix;
}
