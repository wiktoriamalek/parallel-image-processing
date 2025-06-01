__kernel void invertRGBA(__global uchar4* img, const uint pixels)
{
    uint gid = get_global_id(0);
    
    if(gid >= pixels) return;
    
    uchar4 pix = img[gid];
    pix.x = 255 - pix.x;
    pix.y = 255 - pix.y;
    pix.z = 255 - pix.z;
    img[gid] = pix;
}
