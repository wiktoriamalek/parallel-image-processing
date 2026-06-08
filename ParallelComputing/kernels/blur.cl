__kernel void boxBlurRGBA(__global uchar4* in, __global uchar4* out, const uint W, const uint H) //kazdy piksel to 4 bajty
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= W || y >= H) return;

    uint4 sum = (uint4)(0);
    uint count = 0;

    for (int dy = -1; dy <= 1; ++dy) //czy nie wychodza poza obraz
    {
        int yy = clamp(y + dy, 0, (int)H - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = clamp(x + dx, 0, (int)W - 1); //srednia 3x3
            sum += convert_uint4(in[yy * W + xx]);
            count++;
        }
    }

    if (count == 0) count = 1;

    out[y * W + x] = convert_uchar4(sum / count); //srednia kolorow
}
