__constant int GX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 }; //splaszczone
__constant int GY[9] = {  1, 2, 1,  0, 0, 0, -1, -2, -1 };

uchar computeLum(uchar4 p)
{
    return (uchar)((77 * p.x + 150 * p.y + 29 * p.z) >> 8); //proporcje rgb do luminacji wzgledem ludzkiego oka
}

__kernel void sobelRGBA(__global uchar4* in, __global uchar4* out, const uint W, const uint H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= W || y >= H) return;

    int gx = 0, gy = 0, k = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = clamp(y + dy, 0, (int)H - 1);
        for (int dx = -1; dx <= 1; ++dx, ++k)
        {
            int xx = clamp(x + dx, 0, (int)W - 1);
            uchar L = computeLum(in[yy * W + xx]); //pobieranie luminacji 3x3 sasiadujacych pikseli
            gx += GX[k] * L; //wagi algorytmu sobela
            gy += GY[k] * L;
        }
    }

    uchar m = (uchar)((abs(gx) + abs(gy)) >> 2);
    uchar alpha = 200; 
    out[y * W + x] = (uchar4)(m, m, m, alpha);
}
