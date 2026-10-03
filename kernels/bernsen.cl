__kernel void bernsen(__global const unsigned char* input, __global unsigned char* output, int width, int height, int radius, int minContrast)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    int left = max(x- radius, 0);
    int right = min(width-1, x+radius);
    int top = max(y-radius, 0);
    int bottom = min(height-1,y+radius);
    int Jmin  = 255; // because we're going to be searching by min and update the value
    int Jmax  = 0;
    for (int ny=top; ny<=bottom; ny++) {
        for (int nx=left; nx<=right; nx++) {
            size_t index = ny*width + nx;
            int value = input[index];
            Jmin = min(Jmin, value);
            Jmax = max(Jmax, value);
        }
    }
    int contrast = Jmax - Jmin;
    if (contrast < minContrast) {
        output[y*width + x] = 0;
    }
    else {
        int threshold = (Jmax + Jmin)/2;
        int pixel = input[y*width + x];
        if (pixel < threshold) {
           output[y*width + x] = 0;
        }

        else {
            output[y*width + x] = 255;
        }
    }
}