#include "bernsen.hpp"
#include "parallel_utils.hpp"
#include <algorithm>

void bernsenRows(const Image &image, Image &result, int radius, int minContrast, int firstRow, int endRow) {
    for (int y=firstRow; y<endRow; y++) {
        for (int x=0; x<image.width; x++) {
            int left = std::max(x- radius, 0);
            int right = std::min(image.width-1, x+radius);
            int top = std::max(y-radius, 0);
            int bottom = std::min(image.height-1,y+radius);
            int Jmin  = 255; // because we're going to be searching by std::min and update the value
            int Jmax  = 0;
            for (int ny=top; ny<=bottom; ny++) {
                for (int nx=left; nx<=right; nx++) {
                    std::size_t index = ny*image.width + nx;
                    int value = image.pixels[index];
                    Jmin = std::min(Jmin, value);
                    Jmax = std::max(Jmax, value);
                }
            }
            int contrast = Jmax - Jmin;
            if (contrast < minContrast) {
                result.pixels[y*image.width + x] = 0;
            }
            else {
                int threshold = (Jmax + Jmin)/2;
                int pixel = image.pixels[y*image.width + x];
                //result.pixels[y*image.width + x] = pixel < threshold ? 0 : 255; // Andrea Latina speaking....
                if (pixel < threshold) {
                    result.pixels[y*image.width + x] = 0;
                }

                else {
                    result.pixels[y*image.width + x] = 255;
                }
            }
        }
    }
}

void bernsenThreshold(const Image &image, Image &result, int radius, int minContrast, int threads) {
    // we need to divide the rows for threads and for each range call bernsenRows

    parallelRanges(image.height, threads, [&](std::size_t begin, std::size_t end, int) {
        bernsenRows(image, result, radius, minContrast, static_cast<int>(begin), static_cast<int>(end));
    });
}
