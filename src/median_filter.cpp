#include "median_filter.hpp"
#include <array>
#include <algorithm>
#include "parallel_utils.hpp"

Image medianFilter(const Image& image) {
    Image result = image;
    medianFilterRows(image, result, 1, image.height-1);
    return result;
}

void compareSwap(unsigned char& a, unsigned char& b) {
    if (a > b) std::swap(a, b);
}

unsigned char medianOf9(std::array<unsigned char, 9>& p) {
    compareSwap(p[1], p[2]);
    compareSwap(p[4], p[5]);
    compareSwap(p[7], p[8]);

    compareSwap(p[0], p[1]);
    compareSwap(p[3], p[4]);
    compareSwap(p[6], p[7]);

    compareSwap(p[1], p[2]);
    compareSwap(p[4], p[5]);
    compareSwap(p[7], p[8]);

    compareSwap(p[0], p[3]);
    compareSwap(p[5], p[8]);
    compareSwap(p[4], p[7]);
    compareSwap(p[3], p[6]);

    compareSwap(p[1], p[4]);
    compareSwap(p[2], p[5]);
    compareSwap(p[4], p[7]);

    compareSwap(p[4], p[2]);
    compareSwap(p[6], p[4]);
    compareSwap(p[4], p[2]);

    return p[4];
}

// x = columns, y = rows
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow, bool sortingNetwork) {
    for (int y = firstRow; y < endRow; y++) {
        for (int x = 1; x < image.width - 1; x++) {
            std::array<unsigned char, 9> neighbours;
            int index = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int neighbourX = x + kx;
                    int neighbourY = y + ky;
                    std::size_t neighbourIndex = neighbourY * image.width + neighbourX;
                    neighbours[index] = image.pixels[neighbourIndex];
                    index++;
                }
            }
            if (sortingNetwork) {
                medianOf9(neighbours);
            }
            else {
                std::nth_element(neighbours.begin(), neighbours.begin() + 4, neighbours.end());
            }
            std::size_t resultIndex = y * image.width + x;
            result.pixels[resultIndex] = neighbours[4];
        }
    }
}

void medianFilterParallel(const Image& image, Image& result, int threads, bool sortingNetwork) {
    parallelRanges(std::max(0, image.height - 2), threads,
    [&](std::size_t begin, std::size_t end, int) {
        medianFilterRows(image, result, static_cast<int>(begin) + 1, static_cast<int>(end) + 1, sortingNetwork);
    });

}
