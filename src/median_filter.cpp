#include "median_filter.hpp"
#include <array>
#include <algorithm>
#include "parallel_utils.hpp"

Image medianFilter(const Image& image) {
    Image result = image;
    medianFilterRows(image, result, 1, image.height-1);
    return result;
}
// x = columns, y = rows
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow) {
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
            std::nth_element(neighbours.begin(), neighbours.begin() + 4, neighbours.end());
            std::size_t resultIndex = y * image.width + x;
            result.pixels[resultIndex] = neighbours[4];
        }

    }
}

void medianFilterParallel(const Image& image, Image& result, int threads) {
    parallelRanges(std::max(0, image.height - 2), threads,
    [&](std::size_t begin, std::size_t end, int) {
        medianFilterRows(image, result, static_cast<int>(begin) + 1, static_cast<int>(end) + 1);
    });

}
