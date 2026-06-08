
#include "cpu_ops.hpp"
#include <algorithm>
#include <thread>
#include <cmath>
#include <array>
#include <cstring>

void invertCPU(unsigned char* img, std::size_t pixels) {
    for (std::size_t i = 0; i < pixels * 4; i += 4) {
        img[i] = 255 - img[i];     // R
        img[i+1] = 255 - img[i+1]; // G
        img[i+2] = 255 - img[i+2]; // B
        // alpha ta sama
    }
}

void boxBlurCPU(const unsigned char* src, unsigned char* dst, int w, int h) {
    auto idx = [w](int x, int y) { return (y * w + x) * 4; };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int sum[4] = {0}, count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = std::clamp(y + dy, 0, h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = std::clamp(x + dx, 0, w - 1);
                    const unsigned char* p = src + idx(xx, yy);
                    for (int c = 0; c < 4; ++c)
                        sum[c] += p[c];
                    ++count;
                }
            }
            unsigned char* o = dst + idx(x, y);
            for (int c = 0; c < 4; ++c)
                o[c] = sum[c] / count;
        }
    }
}

void sobelCPU(const unsigned char* src, unsigned char* dst, int w, int h) {
    auto lum = [](const unsigned char* p) {
        return (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
    };
    auto idx = [w](int x, int y) { return (y * w + x) * 4; };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int L[3][3];
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = std::clamp(y + dy, 0, h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = std::clamp(x + dx, 0, w - 1);
                    L[dy+1][dx+1] = lum(src + idx(xx, yy));
                }
            }
            int gx = L[0][2] + 2*L[1][2] + L[2][2] - L[0][0] - 2*L[1][0] - L[2][0];
            int gy = L[2][0] + 2*L[2][1] + L[2][2] - L[0][0] - 2*L[0][1] - L[0][2];
            unsigned char val = static_cast<unsigned char>((std::abs(gx) + std::abs(gy)) >> 2);
            unsigned char* out = dst + idx(x, y);
            out[0] = out[1] = out[2] = val;
            out[3] = 255;
        }
    }
}

void boxBlurCPUMT(const unsigned char* src, unsigned char* dst, int w, int h, int threads) {
    std::vector<std::thread> workers;
    int rows = (h + threads - 1) / threads;

    for (int t = 0; t < threads; ++t) {
        int y0 = t * rows;
        int y1 = std::min(h, y0 + rows);
        workers.emplace_back([=]() {
            boxBlurCPU(src + y0 * w * 4, dst + y0 * w * 4, w, y1 - y0);
        });
    }
    for (auto& th : workers) th.join();
}

void sobelCPUMT(const unsigned char* src, unsigned char* dst, int w, int h, int threads) {
    std::vector<std::thread> pool;
    int rows = (h + threads - 1) / threads;

    for (int t = 0; t < threads; ++t) {
        int y0 = t * rows;
        int y1 = std::min(h, y0 + rows);
        pool.emplace_back([=]() {
            sobelCPU(src + y0 * w * 4, dst + y0 * w * 4, w, y1 - y0);
        });
    }
    for (auto& th : pool) th.join();
}

void histCPU(const unsigned char* img, std::size_t pix, uint64_t H[3][256]) {
    std::memset(H, 0, sizeof(uint64_t) * 3 * 256);
    for (std::size_t i = 0; i < pix; ++i) {
        const unsigned char* p = img + i * 4;
        ++H[0][p[0]];
        ++H[1][p[1]];
        ++H[2][p[2]];
    }
}

void histCPUMT(const unsigned char* img, std::size_t pix, uint64_t H[3][256], int threads) { //wersja wielowatkowa
    std::vector<std::array<uint64_t, 256 * 3>> locals(threads);
    std::memset(H, 0, sizeof(uint64_t) * 3 * 256);

    auto worker = [&](int tid) {
        std::fill(locals[tid].begin(), locals[tid].end(), 0);
        std::size_t chunk = (pix + threads - 1) / threads;
        std::size_t start = tid * chunk;
        std::size_t end = std::min(pix, start + chunk);
        for (std::size_t i = start; i < end; ++i) {
            const unsigned char* p = img + i * 4;
            ++locals[tid][0 * 256 + p[0]];
            ++locals[tid][1 * 256 + p[1]];
            ++locals[tid][2 * 256 + p[2]];
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    for (int t = 0; t < threads; ++t) {
        for (int c = 0; c < 3; ++c) {
            for (int b = 0; b < 256; ++b) {
                H[c][b] += locals[t][c * 256 + b];
            }
        }
    }
}
