#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

Image medianFilter(const Image& image);
void medianFilterRows(const Image& image, Image& result, int firstRow, int endRow, bool sortingNetwork=false); // it doesnt need to return a result, it can modify it through a reference
void medianFilterParallel(const Image& image, Image& result, int threads, bool sortingNetwork=false);
unsigned char medianOf9(std::array<unsigned char, 9>& p);

enum class MedianSort { Standard, Network, Bitonic, Shared };

MedianSort parseMedianSort(const std::string& name);
std::string medianSortName(MedianSort sort);

struct MedianNetwork {
    enum class Operation : unsigned char { CompareSwap, Min, Max, Copy };
    struct Step {
        Operation operation;
        int a;
        int b;
    };

    MedianSort sort = MedianSort::Network;
    int radius = 1;
    int strip = 1;
    int inputs = 0;
    int wires = 0;
    std::vector<int> constants;
    std::vector<Step> steps;
    std::vector<Step> layeredSteps;
    std::vector<int> outputs;
    int nominalComparators = 0;

    int size() const { return 2 * radius + 1; }
    int minMaxOperations() const;
    std::string openclDefinitions() const;
};

MedianNetwork buildMedianNetwork(MedianSort sort, int radius, int strip = 1);

void medianFilterParallel(const Image& image, Image& result, int threads, MedianSort sort, int radius, int strip = 1);
void medianFilterParallel(const Image& image, Image& result, int threads, const MedianNetwork& network);

class MedianFilterGPU {
public:
    explicit MedianFilterGPU(const Image& image, bool sortingNetwork = false);
    MedianFilterGPU(const Image& image, MedianSort sort, int radius, int strip = 1);
    ~MedianFilterGPU();
    MedianFilterGPU(const MedianFilterGPU&) = delete;
    MedianFilterGPU& operator=(const MedianFilterGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
    void createBuffers(const char* kernelName);
    void release();
    OpenCL cl;
    cl_kernel kernel = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    std::size_t bufferSize;
    int width;
    int height;
    std::size_t globalWorkSize[2];
};
