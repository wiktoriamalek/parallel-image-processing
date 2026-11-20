#pragma once
#include "image.hpp"
#include "opencl_utils.hpp"
#include <cstddef>
#include <string>
#include <vector>

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

void medianFilterParallel(const Image& image, Image& result, int threads, MedianSort sort, int radius, int strip = 1,
                          bool switching = false);
void medianFilterParallel(const Image& image, Image& result, int threads, const MedianNetwork& network,
                          bool switching = false);

class MedianFilterGPU {
public:
    MedianFilterGPU(const Image& image, MedianSort sort, int radius, int strip = 1, bool switching = false);
    ~MedianFilterGPU();
    MedianFilterGPU(const MedianFilterGPU&) = delete;
    MedianFilterGPU& operator=(const MedianFilterGPU&) = delete;
    void run(const Image& image, Image& output, cl_event* kernelEvent = nullptr);

private:
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
