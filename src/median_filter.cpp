#include "median_filter.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
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

namespace {
using Operation = MedianNetwork::Operation;

std::vector<int> everySecond(const std::vector<int>& wires, std::size_t start) {
    std::vector<int> part;
    for (std::size_t i = start; i < wires.size(); i += 2) part.push_back(wires[i]);
    return part;
}

class NetworkBuilder {
public:
    explicit NetworkBuilder(MedianNetwork& network) : network(network) {}

    void compareSwap(int low, int high) {
        network.steps.push_back({Operation::CompareSwap, low, high});
        ++network.nominalComparators;
    }

    int copy(int source) {
        const int wire = network.wires++;
        network.steps.push_back({Operation::Copy, wire, source});
        return wire;
    }

    std::vector<int> copy(const std::vector<int>& sources) {
        std::vector<int> wires;
        for (int source : sources) wires.push_back(copy(source));
        return wires;
    }

    std::vector<int> merge(const std::vector<int>& first, const std::vector<int>& second) {
        if (first.empty()) return second;
        if (second.empty()) return first;
        if (first.size() == 1 && second.size() == 1) {
            compareSwap(first[0], second[0]);
            return {first[0], second[0]};
        }
        const std::vector<int> even = merge(everySecond(first, 0), everySecond(second, 0));
        const std::vector<int> odd = merge(everySecond(first, 1), everySecond(second, 1));
        std::vector<int> merged{even[0]};
        for (std::size_t i = 0; i < odd.size() || i + 1 < even.size(); ++i) {
            if (i < odd.size() && i + 1 < even.size()) {
                compareSwap(odd[i], even[i + 1]);
                merged.push_back(odd[i]);
                merged.push_back(even[i + 1]);
            } else if (i < odd.size()) {
                merged.push_back(odd[i]);
            } else {
                merged.push_back(even[i + 1]);
            }
        }
        return merged;
    }

    std::vector<int> sort(const std::vector<int>& wires) {
        if (wires.size() <= 1) return wires;
        const auto half = wires.begin() + static_cast<std::ptrdiff_t>(wires.size() / 2);
        return merge(sort(std::vector<int>(wires.begin(), half)), sort(std::vector<int>(half, wires.end())));
    }

private:
    MedianNetwork& network;
};

struct Candidates {
    int count;
    int median;
};

std::vector<int> keepCandidates(const std::vector<int>& sorted, Candidates& candidates) {
    const int size = static_cast<int>(sorted.size());
    const int low = std::max(0, candidates.median - (candidates.count - size));
    const int high = std::min(size - 1, candidates.median);
    candidates.count -= low + (size - 1 - high);
    candidates.median -= low;
    return std::vector<int>(sorted.begin() + low, sorted.begin() + high + 1);
}

class SharedMedian {
public:
    explicit SharedMedian(MedianNetwork& network)
        : network(network), builder(network), size(network.size()),
          uses(static_cast<std::size_t>(network.strip + 2 * network.radius), 0) {}

    void build() {
        countUses(0, network.strip, 0, -1);
        solve(0, network.strip, {}, Candidates{size * size, size * size / 2}, 0, -1);
    }

private:
    std::vector<int> newRows(int first, int last, int coveredFirst, int coveredLast) const {
        std::vector<int> rows;
        for (int row = first; row <= last; ++row) {
            if (row < coveredFirst || row > coveredLast) rows.push_back(row);
        }
        return rows;
    }

    void countUses(int first, int end, int coveredFirst, int coveredLast) {
        const int commonFirst = end - 1;
        const int commonLast = first + 2 * network.radius;
        for (int row : newRows(commonFirst, commonLast, coveredFirst, coveredLast)) ++uses[row];
        if (end - first == 1) return;
        const int middle = (first + end) / 2;
        countUses(first, middle, commonFirst, commonLast);
        countUses(middle, end, commonFirst, commonLast);
    }

    void solve(int first, int end, std::vector<int> list, Candidates candidates, int coveredFirst, int coveredLast) {
        const int commonFirst = end - 1;
        const int commonLast = first + 2 * network.radius;
        list = addRows(list, candidates, newRows(commonFirst, commonLast, coveredFirst, coveredLast));
        if (end - first == 1) {
            network.outputs[first] = list[candidates.median];
            return;
        }
        const int middle = (first + end) / 2;
        solve(first, middle, builder.copy(list), candidates, commonFirst, commonLast);
        solve(middle, end, list, candidates, commonFirst, commonLast);
    }

    std::vector<int> sortedRow(int row) {
        auto found = sorted.find(row);
        if (found == sorted.end()) {
            std::vector<int> wires(static_cast<std::size_t>(size));
            for (int column = 0; column < size; ++column) wires[column] = row * size + column;
            found = sorted.emplace(row, builder.sort(wires)).first;
        }
        if (--uses[row] == 0) return found->second;
        return builder.copy(found->second);
    }

    std::vector<int> addRows(const std::vector<int>& list, Candidates& candidates, const std::vector<int>& rows) {
        std::vector<std::vector<int>> lists;
        for (int row : rows) lists.push_back(sortedRow(row));
        while (lists.size() > 1) {
            std::vector<std::vector<int>> merged;
            for (std::size_t i = 0; i + 1 < lists.size(); i += 2) {
                merged.push_back(keepCandidates(builder.merge(lists[i], lists[i + 1]), candidates));
            }
            if (lists.size() % 2) merged.push_back(lists.back());
            lists = std::move(merged);
        }
        if (lists.empty()) return list;
        return keepCandidates(builder.merge(list, lists.front()), candidates);
    }

    MedianNetwork& network;
    NetworkBuilder builder;
    int size;
    std::vector<int> uses;
    std::map<int, std::vector<int>> sorted;
};

void buildBitonic(MedianNetwork& network) {
    const int area = network.size() * network.size();
    int padded = 1;
    while (padded < area) padded *= 2;
    network.inputs = area;
    network.wires = padded;
    std::vector<int> position(static_cast<std::size_t>(padded));
    std::vector<char> constant(static_cast<std::size_t>(padded), 0);
    for (int wire = 0; wire < padded; ++wire) {
        position[wire] = wire;
        if (wire >= area) {
            constant[wire] = 1;
            network.constants.push_back(wire);
        }
    }
    for (int block = 2; block <= padded; block *= 2) {
        for (int distance = block / 2; distance > 0; distance /= 2) {
            for (int i = 0; i < padded; ++i) {
                const int partner = i ^ distance;
                if (partner <= i) continue;
                ++network.nominalComparators;
                const int low = (i & block) == 0 ? i : partner;
                const int high = (i & block) == 0 ? partner : i;
                if (constant[position[high]]) continue;
                if (constant[position[low]]) {
                    std::swap(position[low], position[high]);
                    continue;
                }
                network.steps.push_back({Operation::CompareSwap, position[low], position[high]});
            }
        }
    }
    network.outputs = {position[area / 2]};
}

void pruneToOutputs(MedianNetwork& network) {
    std::vector<char> needed(static_cast<std::size_t>(network.wires), 0);
    for (int wire : network.outputs) needed[wire] = 1;
    std::vector<MedianNetwork::Step> kept;
    for (auto step = network.steps.rbegin(); step != network.steps.rend(); ++step) {
        if (step->operation == Operation::Copy) {
            if (!needed[step->a]) continue;
            needed[step->a] = 0;
            needed[step->b] = 1;
            kept.push_back(*step);
            continue;
        }
        const bool low = needed[step->a];
        const bool high = needed[step->b];
        if (!low && !high) continue;
        kept.push_back({low && high ? Operation::CompareSwap : low ? Operation::Min : Operation::Max, step->a, step->b});
        needed[step->a] = 1;
        needed[step->b] = 1;
    }
    network.steps.assign(kept.rbegin(), kept.rend());
}

std::vector<MedianNetwork::Step> orderByLayers(const MedianNetwork& network) {
    std::vector<int> wireLayer(static_cast<std::size_t>(network.wires), 0);
    std::vector<std::pair<int, MedianNetwork::Step>> layered;
    layered.reserve(network.steps.size());
    for (const MedianNetwork::Step& step : network.steps) {
        const int layer = 1 + std::max(wireLayer[step.a], wireLayer[step.b]);
        wireLayer[step.a] = layer;
        wireLayer[step.b] = layer;
        layered.emplace_back(layer, step);
    }
    std::stable_sort(layered.begin(), layered.end(), [](const auto& first, const auto& second) {
        return first.first < second.first;
    });
    std::vector<MedianNetwork::Step> steps;
    steps.reserve(layered.size());
    for (const auto& entry : layered) steps.push_back(entry.second);
    return steps;
}

constexpr int medianLanes = 16;
using Lanes = std::array<unsigned char, medianLanes>;

void executeSteps(const std::vector<MedianNetwork::Step>& steps, std::vector<Lanes>& v) {
    for (const MedianNetwork::Step& step : steps) {
        Lanes& a = v[step.a];
        Lanes& b = v[step.b];
        const Lanes first = a;
        const Lanes second = b;
        if (step.operation == Operation::Copy) {
            a = second;
            continue;
        }
        for (int i = 0; i < medianLanes; ++i) {
            a[i] = std::min(first[i], second[i]);
            b[i] = std::max(first[i], second[i]);
        }
    }
}

void medianNetworkStrips(const Image& image, Image& result, const MedianNetwork& network,
                         std::size_t firstStrip, std::size_t endStrip) {
    const int radius = network.radius;
    const int size = network.size();
    const int rows = network.strip + 2 * radius;
    const int lastColumn = image.width - radius;
    std::vector<Lanes> v(static_cast<std::size_t>(network.wires));
    for (int wire : network.constants) v[wire].fill(255);
    for (std::size_t strip = firstStrip; strip < endStrip; ++strip) {
        const int y0 = radius + static_cast<int>(strip) * network.strip;
        for (int start = radius; start < lastColumn; start += medianLanes) {
            const int x = std::max(radius, std::min(start, lastColumn - medianLanes));
            const int lanes = std::min(medianLanes, lastColumn - x);
            for (int row = 0; row < rows; ++row) {
                const int y = std::min(y0 - radius + row, image.height - 1);
                const unsigned char* source = image.pixels.data() + static_cast<std::size_t>(y) * image.width + x - radius;
                for (int column = 0; column < size; ++column) {
                    Lanes& lane = v[row * size + column];
                    if (lanes == medianLanes) {
                        std::memcpy(lane.data(), source + column, medianLanes);
                    } else {
                        for (int i = 0; i < medianLanes; ++i) lane[i] = source[column + std::min(i, lanes - 1)];
                    }
                }
            }
            executeSteps(network.layeredSteps, v);
            for (int j = 0; j < network.strip && y0 + j < image.height - radius; ++j) {
                std::memcpy(result.pixels.data() + static_cast<std::size_t>(y0 + j) * image.width + x,
                            v[network.outputs[j]].data(), static_cast<std::size_t>(lanes));
            }
        }
    }
}

void medianStandardRows(const Image& image, Image& result, int radius, int firstRow, int endRow) {
    const int size = 2 * radius + 1;
    std::vector<unsigned char> window(static_cast<std::size_t>(size) * size);
    const auto middle = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
    for (int y = firstRow; y < endRow; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            std::size_t index = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                const unsigned char* row = image.pixels.data() + static_cast<std::size_t>(y + ky) * image.width + x;
                for (int kx = -radius; kx <= radius; ++kx) window[index++] = row[kx];
            }
            std::nth_element(window.begin(), middle, window.end());
            result.pixels[static_cast<std::size_t>(y) * image.width + x] = *middle;
        }
    }
}

std::string medianProgramPrefix(MedianSort sort, int radius, int strip) {
    if (sort != MedianSort::Standard) return buildMedianNetwork(sort, radius, strip).openclDefinitions();
    if (radius < 1) throw std::invalid_argument("Median radius must be at least 1");
    if (strip != 1) throw std::invalid_argument("Only the shared median processes a strip of rows");
    return "#define MEDIAN_RADIUS " + std::to_string(radius) + "\n";
}
}

MedianSort parseMedianSort(const std::string& name) {
    if (name == "standard") return MedianSort::Standard;
    if (name == "network") return MedianSort::Network;
    if (name == "bitonic") return MedianSort::Bitonic;
    if (name == "shared") return MedianSort::Shared;
    throw std::invalid_argument("Median sort must be standard, network, bitonic or shared");
}

std::string medianSortName(MedianSort sort) {
    switch (sort) {
        case MedianSort::Standard: return "standard";
        case MedianSort::Network: return "network";
        case MedianSort::Bitonic: return "bitonic";
        case MedianSort::Shared: return "shared";
    }
    return "unknown";
}

int MedianNetwork::minMaxOperations() const {
    int operations = 0;
    for (const Step& step : steps) {
        if (step.operation == Operation::CompareSwap) operations += 2;
        else if (step.operation != Operation::Copy) operations += 1;
    }
    return operations;
}

std::string MedianNetwork::openclDefinitions() const {
    std::ostringstream source;
    source << "#define MEDIAN_RADIUS " << radius << '\n'
           << "#define MEDIAN_STRIP " << strip << '\n'
           << "#define MEDIAN_WIRES " << wires << '\n'
           << "#define MEDIAN_NETWORK";
    std::vector<char> ready(static_cast<std::size_t>(wires), 0);
    std::vector<char> constant(static_cast<std::size_t>(wires), 0);
    for (int wire : constants) constant[wire] = 1;
    auto prepare = [&](int wire) {
        if (ready[wire]) return;
        ready[wire] = 1;
        if (wire < inputs) source << " LD(" << wire << ',' << wire / size() << ',' << wire % size() << ')';
        else if (constant[wire]) source << " v[" << wire << "]=255;";
    };
    for (const Step& step : steps) {
        prepare(step.b);
        if (step.operation != Operation::Copy) prepare(step.a);
        ready[step.a] = 1;
        const char* name = step.operation == Operation::CompareSwap ? "CS"
                         : step.operation == Operation::Min ? "LO"
                         : step.operation == Operation::Max ? "HI" : "CP";
        source << ' ' << name << '(' << step.a << ',' << step.b << ')';
    }
    source << "\n#define MEDIAN_OUTPUTS";
    for (std::size_t j = 0; j < outputs.size(); ++j) {
        prepare(outputs[j]);
        source << " result[" << j << "]=v[" << outputs[j] << "];";
    }
    source << '\n';
    return source.str();
}

MedianNetwork buildMedianNetwork(MedianSort sort, int radius, int strip) {
    if (sort == MedianSort::Standard) throw std::invalid_argument("The standard median does not use a network");
    if (radius < 1) throw std::invalid_argument("Median radius must be at least 1");
    if (strip < 1) throw std::invalid_argument("Strip must be at least 1");
    if (sort != MedianSort::Shared && strip != 1) throw std::invalid_argument("Only the shared median processes a strip of rows");
    MedianNetwork network;
    network.sort = sort;
    network.radius = radius;
    network.strip = strip;
    if (sort == MedianSort::Bitonic) {
        buildBitonic(network);
    } else {
        network.inputs = (strip + 2 * radius) * network.size();
        network.wires = network.inputs;
        network.outputs.assign(static_cast<std::size_t>(strip), -1);
        SharedMedian(network).build();
    }
    pruneToOutputs(network);
    network.layeredSteps = orderByLayers(network);
    return network;
}

void medianFilterParallel(const Image& image, Image& result, int threads, MedianSort sort, int radius, int strip) {
    if (sort != MedianSort::Standard) {
        medianFilterParallel(image, result, threads, buildMedianNetwork(sort, radius, strip));
        return;
    }
    if (radius < 1) throw std::invalid_argument("Median radius must be at least 1");
    if (strip != 1) throw std::invalid_argument("Only the shared median processes a strip of rows");
    if (result.pixels.size() != image.pixels.size()) throw std::invalid_argument("Result must have the size of the image");
    parallelRanges(std::max(0, image.height - 2 * radius), threads, [&](std::size_t begin, std::size_t end, int) {
        medianStandardRows(image, result, radius, static_cast<int>(begin) + radius, static_cast<int>(end) + radius);
    });
}

void medianFilterParallel(const Image& image, Image& result, int threads, const MedianNetwork& network) {
    if (result.pixels.size() != image.pixels.size()) throw std::invalid_argument("Result must have the size of the image");
    const int interiorRows = image.height - 2 * network.radius;
    if (image.width <= 2 * network.radius || interiorRows <= 0) return;
    const std::size_t strips = static_cast<std::size_t>((interiorRows + network.strip - 1) / network.strip);
    parallelRanges(strips, threads, [&](std::size_t begin, std::size_t end, int) {
        medianNetworkStrips(image, result, network, begin, end);
    });
}

MedianFilterGPU::MedianFilterGPU(const Image& image, bool sortingNetwork)
    : cl("kernels/median_filter.cl"),
      bufferSize(image.pixels.size()),
      width(image.width),
      height(image.height),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)} {
    cl_int network = sortingNetwork;
    try {
        createBuffers("median_filter3x3");
        checkCL(clSetKernelArg(kernel, 4, sizeof(cl_int), &network), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
}

MedianFilterGPU::MedianFilterGPU(const Image& image, MedianSort sort, int radius, int strip)
    : cl("kernels/median_filter.cl", medianProgramPrefix(sort, radius, strip)),
      bufferSize(image.pixels.size()),
      width(image.width),
      height(image.height),
      globalWorkSize{static_cast<std::size_t>(image.width),
                     static_cast<std::size_t>((image.height + strip - 1) / strip)} {
    try {
        createBuffers(sort == MedianSort::Standard ? "median_insertion" : "median_network");
    } catch (...) {
        release();
        throw;
    }
}

void MedianFilterGPU::createBuffers(const char* kernelName) {
    cl_int error;
    inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
    checkCL(error, "clCreateBuffer input");
    outputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
    checkCL(error, "clCreateBuffer output");
    kernel = clCreateKernel(cl.program, kernelName, &error);
    checkCL(error, "clCreateKernel");
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 2, sizeof(cl_int), &width), "clSetKernelArg");
    checkCL(clSetKernelArg(kernel, 3, sizeof(cl_int), &height), "clSetKernelArg");
}

void MedianFilterGPU::release() {
    if (kernel) clReleaseKernel(kernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (outputBuffer) clReleaseMemObject(outputBuffer);
}

MedianFilterGPU::~MedianFilterGPU() {
    release();
}

void MedianFilterGPU::run(const Image& image, Image& output, cl_event* kernelEvent) {
    if (image.width != width || image.height != height ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    // image.pixels.data() is a data address on CPU
    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer"); // 0 means we dont wait for other events
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, kernelEvent), "clEnqueueNDRangeKernel"); // this function enqueues the execution of a kernel on an OpenCL device
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
