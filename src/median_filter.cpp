#include "median_filter.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <map>
#include <sstream>
#include <stdexcept>
#include "parallel_utils.hpp"

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

#if defined(__ARM_NEON)
using MedianVector = uint8x16_t;
inline MedianVector medianLoad(const unsigned char* source) { return vld1q_u8(source); }
inline void medianStore(unsigned char* target, MedianVector value) { vst1q_u8(target, value); }
inline MedianVector medianMin(MedianVector a, MedianVector b) { return vminq_u8(a, b); }
inline MedianVector medianMax(MedianVector a, MedianVector b) { return vmaxq_u8(a, b); }
inline MedianVector medianFull() { return vdupq_n_u8(255); }
#else
struct MedianVector {
    unsigned char lane[medianLanes];
};
inline MedianVector medianLoad(const unsigned char* source) {
    MedianVector value;
    std::memcpy(value.lane, source, medianLanes);
    return value;
}
inline void medianStore(unsigned char* target, const MedianVector& value) { std::memcpy(target, value.lane, medianLanes); }
inline MedianVector medianMin(const MedianVector& a, const MedianVector& b) {
    MedianVector value;
    for (int i = 0; i < medianLanes; ++i) value.lane[i] = std::min(a.lane[i], b.lane[i]);
    return value;
}
inline MedianVector medianMax(const MedianVector& a, const MedianVector& b) {
    MedianVector value;
    for (int i = 0; i < medianLanes; ++i) value.lane[i] = std::max(a.lane[i], b.lane[i]);
    return value;
}
inline MedianVector medianFull() {
    MedianVector value;
    std::memset(value.lane, 255, medianLanes);
    return value;
}
#endif

inline void medianCompareSwap(MedianVector& a, MedianVector& b) {
    const MedianVector low = medianMin(a, b);
    b = medianMax(a, b);
    a = low;
}

using CompiledMedian = void (*)(const unsigned char* const* rows, unsigned char* const* results);

struct CompiledMedianEntry {
    std::uint64_t fingerprint;
    CompiledMedian function;
};

#ifdef MEDIAN_COMPILED_NETWORKS
#include "median_networks_generated.inc"
#else
constexpr int compiledMedianBlocks = 1;
const CompiledMedianEntry compiledMedianNetworks[] = {{0, nullptr}};
#endif
constexpr int compiledMedianWidth = compiledMedianBlocks * medianLanes;

CompiledMedian findCompiledMedian(const MedianNetwork& network) {
    const std::uint64_t fingerprint = network.fingerprint();
    for (const CompiledMedianEntry& entry : compiledMedianNetworks) {
        if (entry.function && entry.fingerprint == fingerprint) return entry.function;
    }
    return nullptr;
}

void executeSteps(const std::vector<MedianNetwork::Step>& steps, std::vector<Lanes>& v) {
    for (const MedianNetwork::Step& step : steps) {
        unsigned char* a = v[step.a].data();
        unsigned char* b = v[step.b].data();
        switch (step.operation) {
            case Operation::CompareSwap: {
                const MedianVector first = medianLoad(a);
                const MedianVector second = medianLoad(b);
                medianStore(a, medianMin(first, second));
                medianStore(b, medianMax(first, second));
                break;
            }
            case Operation::Min:
                medianStore(a, medianMin(medianLoad(a), medianLoad(b)));
                break;
            case Operation::Max:
                medianStore(b, medianMax(medianLoad(a), medianLoad(b)));
                break;
            case Operation::Copy:
                std::memcpy(a, b, medianLanes);
                break;
        }
    }
}

bool isImpulse(unsigned char pixel) {
    return pixel == 0 || pixel == 255;
}

inline void medianSwitch(const unsigned char* pixels, const unsigned char* median, unsigned char* target) {
#if defined(__ARM_NEON)
    const uint8x16_t pixel = vld1q_u8(pixels);
    const uint8x16_t impulse = vorrq_u8(vceqq_u8(pixel, vdupq_n_u8(0)), vceqq_u8(pixel, vdupq_n_u8(255)));
    vst1q_u8(target, vbslq_u8(impulse, vld1q_u8(median), pixel));
#else
    for (int i = 0; i < medianLanes; ++i) target[i] = isImpulse(pixels[i]) ? median[i] : pixels[i];
#endif
}

void medianCompiledStrips(const Image& image, Image& result, const MedianNetwork& network, CompiledMedian function,
                          std::size_t firstStrip, std::size_t endStrip, bool switching) {
    const int radius = network.radius;
    const int rows = network.strip + 2 * radius;
    const int lastColumn = image.width - radius;
    std::vector<const unsigned char*> sources(static_cast<std::size_t>(rows));
    std::vector<unsigned char*> targets(static_cast<std::size_t>(network.strip));
    std::vector<unsigned char> medians(static_cast<std::size_t>(network.strip) * compiledMedianWidth);
    std::array<unsigned char, compiledMedianWidth> unused{};
    for (std::size_t strip = firstStrip; strip < endStrip; ++strip) {
        const int y0 = radius + static_cast<int>(strip) * network.strip;
        for (int start = radius; start < lastColumn; start += compiledMedianWidth) {
            const int x = std::max(radius, std::min(start, lastColumn - compiledMedianWidth));
            for (int row = 0; row < rows; ++row) {
                const int y = std::min(y0 - radius + row, image.height - 1);
                sources[row] = image.pixels.data() + static_cast<std::size_t>(y) * image.width + x - radius;
            }
            for (int j = 0; j < network.strip; ++j) {
                if (y0 + j >= image.height - radius) targets[j] = unused.data();
                else if (switching) targets[j] = medians.data() + static_cast<std::size_t>(j) * compiledMedianWidth;
                else targets[j] = result.pixels.data() + static_cast<std::size_t>(y0 + j) * image.width + x;
            }
            function(sources.data(), targets.data());
            if (!switching) continue;
            for (int j = 0; j < network.strip && y0 + j < image.height - radius; ++j) {
                const std::size_t offset = static_cast<std::size_t>(y0 + j) * image.width + x;
                const unsigned char* pixels = image.pixels.data() + offset;
                const unsigned char* median = medians.data() + static_cast<std::size_t>(j) * compiledMedianWidth;
                unsigned char* target = result.pixels.data() + offset;
                for (int i = 0; i < compiledMedianWidth; i += medianLanes) medianSwitch(pixels + i, median + i, target + i);
            }
        }
    }
}

void medianNetworkStrips(const Image& image, Image& result, const MedianNetwork& network,
                         std::size_t firstStrip, std::size_t endStrip, bool switching) {
    const int radius = network.radius;
    const int size = network.size();
    const int rows = network.strip + 2 * radius;
    const int lastColumn = image.width - radius;
    std::vector<Lanes> v(static_cast<std::size_t>(network.wires));
    for (std::size_t strip = firstStrip; strip < endStrip; ++strip) {
        const int y0 = radius + static_cast<int>(strip) * network.strip;
        for (int start = radius; start < lastColumn; start += medianLanes) {
            const int x = std::max(radius, std::min(start, lastColumn - medianLanes));
            const int lanes = std::min(medianLanes, lastColumn - x);
            for (int wire : network.constants) v[wire].fill(255);
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
                const std::size_t offset = static_cast<std::size_t>(y0 + j) * image.width + x;
                const Lanes& median = v[network.outputs[j]];
                unsigned char* target = result.pixels.data() + offset;
                if (!switching) {
                    std::memcpy(target, median.data(), static_cast<std::size_t>(lanes));
                    continue;
                }
                const unsigned char* pixels = image.pixels.data() + offset;
                for (int i = 0; i < lanes; ++i) target[i] = isImpulse(pixels[i]) ? median[i] : pixels[i];
            }
        }
    }
}

void medianStandardRows(const Image& image, Image& result, int radius, int firstRow, int endRow, bool switching) {
    const int size = 2 * radius + 1;
    std::vector<unsigned char> window(static_cast<std::size_t>(size) * size);
    const auto middle = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
    for (int y = firstRow; y < endRow; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            const std::size_t pixelIndex = static_cast<std::size_t>(y) * image.width + x;
            if (switching && !isImpulse(image.pixels[pixelIndex])) {
                result.pixels[pixelIndex] = image.pixels[pixelIndex];
                continue;
            }
            std::size_t index = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                const unsigned char* row = image.pixels.data() + static_cast<std::size_t>(y + ky) * image.width + x;
                for (int kx = -radius; kx <= radius; ++kx) window[index++] = row[kx];
            }
            std::nth_element(window.begin(), middle, window.end());
            result.pixels[pixelIndex] = *middle;
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

std::string MedianNetwork::cpuFunction(const std::string& name, int blocks) const {
    std::ostringstream source;
    auto variable = [](int block, int wire) { return "v" + std::to_string(block) + "_" + std::to_string(wire); };
    source << "void " << name << "(const unsigned char* const* rows, unsigned char* const* results) {\n";
    for (int block = 0; block < blocks; ++block) {
        source << "    MedianVector";
        for (int wire = 0; wire < wires; ++wire) source << (wire ? ", " : " ") << variable(block, wire);
        source << ";\n";
    }
    std::vector<char> ready(static_cast<std::size_t>(wires), 0);
    std::vector<char> constant(static_cast<std::size_t>(wires), 0);
    for (int wire : constants) constant[wire] = 1;
    auto prepare = [&](int wire) {
        if (ready[wire]) return;
        ready[wire] = 1;
        for (int block = 0; block < blocks; ++block) {
            if (wire < inputs) {
                source << "    " << variable(block, wire) << " = medianLoad(rows[" << wire / size() << "] + "
                       << wire % size() + 16 * block << ");\n";
            } else if (constant[wire]) {
                source << "    " << variable(block, wire) << " = medianFull();\n";
            }
        }
    };
    for (const Step& step : steps) {
        prepare(step.b);
        if (step.operation != Operation::Copy) prepare(step.a);
        ready[step.a] = 1;
        for (int block = 0; block < blocks; ++block) {
            const std::string a = variable(block, step.a);
            const std::string b = variable(block, step.b);
            switch (step.operation) {
                case Operation::CompareSwap: source << "    medianCompareSwap(" << a << ", " << b << ");\n"; break;
                case Operation::Min: source << "    " << a << " = medianMin(" << a << ", " << b << ");\n"; break;
                case Operation::Max: source << "    " << b << " = medianMax(" << a << ", " << b << ");\n"; break;
                case Operation::Copy: source << "    " << a << " = " << b << ";\n"; break;
            }
        }
    }
    for (std::size_t j = 0; j < outputs.size(); ++j) {
        prepare(outputs[j]);
        for (int block = 0; block < blocks; ++block) {
            source << "    medianStore(results[" << j << "] + " << 16 * block << ", " << variable(block, outputs[j]) << ");\n";
        }
    }
    source << "}\n";
    return source.str();
}

std::uint64_t MedianNetwork::fingerprint() const {
    std::uint64_t value = 1469598103934665603ull;
    auto add = [&](long long number) {
        value ^= static_cast<std::uint64_t>(number);
        value *= 1099511628211ull;
    };
    add(radius);
    add(strip);
    add(inputs);
    add(wires);
    add(static_cast<long long>(constants.size()));
    for (int wire : constants) add(wire);
    add(static_cast<long long>(steps.size()));
    for (const Step& step : steps) {
        add(static_cast<int>(step.operation));
        add(step.a);
        add(step.b);
    }
    add(static_cast<long long>(outputs.size()));
    for (int wire : outputs) add(wire);
    return value;
}

MedianNetwork buildMedianNetwork(MedianSort sort, int radius, int strip) {
    if (sort == MedianSort::Standard) throw std::invalid_argument("The standard median does not use a network");
    if (radius < 1) throw std::invalid_argument("Median radius must be at least 1");
    if (strip < 1) throw std::invalid_argument("Strip must be at least 1");
    if (sort != MedianSort::Shared && strip != 1) throw std::invalid_argument("Only the shared median processes a strip of rows");
    MedianNetwork network;
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

void medianFilterParallel(const Image& image, Image& result, int threads, MedianSort sort, int radius, int strip,
                          bool switching) {
    if (sort != MedianSort::Standard) {
        medianFilterParallel(image, result, threads, buildMedianNetwork(sort, radius, strip), switching);
        return;
    }
    if (radius < 1) throw std::invalid_argument("Median radius must be at least 1");
    if (strip != 1) throw std::invalid_argument("Only the shared median processes a strip of rows");
    if (result.pixels.size() != image.pixels.size()) throw std::invalid_argument("Result must have the size of the image");
    parallelRanges(std::max(0, image.height - 2 * radius), threads, [&](std::size_t begin, std::size_t end, int) {
        medianStandardRows(image, result, radius, static_cast<int>(begin) + radius, static_cast<int>(end) + radius, switching);
    });
}

void medianFilterParallel(const Image& image, Image& result, int threads, const MedianNetwork& network, bool switching) {
    if (result.pixels.size() != image.pixels.size()) throw std::invalid_argument("Result must have the size of the image");
    const int interiorRows = image.height - 2 * network.radius;
    if (image.width <= 2 * network.radius || interiorRows <= 0) return;
    const std::size_t strips = static_cast<std::size_t>((interiorRows + network.strip - 1) / network.strip);
    const CompiledMedian compiled = image.width - 2 * network.radius >= compiledMedianWidth ? findCompiledMedian(network) : nullptr;
    parallelRanges(strips, threads, [&](std::size_t begin, std::size_t end, int) {
        if (compiled) medianCompiledStrips(image, result, network, compiled, begin, end, switching);
        else medianNetworkStrips(image, result, network, begin, end, switching);
    });
}

MedianFilterGPU::MedianFilterGPU(const Image& image, MedianSort sort, int radius, int strip, bool switching)
    : cl("kernels/median_filter.cl", medianProgramPrefix(sort, radius, strip)),
      bufferSize(image.pixels.size()),
      width(image.width),
      height(image.height),
      globalWorkSize{static_cast<std::size_t>(image.width),
                     static_cast<std::size_t>((image.height + strip - 1) / strip)} {
    const cl_int replaceOnlyImpulses = switching;
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer input");
        outputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer output");
        kernel = clCreateKernel(cl.program, sort == MedianSort::Standard ? "median_insertion" : "median_network", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(cl_int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(cl_int), &height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(cl_int), &replaceOnlyImpulses), "clSetKernelArg");
    } catch (...) {
        release();
        throw;
    }
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
    enqueueTiles(cl.queue, kernel, globalWorkSize[0], globalWorkSize[1], kernelEvent); // this function enqueues the execution of a kernel on an OpenCL device
    checkCL(clEnqueueReadBuffer(cl.queue, outputBuffer, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
