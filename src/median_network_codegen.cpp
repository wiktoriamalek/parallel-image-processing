#include "median_filter.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: median_network_codegen <output file> [blocks]" << std::endl;
        return 1;
    }
    const int blocks = argc == 3 ? std::stoi(argv[2]) : 2;
    std::ostringstream functions;
    std::ostringstream table;
    int index = 0;
    for (MedianSort sort : {MedianSort::Network, MedianSort::Bitonic, MedianSort::Shared}) {
        for (int radius = 1; radius <= 3; ++radius) {
            for (int strip : {1, 2, 4, 8, 16, 32}) {
                if (sort != MedianSort::Shared && strip != 1) continue;
                const MedianNetwork network = buildMedianNetwork(sort, radius, strip);
                const std::string name = "compiledMedian" + std::to_string(index++);
                functions << network.cpuFunction(name, blocks) << '\n';
                table << "    {" << network.fingerprint() << "ull, " << name << "},\n";
            }
        }
    }
    std::ofstream output(argv[1]);
    output << "constexpr int compiledMedianBlocks = " << blocks << ";\n" << functions.str() << "const CompiledMedianEntry compiledMedianNetworks[] = {\n" << table.str() << "};\n";
    if (!output) {
        std::cerr << "Could not write " << argv[1] << std::endl;
        return 1;
    }
    return 0;
}
