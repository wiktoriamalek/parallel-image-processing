#include "common_benchmarking.hpp"
#include <filesystem>
#include <fstream>
#include <utility>

BenchmarkOptions::BenchmarkOptions(int argc, char** argv, std::string defaultImage,
                                   std::string defaultOutput, std::initializer_list<std::string> extraNames)
    : image(std::move(defaultImage)), output(std::move(defaultOutput)) {
    for (int i = 1; i < argc; i += 2) {
        std::string name = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + name);
        std::string value = argv[i + 1];
        if (name == "--image") image = value;
        else if (name == "--output") output = value;
        else if (name == "--threads") threads = parseNumber<int>(value);
        else if (name == "--repeats") repeats = parseNumber<int>(value);
        else if (std::find(extraNames.begin(), extraNames.end(), name) != extraNames.end()) parameters[name] = value;
        else throw std::invalid_argument("Unknown option: " + name);
    }
    if (threads <= 0 || repeats <= 0) throw std::invalid_argument("Threads and repeats must be positive");
    if (image.empty() || output.empty()) throw std::invalid_argument("Image and output paths must not be empty");
}

std::string BenchmarkOptions::parameter(const std::string& name, const std::string& fallback) const {
    auto found = parameters.find(name);
    return found == parameters.end() ? fallback : found->second;
}

void appendCsv(const std::string& path, const std::string& header, const std::string& row) {
    std::filesystem::path output(path);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    bool empty = !std::filesystem::exists(output) || std::filesystem::file_size(output) == 0;
    if (!empty) {
        std::ifstream existing(output);
        std::string currentHeader;
        if (!std::getline(existing, currentHeader)) throw std::runtime_error("Could not read " + path);
        if (!currentHeader.empty() && currentHeader.back() == '\r') currentHeader.pop_back();
        if (currentHeader != header) throw std::runtime_error("Unexpected CSV header in " + path);
    }
    std::ofstream file(output, std::ios::app);
    if (!file) throw std::runtime_error("Could not open " + path);
    if (empty) file << header << '\n';
    file << row << '\n';
    file.close();
    if (!file) throw std::runtime_error("Could not write " + path);
}
