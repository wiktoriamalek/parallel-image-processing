#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T>
double median(std::vector<T> values) {
    if (values.empty()) throw std::runtime_error("No timing samples");
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n % 2) return static_cast<double>(values[n / 2]);
    return (static_cast<double>(values[n / 2 - 1]) + static_cast<double>(values[n / 2])) / 2.0;
}

template<class Operation, class Before, class After>
double measure(Operation operation, int repeats, Before before, After after) {
    if (repeats <= 0) throw std::invalid_argument("Repeat count must be positive");
    std::vector<double> times;
    times.reserve(repeats);
    for (int repeat = -5; repeat < repeats; ++repeat) {
        before();
        auto start = std::chrono::steady_clock::now();
        operation();
        auto end = std::chrono::steady_clock::now();
        after(repeat);
        if (repeat >= 0) times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    return median(times);
}

template<class Operation>
double measure(Operation operation, int repeats = 100) {
    return measure(operation, repeats, [] {}, [](int) {});
}

template<class T>
T parseNumber(const std::string& text) {
    std::istringstream input(text);
    T value;
    if (!(input >> value)) throw std::invalid_argument("Invalid number: " + text);
    input >> std::ws;
    if (!input.eof() || !std::isfinite(static_cast<double>(value))) {
        throw std::invalid_argument("Invalid number: " + text);
    }
    return value;
}

struct BenchmarkOptions {
    std::string image, output;
    int threads = 2;
    int repeats = 100;

    BenchmarkOptions(int argc, char** argv, std::string defaultImage, std::string defaultOutput,
                     std::initializer_list<std::string> extraNames = {});
    std::string parameter(const std::string& name, const std::string& fallback) const;

private:
    std::map<std::string, std::string> parameters;
};

void appendCsv(const std::string& path, const std::string& header, const std::string& row);
