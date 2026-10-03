#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

template<class Function>
void parallelRanges(std::size_t count, int threadCount, Function operation) {
    if (threadCount <= 0) throw std::invalid_argument("Thread count must be positive");
    if (count == 0) return;
    if (threadCount == 1) {
        operation(0, count, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(threadCount);
    std::size_t block = count / threadCount;
    std::size_t remainder = count % threadCount;
    try {
        // let's create threads
        for (int i = 0; i < threadCount; ++i) {
            std::size_t begin = block * i + std::min<std::size_t>(i, remainder);
            std::size_t end = begin + block + (static_cast<std::size_t>(i) < remainder);
            threads.emplace_back([&, begin, end, i] { operation(begin, end, i); });
        }
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join(); // waiting for all of them to finish
}
