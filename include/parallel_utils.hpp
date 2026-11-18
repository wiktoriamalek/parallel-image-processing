#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

enum class Schedule { Static, Dynamic, Guided };

struct ParallelSettings {
    Schedule schedule = Schedule::Static;
    int chunksPerThread = 8;
    std::vector<double>* busyMicroseconds = nullptr;
};

inline ParallelSettings& parallelSettings() {
    static ParallelSettings settings;
    return settings;
}

class ScopedParallelSettings {
public:
    explicit ScopedParallelSettings(const ParallelSettings& settings) : previous(parallelSettings()) {
        parallelSettings() = settings;
    }
    ~ScopedParallelSettings() { parallelSettings() = previous; }
    ScopedParallelSettings(const ScopedParallelSettings&) = delete;
    ScopedParallelSettings& operator=(const ScopedParallelSettings&) = delete;

private:
    ParallelSettings previous;
};

template<class Function>
void parallelRanges(std::size_t count, int threadCount, Function operation) {
    if (threadCount <= 0) throw std::invalid_argument("Thread count must be positive");
    if (count == 0) return;
    const ParallelSettings settings = parallelSettings();
    if (settings.chunksPerThread <= 0) throw std::invalid_argument("Chunks per thread must be positive");
    if (settings.busyMicroseconds) settings.busyMicroseconds->assign(threadCount, 0.0);
    auto timed = [&](std::size_t begin, std::size_t end, int thread) {
        if (!settings.busyMicroseconds) {
            operation(begin, end, thread);
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        operation(begin, end, thread);
        (*settings.busyMicroseconds)[thread] +=
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    };
    if (threadCount == 1) {
        timed(0, count, 0);
        return;
    }

    const auto workers = static_cast<std::size_t>(threadCount);
    const std::size_t block = count / workers;
    const std::size_t remainder = count % workers;
    const std::size_t chunk = std::max<std::size_t>(1, count / (workers * settings.chunksPerThread));
    std::atomic<std::size_t> next{0};
    auto work = [&](int i) {
        if (settings.schedule == Schedule::Static) {
            std::size_t begin = block * i + std::min<std::size_t>(i, remainder);
            std::size_t end = begin + block + (static_cast<std::size_t>(i) < remainder);
            timed(begin, end, i);
            return;
        }
        for (;;) {
            std::size_t begin = next.load();
            std::size_t size = chunk;
            if (settings.schedule == Schedule::Dynamic) {
                begin = next.fetch_add(chunk);
            } else {
                do {
                    if (begin >= count) return;
                    size = std::max(chunk, (count - begin) / (2 * workers));
                } while (!next.compare_exchange_weak(begin, begin + size));
            }
            if (begin >= count) return;
            timed(begin, std::min(begin + size, count), i);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(threadCount);
    try {
        // let's create threads
        for (int i = 0; i < threadCount; ++i) {
            threads.emplace_back([&work, i] { work(i); });
        }
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join(); // waiting for all of them to finish
}
