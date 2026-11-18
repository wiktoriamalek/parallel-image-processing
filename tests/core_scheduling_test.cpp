#include "bernsen.hpp"
#include "common_benchmarking.hpp"
#include "contrast.hpp"
#include "image_io.hpp"
#include "median_filter.hpp"
#include "parallel_utils.hpp"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace {
struct Algorithm {
    std::string name;
    std::function<void(const Image&, Image&, int)> run;
};

const std::vector<Algorithm>& algorithms() {
    static const std::vector<Algorithm> all = {
        {"bernsen", [](const Image& image, Image& output, int threads) { bernsenThreshold(image, output, 7, 15, threads); }},
        {"niblack", [](const Image& image, Image& output, int threads) { niblackThreshold(image, output, 7, 0.2, threads); }},
        {"median_filter", [](const Image& image, Image& output, int threads) { medianFilterParallel(image, output, threads); }},
        {"contrast", [](const Image& image, Image& output, int threads) { contrastParallel(image, output, threads, 2.0f); }},
    };
    return all;
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> parts;
    std::istringstream input(text);
    for (std::string part; std::getline(input, part, ',');) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

const char* scheduleName(Schedule schedule) {
    switch (schedule) {
        case Schedule::Static: return "static";
        case Schedule::Dynamic: return "dynamic";
        case Schedule::Guided: return "guided";
    }
    return "?";
}

int systemNumber(const char* name) {
#ifdef __APPLE__
    int value = 0;
    std::size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) return value;
#else
    (void)name;
#endif
    return -1;
}

std::string imageColumns(const BenchmarkOptions& options, const Image& image) {
    std::ostringstream text;
    text << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height;
    return text.str();
}

std::vector<double> measureThroughput(const Algorithm& algorithm, const Image& image, const Image& expected,
                                      int maxInstances, const BenchmarkOptions& options, const std::string& output) {
    std::vector<double> times;
    for (int instances = 1; instances <= maxInstances; ++instances) {
        std::vector<Image> outputs(instances, image);
        const double time = measure([&] {
            std::vector<std::thread> threads;
            for (int i = 0; i < instances; ++i) threads.emplace_back([&, i] { algorithm.run(image, outputs[i], 1); });
            for (auto& thread : threads) thread.join();
        }, options.repeats);
        for (const Image& result : outputs) {
            if (result.pixels != expected.pixels) throw std::runtime_error(algorithm.name + ": concurrent copies differ");
        }
        times.push_back(time);
        std::ostringstream row;
        row << std::setprecision(12) << algorithm.name << ',' << imageColumns(options, image) << ',' << instances << ','
            << options.repeats << ',' << time;
        appendCsv(output, "algorithm,image,width,height,instances,repeats,time_us", row.str());
    }
    return times;
}

double measureSchedule(const Algorithm& algorithm, const Image& image, const Image& expected, int threads,
                       ParallelSettings settings, const BenchmarkOptions& options) {
    Image result = image;
    double time = 0.0;
    {
        ScopedParallelSettings scope(settings);
        time = measure([&] { algorithm.run(image, result, threads); }, options.repeats);
    }
    if (result.pixels != expected.pixels) {
        throw std::runtime_error(algorithm.name + ": " + scheduleName(settings.schedule) + " with " + std::to_string(threads) +
                                 " threads differs from 1 thread");
    }
    std::vector<double> busy;
    settings.busyMicroseconds = &busy;
    {
        ScopedParallelSettings scope(settings);
        algorithm.run(image, result, threads);
    }
    std::ostringstream row;
    row << std::setprecision(12) << algorithm.name << ',' << imageColumns(options, image) << ',' << threads << ','
        << scheduleName(settings.schedule) << ',' << settings.chunksPerThread << ',' << options.repeats << ',' << time << ','
        << *std::min_element(busy.begin(), busy.end()) << ','
        << std::accumulate(busy.begin(), busy.end(), 0.0) / busy.size() << ','
        << *std::max_element(busy.begin(), busy.end());
    appendCsv(options.output, "algorithm,image,width,height,threads,schedule,chunks_per_thread,repeats,time_us,busy_min_us,busy_mean_us,busy_max_us",
              row.str());
    return time;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/core_scheduling.csv",
                                 {"--algorithms", "--max-threads", "--chunks", "--throughput-output"});
        const int hardwareThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int maxThreads = parseNumber<int>(options.parameter("--max-threads", std::to_string(hardwareThreads)));
        const int chunks = parseNumber<int>(options.parameter("--chunks", "8"));
        const std::string throughputOutput = options.parameter("--throughput-output", "results/core_throughput.csv");
        if (maxThreads <= 0 || chunks <= 0) throw std::invalid_argument("--max-threads and --chunks must be positive");
        const std::vector<std::string> selected = split(options.parameter("--algorithms", "bernsen,niblack,median_filter,contrast"));

        const int performanceCores = systemNumber("hw.perflevel0.physicalcpu");
        std::cout << "Logical CPUs: " << hardwareThreads;
        if (performanceCores > 0) {
            std::cout << ", performance cores: " << performanceCores << ", efficiency cores: "
                      << systemNumber("hw.perflevel1.physicalcpu") << ", physical cores: " << systemNumber("hw.physicalcpu");
        }
        std::cout << '\n';

        const Image image = loadImage(options.image);
        for (const std::string& name : selected) {
            auto found = std::find_if(algorithms().begin(), algorithms().end(), [&](const Algorithm& a) { return a.name == name; });
            if (found == algorithms().end()) throw std::invalid_argument("Unknown algorithm: " + name);
            const Algorithm& algorithm = *found;
            Image expected = image;
            algorithm.run(image, expected, 1);

            const std::vector<double> copies = measureThroughput(algorithm, image, expected, maxThreads, options, throughputOutput);
            std::cout << algorithm.name << ": " << maxThreads << " copies at once do the work of "
                      << maxThreads * copies.front() / copies.back() << " copies on one core";
            if (performanceCores > 0 && performanceCores < maxThreads) {
                const double onPerformance = performanceCores * copies.front() / copies[performanceCores - 1];
                const double onAll = maxThreads * copies.front() / copies.back();
                std::cout << " (" << onPerformance << " with " << performanceCores << " copies), so an efficiency core is about "
                          << (onAll - onPerformance) / (maxThreads - performanceCores) << " of a performance core";
            }
            std::cout << '\n';

            std::vector<double> times[3];
            for (Schedule schedule : {Schedule::Static, Schedule::Dynamic, Schedule::Guided}) {
                for (int threads = 1; threads <= maxThreads; ++threads) {
                    times[static_cast<int>(schedule)].push_back(
                        measureSchedule(algorithm, image, expected, threads, {schedule, chunks, nullptr}, options));
                }
            }
            for (int sweep : {1, 2, 4, 16, 32}) {
                if (sweep != chunks) measureSchedule(algorithm, image, expected, maxThreads, {Schedule::Dynamic, sweep, nullptr}, options);
            }
            const double single = times[0].front();
            std::cout << algorithm.name << ": 1 thread " << single / 1000 << " ms; speedup with " << maxThreads
                      << " threads: static " << single / times[0].back() << "x, dynamic " << single / times[1].back()
                      << "x, guided " << single / times[2].back() << "x\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
