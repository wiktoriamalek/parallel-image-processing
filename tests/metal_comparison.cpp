#include "algorithms.hpp"
#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "metal_cases.hpp"
#include "metal_utils.hpp"
#include "opencl_utils.hpp"
#include <iomanip>
#include <iostream>
#include <sys/utsname.h>

namespace {
std::string openCLInfo(cl_device_id device, cl_device_info field) {
    std::size_t size = 0;
    checkCL(clGetDeviceInfo(device, field, 0, nullptr, &size), "clGetDeviceInfo");
    std::string value(size, '\0');
    checkCL(clGetDeviceInfo(device, field, size, value.data(), nullptr), "clGetDeviceInfo");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::string parameterText(const Parameters& p) {
    std::ostringstream text;
    text << std::setprecision(12) << "threshold=" << p.threshold << " contrast=" << p.contrastFactor
         << " classes=" << p.classes << " max_classes=" << p.maxClasses << " radius=" << p.radius
         << " min_contrast=" << p.minContrast << " k=" << p.localThresholdK << " filter=" << p.filter
         << " median_sort=" << p.medianSort << " median_radius=" << p.medianRadius << " strip=" << p.medianStrip
         << " switching=" << p.medianSwitching << " distortion_model=" << p.distortionModel
         << " k1=" << p.k1 << " k2=" << p.k2 << " k3=" << p.k3 << " p1=" << p.p1 << " p2=" << p.p2
         << " bilinear=" << p.bilinear << " scale=" << p.scale << " seed_x=" << p.seedX << " seed_y=" << p.seedY
         << " tolerance=" << p.tolerance << " local_tiles=" << p.localTiles << " low=" << p.cannyLow << " high=" << p.cannyHigh;
    return text.str();
}

std::string thresholdText(const std::optional<std::vector<int>>& thresholds) {
    if (!thresholds) return "";
    std::ostringstream text;
    for (std::size_t i = 0; i < thresholds->size(); ++i) {
        if (i > 0) text << ' ';
        text << (*thresholds)[i];
    }
    return text.str();
}

Parameters parseParameters(const BenchmarkOptions& options) {
    Parameters p;
    p.threshold = parseNumber<int>(options.parameter("--threshold", "128"));
    p.contrastFactor = parseNumber<float>(options.parameter("--factor", "2"));
    p.classes = parseNumber<int>(options.parameter("--classes", "3"));
    p.maxClasses = parseNumber<int>(options.parameter("--max-classes", "3"));
    p.radius = parseNumber<int>(options.parameter("--radius", "7"));
    p.minContrast = parseNumber<int>(options.parameter("--min-contrast", "15"));
    p.localThresholdK = parseNumber<double>(options.parameter("--k", "0.2"));
    p.filter = parseNumber<int>(options.parameter("--filter", "0"));
    p.medianSort = static_cast<int>(parseMedianSort(options.parameter("--median-sort", "standard")));
    p.medianRadius = parseNumber<int>(options.parameter("--median-radius", "1"));
    p.medianStrip = parseNumber<int>(options.parameter("--strip", "8"));
    p.medianSwitching = parseNumber<int>(options.parameter("--switching", "0")) != 0;
    p.distortionModel = parseNumber<int>(options.parameter("--model", "0"));
    p.k1 = parseNumber<float>(options.parameter("--k1", "0.15"));
    p.k2 = parseNumber<float>(options.parameter("--k2", "0"));
    p.k3 = parseNumber<float>(options.parameter("--k3", "0"));
    p.p1 = parseNumber<float>(options.parameter("--p1", "0"));
    p.p2 = parseNumber<float>(options.parameter("--p2", "0"));
    p.bilinear = parseNumber<int>(options.parameter("--bilinear", "1")) != 0;
    p.scale = parseNumber<float>(options.parameter("--scale", "2"));
    p.tolerance = parseNumber<int>(options.parameter("--tolerance", "20"));
    p.localTiles = parseNumber<int>(options.parameter("--local-tiles", "1")) != 0;
    p.cannyLow = parseNumber<float>(options.parameter("--low", "50"));
    p.cannyHigh = parseNumber<float>(options.parameter("--high", "100"));
    return p;
}

bool selectedCase(const MetalCase& test, const std::string& selection) {
    if (selection == "all" || test.name == selection) return true;
    const auto basic = metalCases(false);
    const auto found = std::find_if(basic.begin(), basic.end(), [&](const MetalCase& candidate) {
        return candidate.name == selection;
    });
    return found != basic.end() && test.algorithm == found->algorithm;
}

void verifyResult(const MetalCase& test, const Image& image, const Image& reference, const Image& output,
                  const Timing& timing, const std::string& backend) {
    const ImageDifference difference = metalDifference(reference, output);
    if (test.algorithm == Algorithm::Otsu || test.algorithm == Algorithm::MultiOtsu) {
        std::vector<int> thresholds;
        if (test.algorithm == Algorithm::MultiOtsu) {
            if (!timing.thresholds) throw std::runtime_error(backend + " did not return thresholds");
            thresholds = *timing.thresholds;
        } else {
            int threshold = 0;
            for (std::size_t i = 0; i < image.pixels.size(); ++i) {
                if (output.pixels[i] == 0) threshold = std::max(threshold, static_cast<int>(image.pixels[i]));
            }
            thresholds.push_back(threshold);
        }
        checkMetalSegmentation(test, image, output, thresholds);
        return;
    }
    if (isLocalThreshold(test)) {
        checkMetalLocalThreshold(test, image, reference, output, backend);
        return;
    }
    const bool floating = test.algorithm == Algorithm::Contrast || test.algorithm == Algorithm::Convolution ||
                          test.algorithm == Algorithm::Resize || test.algorithm == Algorithm::Distortion;
    if (difference.maximum > (floating ? 1 : 0)) {
        throw std::runtime_error(test.name + " " + backend + " differs from CPU on " + std::to_string(difference.pixels) +
                                 " pixels, max error " + std::to_string(difference.maximum));
    }
}
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "metal_comparison --image PATH --output CSV --algorithm all --variants basic --repeats 100 --rounds 3\n"
                         "Algorithms: threshold negative contrast otsu multi_otsu peaks_valleys bernsen niblack sauvola convolution median distortion resize region_growing canny\n"
                         "--variants basic|all; --cpu 0|1; --threads N\n"
                         "Parameters: --threshold --factor --classes --max-classes --radius --min-contrast --k --filter\n"
                         "--median-sort standard|network|bitonic|shared --median-radius --strip --switching 0|1\n"
                         "--model 0|1 --k1 --k2 --k3 --p1 --p2 --bilinear 0|1 --scale\n"
                         "--seed-x --seed-y --tolerance --local-tiles 0|1 --low --high\n"
                         "All durations are medians in microseconds after warm-up. A total-time ratio > 1 favors Metal.\n";
            return 0;
        }
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/opencl_metal.csv",
            {"--algorithm", "--variants", "--rounds", "--cpu", "--threshold", "--factor", "--classes", "--max-classes",
             "--radius", "--min-contrast", "--k", "--filter", "--median-sort", "--median-radius", "--strip", "--switching",
             "--model", "--k1", "--k2", "--k3", "--p1", "--p2", "--bilinear", "--scale", "--seed-x", "--seed-y",
             "--tolerance", "--local-tiles", "--low", "--high"});
        const std::string variants = options.parameter("--variants", "basic");
        if (variants != "basic" && variants != "all") throw std::invalid_argument("--variants must be basic or all");
        const int rounds = parseNumber<int>(options.parameter("--rounds", "3"));
        if (rounds < 1) throw std::invalid_argument("--rounds must be positive");
        const bool measureCPU = parseNumber<int>(options.parameter("--cpu", "0")) != 0;
        const Image image = loadImage(options.image);
        Parameters base = parseParameters(options);
        base.seedX = parseNumber<int>(options.parameter("--seed-x", std::to_string(image.width / 2)));
        base.seedY = parseNumber<int>(options.parameter("--seed-y", std::to_string(image.height / 2)));
        const std::string selection = options.parameter("--algorithm", "all");
        std::vector<MetalCase> cases;
        for (const auto& test : metalCases(variants == "all", base)) {
            if (selectedCase(test, selection)) cases.push_back(test);
        }
        if (cases.empty()) throw std::invalid_argument("Unknown algorithm or variant: " + selection);
        OpenCL opencl("kernels/negative.cl");
        const std::string openclName = openCLInfo(opencl.device, CL_DEVICE_NAME);
        const std::string driver = openCLInfo(opencl.device, CL_DRIVER_VERSION);
        const std::string metalName = metalDeviceName();
        struct utsname system{};
        if (uname(&system) != 0) throw std::runtime_error("Could not read the OS version");
        const std::string os = std::string(system.sysname) + " " + system.release + " " + system.machine;
        std::cout << "OpenCL: " << openclName << " / Metal: " << metalName << " / build: " << IMAGE_BUILD_TYPE << std::endl;
        const Histogram histogram = calculateHistogram(image);
        const auto distinct = std::count_if(histogram.begin(), histogram.end(), [](auto count) { return count != 0; });
        for (const auto& test : cases) {
            if (test.algorithm == Algorithm::MultiOtsu && test.parameters.classes > distinct && cases.size() > 1) {
                std::cout << "Skipping " << test.name << ": image has only " << distinct << " distinct brightness levels" << std::endl;
                continue;
            }
            const Image reference = metalCPUReference(test, image);
            double cpu1 = -1;
            double cpuN = -1;
            if (measureCPU) {
                Image cpuOutput;
                cpu1 = runAlgorithm(test.algorithm, Mode::CPU1, image, test.parameters,
                                    options.threads, options.repeats, cpuOutput).totalUs;
                cpuN = runAlgorithm(test.algorithm, Mode::CPUN, image, test.parameters,
                                    options.threads, options.repeats, cpuOutput).totalUs;
            }
            for (int round = 0; round < rounds; ++round) {
                Image clOutput;
                Image metalOutput;
                Timing clTiming;
                Timing metalTiming;
                const bool metalFirst = round % 2 != 0;
                if (metalFirst) {
                    metalTiming = runAlgorithm(test.algorithm, Mode::Metal, image, test.parameters, options.threads, options.repeats, metalOutput);
                    clTiming = runAlgorithm(test.algorithm, Mode::GPU, image, test.parameters, options.threads, options.repeats, clOutput);
                } else {
                    clTiming = runAlgorithm(test.algorithm, Mode::GPU, image, test.parameters, options.threads, options.repeats, clOutput);
                    metalTiming = runAlgorithm(test.algorithm, Mode::Metal, image, test.parameters, options.threads, options.repeats, metalOutput);
                }
                verifyResult(test, image, reference, clOutput, clTiming, "OpenCL");
                verifyResult(test, image, reference, metalOutput, metalTiming, "Metal");
                const ImageDifference clError = metalDifference(reference, clOutput);
                const ImageDifference metalError = metalDifference(reference, metalOutput);
                const ImageDifference pairError = metalDifference(clOutput, metalOutput);
                const double totalRatio = clTiming.totalUs / metalTiming.totalUs;
                const double kernelRatio = metalTiming.kernelUs > 0 ? clTiming.kernelUs / metalTiming.kernelUs : -1;
                std::ostringstream row;
                row << std::setprecision(12) << std::quoted(options.image, '"', '"') << ',' << image.width << ',' << image.height << ','
                    << std::quoted(test.name, '"', '"') << ',' << std::quoted(parameterText(test.parameters), '"', '"') << ',' << round + 1 << ','
                    << (metalFirst ? "metal_first" : "opencl_first") << ',' << options.repeats << ',' << options.threads << ','
                    << std::quoted(openclName, '"', '"') << ',' << std::quoted(driver, '"', '"') << ',' << std::quoted(metalName, '"', '"') << ','
                    << std::quoted(os, '"', '"') << ',' << std::quoted(IMAGE_BUILD_TYPE, '"', '"') << ',' << std::quoted(__clang_version__, '"', '"') << ','
                    << cpu1 << ',' << cpuN << ',' << clTiming.totalUs << ',' << metalTiming.totalUs << ','
                    << clTiming.kernelUs << ',' << metalTiming.kernelUs << ',' << std::quoted(clTiming.gpuTimingMethod, '"', '"') << ','
                    << std::quoted(metalTiming.gpuTimingMethod, '"', '"') << ',' << totalRatio << ',' << kernelRatio << ','
                    << clError.pixels << ',' << clError.maximum << ',' << clError.mean << ','
                    << metalError.pixels << ',' << metalError.maximum << ',' << metalError.mean << ','
                    << pairError.pixels << ',' << pairError.maximum << ',' << pairError.mean << ','
                    << std::quoted(thresholdText(clTiming.thresholds), '"', '"') << ',' << std::quoted(thresholdText(metalTiming.thresholds), '"', '"');
                appendCsv(options.output,
                    "image,width,height,algorithm,parameters,round,backend_order,repeats,threads,opencl_device,opencl_driver,metal_device,os,build_type,compiler,cpu_1t_us,cpu_nt_us,opencl_total_us,metal_total_us,opencl_kernel_us,metal_kernel_us,opencl_gpu_timing,metal_gpu_timing,opencl_over_metal_total,opencl_over_metal_kernel,opencl_cpu_different_pixels,opencl_cpu_max_error,opencl_cpu_mean_error,metal_cpu_different_pixels,metal_cpu_max_error,metal_cpu_mean_error,opencl_metal_different_pixels,opencl_metal_max_error,opencl_metal_mean_error,opencl_thresholds,metal_thresholds",
                    row.str());
                std::cout << test.name << " round " << round + 1 << ": OpenCL " << clTiming.totalUs << " us / Metal "
                          << metalTiming.totalUs << " us; OpenCL/Metal " << totalRatio << "; different pixels " << pairError.pixels << std::endl;
            }
        }
        std::cout << "Saved " << options.output << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
