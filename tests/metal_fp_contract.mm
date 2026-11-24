#include "common_benchmarking.hpp"
#include "convolution.hpp"
#include "distortion.hpp"
#include "image_io.hpp"
#include "opencl_utils.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Result {
    std::size_t distortionDifferent;
    std::size_t blurDifferent;
    double distortionUs;
    double blurUs;
};

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open " + path);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

std::size_t differences(const std::vector<unsigned char>& expected, const unsigned char* actual) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) count += expected[i] != actual[i];
    return count;
}

template<class T>
void setArgument(cl_kernel kernel, cl_uint index, const T& value) {
    checkCL(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

Result runOpenCL(const OpenCL& cl, const std::string& source, const char* buildOptions, const Image& image, const float* filter,
                 const Image& distortionReference, const Image& blurReference, float k1, float k2, int repeats) {
    const int width = image.width;
    const int height = image.height;
    const int absoluteValue = 0;
    const std::size_t size = image.pixels.size();
    const char* text = source.c_str();
    cl_int error;
    cl_program program = clCreateProgramWithSource(cl.context, 1, &text, nullptr, &error);
    checkCL(error, "clCreateProgramWithSource");
    checkCL(clBuildProgram(program, 1, &cl.device, buildOptions, nullptr, nullptr), "clBuildProgram");
    cl_kernel distortion = clCreateKernel(program, "distortion_correction", &error);
    checkCL(error, "clCreateKernel");
    cl_kernel blur = clCreateKernel(program, "convolution3x3", &error);
    checkCL(error, "clCreateKernel");
    cl_mem input = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, const_cast<unsigned char*>(image.pixels.data()), &error);
    checkCL(error, "clCreateBuffer");
    cl_mem output = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, size, nullptr, &error);
    checkCL(error, "clCreateBuffer");
    cl_mem coefficients = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof(float), const_cast<float*>(filter), &error);
    checkCL(error, "clCreateBuffer");
    setArgument(distortion, 0, input);
    setArgument(distortion, 1, output);
    setArgument(distortion, 2, height);
    setArgument(distortion, 3, width);
    setArgument(distortion, 4, k1);
    setArgument(distortion, 5, k2);
    setArgument(blur, 0, input);
    setArgument(blur, 1, output);
    setArgument(blur, 2, width);
    setArgument(blur, 3, height);
    setArgument(blur, 4, coefficients);
    setArgument(blur, 5, absoluteValue);
    std::vector<unsigned char> result(size);
    auto run = [&](cl_kernel kernel, const Image& reference, std::size_t& different) {
        std::vector<double> times;
        for (int repeat = -5; repeat < repeats; ++repeat) {
            cl_event event = nullptr;
            enqueueTiles(cl.queue, kernel, static_cast<std::size_t>(width), static_cast<std::size_t>(height), &event);
            checkCL(clFinish(cl.queue), "clFinish");
            if (repeat >= 0) times.push_back(kernelTimeUs(event));
            checkCL(clReleaseEvent(event), "clReleaseEvent");
        }
        checkCL(clEnqueueReadBuffer(cl.queue, output, CL_TRUE, 0, size, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
        different = differences(reference.pixels, result.data());
        return median(times);
    };
    Result values{};
    values.distortionUs = run(distortion, distortionReference, values.distortionDifferent);
    values.blurUs = run(blur, blurReference, values.blurDifferent);
    for (cl_mem buffer : {input, output, coefficients}) clReleaseMemObject(buffer);
    clReleaseKernel(distortion);
    clReleaseKernel(blur);
    clReleaseProgram(program);
    return values;
}

Result runMetal(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, const Image& image, const float* filter,
                const Image& distortionReference, const Image& blurReference, float k1, float k2, int repeats) {
    const int width = image.width;
    const int height = image.height;
    const int absoluteValue = 0;
    const std::size_t size = image.pixels.size();
    NSError* error = nil;
    id<MTLComputePipelineState> distortion =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"distortion_correction"] error:&error];
    id<MTLComputePipelineState> blur = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"convolution3x3"] error:&error];
    if (!distortion || !blur) throw std::runtime_error("Could not create the Metal pipelines");
    id<MTLBuffer> input = [device newBufferWithBytes:image.pixels.data() length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> coefficients = [device newBufferWithBytes:filter length:9 * sizeof(float) options:MTLResourceStorageModeShared];
    auto run = [&](id<MTLComputePipelineState> state, bool isDistortion, const Image& reference, std::size_t& different) {
        std::vector<double> times;
        for (int repeat = -5; repeat < repeats; ++repeat) {
            @autoreleasepool {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:state];
                [encoder setBuffer:input offset:0 atIndex:0];
                [encoder setBuffer:output offset:0 atIndex:1];
                if (isDistortion) {
                    [encoder setBytes:&height length:sizeof(int) atIndex:2];
                    [encoder setBytes:&width length:sizeof(int) atIndex:3];
                    [encoder setBytes:&k1 length:sizeof(float) atIndex:4];
                    [encoder setBytes:&k2 length:sizeof(float) atIndex:5];
                } else {
                    [encoder setBytes:&width length:sizeof(int) atIndex:2];
                    [encoder setBytes:&height length:sizeof(int) atIndex:3];
                    [encoder setBuffer:coefficients offset:0 atIndex:4];
                    [encoder setBytes:&absoluteValue length:sizeof(int) atIndex:5];
                }
                [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (repeat >= 0) times.push_back(([command GPUEndTime] - [command GPUStartTime]) * 1e6);
            }
        }
        different = differences(reference.pixels, static_cast<const unsigned char*>([output contents]));
        return median(times);
    };
    Result values{};
    values.distortionUs = run(distortion, true, distortionReference, values.distortionDifferent);
    values.blurUs = run(blur, false, blurReference, values.blurDifferent);
    return values;
}

void report(const std::string& output, const std::string& api, const std::string& compilation, const Result& result) {
    std::ostringstream row;
    row << std::fixed << std::setprecision(3) << api << ',' << std::quoted(compilation, '"', '"') << ',' << result.distortionDifferent << ','
        << result.blurDifferent << ',' << result.distortionUs << ',' << result.blurUs;
    appendCsv(output, "api,compilation,distortion_different_pixels,box_blur_different_pixels,distortion_kernel_us,box_blur_kernel_us",
              row.str());
    std::cout << api << " (" << compilation << "): distortion differs from CPU in " << result.distortionDifferent
              << " pixels, box blur in " << result.blurDifferent << "; kernels " << result.distortionUs << " / " << result.blurUs
              << " us" << std::endl;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/metal_fp_contract.csv", {"--k1", "--k2"});
        const float k1 = parseNumber<float>(options.parameter("--k1", "0.15"));
        const float k2 = parseNumber<float>(options.parameter("--k2", "0"));
        const Image image = loadImage(options.image);
        Image distortionReference = image;
        correctRadialDistortionCPU(image, k1, k2, distortionReference, 1);
        const ConvolutionFilter& box = convolutionFilters().at(1);
        Image blurReference = image;
        convolution(image, box.kernel, box.absoluteValue, blurReference, 1);
        float filter[9];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) filter[row * 3 + column] = box.kernel[row][column];
        }

        OpenCL cl("kernels/negative.cl");
        const std::string source = readFile("kernels/distortion_correction.cl") + "\n" + readFile("kernels/convolution.cl");
        for (const char* buildOptions : {"", "-cl-mad-enable", "-cl-fp32-correctly-rounded-divide-sqrt", "-cl-fast-relaxed-math"}) {
            report(options.output, "OpenCL", *buildOptions ? buildOptions : "default", runOpenCL(cl, source, buildOptions, image, filter,
                   distortionReference, blurReference, k1, k2, options.repeats));
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU found");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const std::vector<std::pair<std::string, std::string>> variants = {
            {"off", "-fno-fast-math -ffp-contract=off"}, {"on", "-fno-fast-math -ffp-contract=on"},
            {"fast", "-fno-fast-math -ffp-contract=fast"}, {"fastmath", "default (fast math)"}};
        for (const auto& [name, compilation] : variants) {
            NSError* error = nil;
            const std::string path = std::string(METAL_VARIANT_DIRECTORY) + "/fp_" + name + ".metallib";
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&error];
            if (!library) throw std::runtime_error("Could not load " + path);
            report(options.output, "Metal", compilation, runMetal(device, queue, library, image, filter, distortionReference, blurReference,
                   k1, k2, options.repeats));
        }
        std::cout << "Saved " << options.output << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
