#include "common_benchmarking.hpp"
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
std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open " + path);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void report(const std::string& output, const std::string& api, const std::string& compilation, std::size_t localX, std::size_t localY,
            double kernelUs) {
    std::ostringstream row;
    row << std::fixed << std::setprecision(3) << api << ',' << std::quoted(compilation, '"', '"') << ',' << localX << ',' << localY << ','
        << kernelUs;
    appendCsv(output, "api,compilation,local_x,local_y,kernel_us", row.str());
    std::cout << api << ", " << compilation << ", group " << localX << 'x' << localY << ": " << kernelUs << " us" << std::endl;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/metal_median_options.csv");
        const Image image = loadImage(options.image);
        const int width = image.width;
        const int height = image.height;
        const int switching = 0;
        const std::size_t size = image.pixels.size();

        OpenCL cl("kernels/median_filter.cl", "#define MEDIAN_RADIUS 1");
        cl_int error;
        cl_kernel kernel = clCreateKernel(cl.program, "median_insertion", &error);
        checkCL(error, "clCreateKernel");
        cl_mem input = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, const_cast<unsigned char*>(image.pixels.data()), &error);
        checkCL(error, "clCreateBuffer");
        cl_mem output = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, size, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        checkCL(clSetKernelArg(kernel, 0, sizeof(input), &input), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 1, sizeof(output), &output), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 2, sizeof(int), &width), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 3, sizeof(int), &height), "clSetKernelArg");
        checkCL(clSetKernelArg(kernel, 4, sizeof(int), &switching), "clSetKernelArg");
        std::vector<double> openclTimes;
        for (int repeat = -10; repeat < options.repeats; ++repeat) {
            cl_event event = nullptr;
            enqueueTiles(cl.queue, kernel, static_cast<std::size_t>(width), static_cast<std::size_t>(height), &event);
            checkCL(clFinish(cl.queue), "clFinish");
            if (repeat >= 0) openclTimes.push_back(kernelTimeUs(event));
            checkCL(clReleaseEvent(event), "clReleaseEvent");
        }
        report(options.output, "OpenCL", "runtime, default options", tileGroupSize, tileGroupSize, median(openclTimes));
        clReleaseMemObject(input);
        clReleaseMemObject(output);
        clReleaseKernel(kernel);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU found");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> metalInput = [device newBufferWithBytes:image.pixels.data() length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> metalOutput = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        NSError* metalError = nil;
        const std::string source = "#define MEDIAN_RADIUS 1\n" + readFile("kernels/metal/median_filter.metal");
        NSString* code = [NSString stringWithUTF8String:source.c_str()];
        MTLCompileOptions* safe = [MTLCompileOptions new];
        safe.mathMode = MTLMathModeSafe;
        safe.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        std::vector<std::pair<std::string, id<MTLLibrary>>> libraries = {
            {"runtime, MTLMathModeSafe and precise functions", [device newLibraryWithSource:code options:safe error:&metalError]},
            {"runtime, default options", [device newLibraryWithSource:code options:[MTLCompileOptions new] error:&metalError]}};
        for (const auto& [name, file] : std::vector<std::pair<std::string, std::string>>{
                 {"offline, project flags (-O2 -fno-fast-math -ffp-contract=on)", "median_project.metallib"},
                 {"offline, -O2", "median_default.metallib"}, {"offline, -Os", "median_size.metallib"}}) {
            const std::string path = std::string(METAL_VARIANT_DIRECTORY) + "/" + file;
            libraries.push_back({name, [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]]
                                                           error:&metalError]});
        }
        for (const auto& [name, library] : libraries) {
            if (!library) throw std::runtime_error("Could not build the Metal median variant: " + name);
            id<MTLComputePipelineState> state =
                [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"median_insertion"] error:&metalError];
            if (!state) throw std::runtime_error("Could not create the Metal median pipeline");
            std::vector<double> times;
            for (int repeat = -10; repeat < options.repeats; ++repeat) {
                @autoreleasepool {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:state];
                    [encoder setBuffer:metalInput offset:0 atIndex:0];
                    [encoder setBuffer:metalOutput offset:0 atIndex:1];
                    [encoder setBytes:&width length:sizeof(int) atIndex:2];
                    [encoder setBytes:&height length:sizeof(int) atIndex:3];
                    [encoder setBytes:&switching length:sizeof(int) atIndex:4];
                    [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(tileGroupSize, tileGroupSize, 1)];
                    [encoder endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                    if (repeat >= 0) times.push_back(([command GPUEndTime] - [command GPUStartTime]) * 1e6);
                }
            }
            report(options.output, "Metal", name, tileGroupSize, tileGroupSize, median(times));
        }
        std::cout << "Saved " << options.output << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
