#include "common_benchmarking.hpp"
#include "convolution.hpp"
#include "image_io.hpp"
#include "opencl_utils.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Case {
    std::string kernel;
    std::size_t localX;
    std::size_t localY;
};

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open " + path);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

cl_kernel createKernel(cl_program program, const char* name) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, name, &error);
    checkCL(error, "clCreateKernel");
    return kernel;
}

cl_mem createBuffer(cl_context context, cl_mem_flags flags, std::size_t size, const void* host) {
    cl_int error;
    cl_mem buffer = clCreateBuffer(context, flags, size, const_cast<void*>(host), &error);
    checkCL(error, "clCreateBuffer");
    return buffer;
}

template<class T>
void setArgument(cl_kernel kernel, cl_uint index, const T& value) {
    checkCL(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw std::runtime_error("Metal kernel not found");
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) throw std::runtime_error("Could not create a Metal pipeline");
    return state;
}

std::size_t workGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info field) {
    std::size_t value = 0;
    checkCL(clGetKernelWorkGroupInfo(kernel, device, field, sizeof(value), &value, nullptr), "clGetKernelWorkGroupInfo");
    return value;
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/opencl_metal_workgroups.csv");
        const Image image = loadImage(options.image);
        const int width = image.width;
        const int height = image.height;
        const int pixelCount = static_cast<int>(image.pixels.size());
        const std::size_t size = image.pixels.size();
        const int threshold = 128;
        const int absoluteValue = 0;
        const int switching = 0;
        const auto& box = convolutionFilters().at(1).kernel;
        float filter[9];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) filter[row * 3 + column] = box[row][column];
        }

        OpenCL cl({"kernels/threshold.cl", "kernels/convolution.cl", "kernels/median_filter.cl"}, "#define MEDIAN_RADIUS 1");
        cl_mem input = createBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, image.pixels.data());
        cl_mem pixels = createBuffer(cl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, image.pixels.data());
        cl_mem output = createBuffer(cl.context, CL_MEM_WRITE_ONLY, size, nullptr);
        cl_mem coefficients = createBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(filter), filter);
        cl_kernel clThreshold = createKernel(cl.program, "threshold_kernel");
        setArgument(clThreshold, 0, pixels);
        setArgument(clThreshold, 1, threshold);
        setArgument(clThreshold, 2, pixelCount);
        cl_kernel clConvolution = createKernel(cl.program, "convolution3x3");
        setArgument(clConvolution, 0, input);
        setArgument(clConvolution, 1, output);
        setArgument(clConvolution, 2, width);
        setArgument(clConvolution, 3, height);
        setArgument(clConvolution, 4, coefficients);
        setArgument(clConvolution, 5, absoluteValue);
        cl_kernel clMedian = createKernel(cl.program, "median_insertion");
        setArgument(clMedian, 0, input);
        setArgument(clMedian, 1, output);
        setArgument(clMedian, 2, width);
        setArgument(clMedian, 3, height);
        setArgument(clMedian, 4, switching);
        const std::size_t maximumGroup = workGroupInfo(clThreshold, cl.device, CL_KERNEL_WORK_GROUP_SIZE);
        std::cout << "OpenCL: largest work-group " << maximumGroup << ", preferred multiple "
                  << workGroupInfo(clThreshold, cl.device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE) << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU found");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* metalError = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@METAL_LIBRARY_PATH] error:&metalError];
        const std::string medianSource = "#define MEDIAN_RADIUS 1\n" + readFile("kernels/metal/median_filter.metal");
        MTLCompileOptions* compileOptions = [MTLCompileOptions new];
        compileOptions.mathMode = MTLMathModeSafe;
        compileOptions.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        id<MTLLibrary> medianLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:medianSource.c_str()]
                                                            options:compileOptions error:&metalError];
        if (!library || !medianLibrary) throw std::runtime_error("Could not load the Metal kernels");
        id<MTLComputePipelineState> metalThreshold = pipeline(device, library, @"threshold_kernel");
        id<MTLComputePipelineState> metalConvolution = pipeline(device, library, @"convolution3x3");
        id<MTLComputePipelineState> metalMedian = pipeline(device, medianLibrary, @"median_insertion");
        std::cout << "Metal: SIMD width " << [metalThreshold threadExecutionWidth] << ", largest threadgroup "
                  << [metalThreshold maxTotalThreadsPerThreadgroup] << std::endl;
        id<MTLBuffer> metalInput = [device newBufferWithBytes:image.pixels.data() length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> metalPixels = [device newBufferWithBytes:image.pixels.data() length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> metalOutput = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> metalFilter = [device newBufferWithBytes:filter length:sizeof(filter) options:MTLResourceStorageModeShared];
        id<MTLCounterSampleBuffer> samples = nil;
        for (id<MTLCounterSet> counterSet in [device counterSets]) {
            if (![[counterSet name] isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet = counterSet;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = 2;
            samples = [device newCounterSampleBufferWithDescriptor:descriptor error:&metalError];
        }
        if (!samples) throw std::runtime_error("Metal GPU timestamps are unavailable");

        std::vector<Case> cases;
        for (std::size_t local : {32, 64, 128, 256, 512, 1024}) cases.push_back({"threshold", local, 1});
        cases.push_back({"threshold", 0, 0});
        for (auto [x, y] : std::vector<std::pair<std::size_t, std::size_t>>{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {32, 1}, {64, 1}, {128, 1},
                                                                           {256, 1}, {32, 2}, {32, 4}, {32, 8}, {64, 4}, {16, 4}, {8, 32}, {1, 32}}) {
            cases.push_back({"convolution", x, y});
            cases.push_back({"median", x, y});
        }
        cases.push_back({"convolution", 0, 0});
        cases.push_back({"median", 0, 0});

        std::vector<unsigned char> clResult(size);
        for (int pass = 0; pass < 2; ++pass) {
            for (const Case& test : cases) {
                const bool line = test.kernel == "threshold";
                cl_kernel clKernel = line ? clThreshold : test.kernel == "convolution" ? clConvolution : clMedian;
                id<MTLComputePipelineState> metalKernel = line ? metalThreshold : test.kernel == "convolution" ? metalConvolution : metalMedian;
                const bool openclAllowed = test.localX * test.localY <= maximumGroup;
                const std::size_t lineGlobal = (size + std::max<std::size_t>(test.localX, 1) - 1) / std::max<std::size_t>(test.localX, 1)
                                               * std::max<std::size_t>(test.localX, 1);
                const std::size_t tileGlobal[2] = {
                    test.localX ? (static_cast<std::size_t>(width) + test.localX - 1) / test.localX * test.localX : static_cast<std::size_t>(width),
                    test.localY ? (static_cast<std::size_t>(height) + test.localY - 1) / test.localY * test.localY : static_cast<std::size_t>(height)};
                const std::size_t tileLocal[2] = {test.localX, test.localY};
                std::vector<double> openclTimes;
                for (int repeat = -5; openclAllowed && repeat < options.repeats; ++repeat) {
                    cl_event event = nullptr;
                    if (line) {
                        checkCL(clEnqueueNDRangeKernel(cl.queue, clKernel, 1, nullptr, &lineGlobal, test.localX ? &test.localX : nullptr,
                                                       0, nullptr, &event), "clEnqueueNDRangeKernel");
                    } else {
                        checkCL(clEnqueueNDRangeKernel(cl.queue, clKernel, 2, nullptr, tileGlobal, test.localX ? tileLocal : nullptr,
                                                       0, nullptr, &event), "clEnqueueNDRangeKernel");
                    }
                    checkCL(clFinish(cl.queue), "clFinish");
                    if (repeat >= 0) openclTimes.push_back(kernelTimeUs(event));
                    checkCL(clReleaseEvent(event), "clReleaseEvent");
                }
                if (openclAllowed) {
                    checkCL(clEnqueueReadBuffer(cl.queue, line ? pixels : output, CL_TRUE, 0, size, clResult.data(), 0, nullptr, nullptr),
                            "clEnqueueReadBuffer");
                }
                double metalCounterUs = -1.0;
                double metalCommandUs = -1.0;
                std::size_t different = 0;
                if (test.localX) {
                    std::vector<double> counterTimes, commandTimes;
                    for (int repeat = -5; repeat < options.repeats; ++repeat) {
                        @autoreleasepool {
                            MTLComputePassDescriptor* descriptor = [MTLComputePassDescriptor computePassDescriptor];
                            descriptor.sampleBufferAttachments[0].sampleBuffer = samples;
                            descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
                            descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
                            id<MTLCommandBuffer> command = [queue commandBuffer];
                            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDescriptor:descriptor];
                            [encoder setComputePipelineState:metalKernel];
                            if (line) {
                                [encoder setBuffer:metalPixels offset:0 atIndex:0];
                                [encoder setBytes:&threshold length:sizeof(int) atIndex:1];
                                [encoder setBytes:&pixelCount length:sizeof(int) atIndex:2];
                                [encoder dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(test.localX, 1, 1)];
                            } else {
                                [encoder setBuffer:metalInput offset:0 atIndex:0];
                                [encoder setBuffer:metalOutput offset:0 atIndex:1];
                                [encoder setBytes:&width length:sizeof(int) atIndex:2];
                                [encoder setBytes:&height length:sizeof(int) atIndex:3];
                                if (test.kernel == "convolution") {
                                    [encoder setBuffer:metalFilter offset:0 atIndex:4];
                                    [encoder setBytes:&absoluteValue length:sizeof(int) atIndex:5];
                                } else {
                                    [encoder setBytes:&switching length:sizeof(int) atIndex:4];
                                }
                                [encoder dispatchThreads:MTLSizeMake(width, height, 1)
                                   threadsPerThreadgroup:MTLSizeMake(test.localX, test.localY, 1)];
                            }
                            [encoder endEncoding];
                            [command commit];
                            [command waitUntilCompleted];
                            NSData* resolved = [samples resolveCounterRange:NSMakeRange(0, 2)];
                            const auto* timestamps = static_cast<const MTLCounterResultTimestamp*>([resolved bytes]);
                            if (repeat >= 0) {
                                counterTimes.push_back(static_cast<double>(timestamps[1].timestamp - timestamps[0].timestamp) / 1000.0);
                                commandTimes.push_back(([command GPUEndTime] - [command GPUStartTime]) * 1e6);
                            }
                        }
                    }
                    metalCounterUs = median(counterTimes);
                    metalCommandUs = median(commandTimes);
                    const auto* result = static_cast<const unsigned char*>([(line ? metalPixels : metalOutput) contents]);
                    if (openclAllowed) {
                        for (std::size_t i = 0; i < size; ++i) different += result[i] != clResult[i];
                    }
                }
                if (pass == 0) continue;
                const double openclUs = openclAllowed ? median(openclTimes) : -1.0;
                std::ostringstream row;
                row << std::fixed << std::setprecision(3) << test.kernel << ',' << test.localX << ',' << test.localY << ',' << openclUs << ','
                    << metalCounterUs << ',' << metalCommandUs << ',' << different;
                appendCsv(options.output, "kernel,local_x,local_y,opencl_kernel_us,metal_kernel_us,metal_command_buffer_us,different_bytes",
                          row.str());
                std::cout << test.kernel << ' ' << test.localX << 'x' << test.localY << ": OpenCL " << openclUs << " us, Metal "
                          << metalCounterUs << " us, different bytes " << different << std::endl;
            }
        }
        for (cl_kernel kernel : {clThreshold, clConvolution, clMedian}) clReleaseKernel(kernel);
        for (cl_mem buffer : {input, pixels, output, coefficients}) clReleaseMemObject(buffer);
        std::cout << "Saved " << options.output << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
