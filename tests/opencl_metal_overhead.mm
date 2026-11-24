#include "common_benchmarking.hpp"
#include "image_io.hpp"
#include "opencl_utils.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* openclSource =
    "__kernel void empty_kernel(__global uchar* pixels) {}\n"
    "__kernel void negative_gpu(__global const uchar* input, __global uchar* output) {\n"
    "    size_t x = get_global_id(0);\n"
    "    output[x] = 255 - input[x];\n"
    "}\n"
    "__kernel void spin(__global float* output, int iterations) {\n"
    "    float value = get_global_id(0);\n"
    "    for (int i = 0; i < iterations; ++i) value = value * 0.999f + 0.5f;\n"
    "    output[get_global_id(0)] = value;\n"
    "}\n";

NSString* const metalSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void empty_kernel(device uchar* pixels [[buffer(0)]], uint x [[thread_position_in_grid]]) {}\n"
     "kernel void negative_gpu(device const uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],\n"
     "                         uint x [[thread_position_in_grid]]) {\n"
     "    output[x] = 255 - input[x];\n"
     "}\n"
     "kernel void spin(device float* output [[buffer(0)]], constant int& iterations [[buffer(1)]],\n"
     "                 uint x [[thread_position_in_grid]]) {\n"
     "    float value = x;\n"
     "    for (int i = 0; i < iterations; ++i) value = value * 0.999f + 0.5f;\n"
     "    output[x] = value;\n"
     "}\n";

const std::size_t pageSize = 16384;

cl_kernel createKernel(cl_program program, const char* name) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, name, &error);
    checkCL(error, "clCreateKernel");
    return kernel;
}

cl_mem createBuffer(cl_context context, cl_mem_flags flags, std::size_t size, void* host = nullptr) {
    cl_int error;
    cl_mem buffer = clCreateBuffer(context, flags, size, host, &error);
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

double runMetal(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state, NSArray<id<MTLBuffer>>* buffers,
                MTLSize grid, MTLSize group) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state];
        for (NSUInteger i = 0; i < [buffers count]; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return ([command GPUEndTime] - [command GPUStartTime]) * 1e6;
    }
}

struct AlignedPixels {
    explicit AlignedPixels(std::size_t size) : size((size + pageSize - 1) / pageSize * pageSize) {
        if (posix_memalign(&data, pageSize, this->size) != 0) throw std::bad_alloc();
        std::memset(data, 0, this->size);
    }
    ~AlignedPixels() { std::free(data); }
    AlignedPixels(const AlignedPixels&) = delete;
    AlignedPixels& operator=(const AlignedPixels&) = delete;
    unsigned char* bytes() { return static_cast<unsigned char*>(data); }
    void* data = nullptr;
    std::size_t size;
};

void checkNegative(const std::vector<unsigned char>& input, const unsigned char* output, const std::string& name) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (output[i] != 255 - input[i]) throw std::runtime_error(name + " gives a wrong negative");
    }
}
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkOptions options(argc, argv, "images/cms_resolution_change/cms_2048.png", "results/opencl_metal_overhead.csv",
                                 {"--units-output"});
        const std::string unitsOutput = options.parameter("--units-output", "results/metal_timestamp_units.csv");
        const Image image = loadImage(options.image);
        OpenCL cl("kernels/negative.cl");
        cl_int error;
        cl_program program = clCreateProgramWithSource(cl.context, 1, &openclSource, nullptr, &error);
        checkCL(error, "clCreateProgramWithSource");
        checkCL(clBuildProgram(program, 1, &cl.device, nullptr, nullptr, nullptr), "clBuildProgram");
        cl_kernel clEmpty = createKernel(program, "empty_kernel");
        cl_kernel clNegative = createKernel(program, "negative_gpu");
        cl_kernel clSpin = createKernel(program, "spin");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU found");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* metalError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:metalSource options:[MTLCompileOptions new] error:&metalError];
        if (!library) throw std::runtime_error("Could not compile the Metal kernels");
        id<MTLComputePipelineState> metalEmpty = pipeline(device, library, @"empty_kernel");
        id<MTLComputePipelineState> metalNegative = pipeline(device, library, @"negative_gpu");
        id<MTLComputePipelineState> metalSpin = pipeline(device, library, @"spin");

        for (int side : {256, 512, 1024, 2048, 4096, 8192}) {
            const std::size_t size = static_cast<std::size_t>(side) * side;
            std::vector<unsigned char> host(size), result(size);
            for (std::size_t i = 0; i < size; ++i) host[i] = image.pixels[i % image.pixels.size()];
            const MTLSize line = MTLSizeMake(size, 1, 1);
            const MTLSize group = MTLSizeMake(256, 1, 1);
            const std::size_t one = 1;

            const double copyUs = measure([&] { std::memcpy(result.data(), host.data(), size); }, options.repeats);

            cl_mem input = createBuffer(cl.context, CL_MEM_READ_ONLY, size);
            cl_mem output = createBuffer(cl.context, CL_MEM_WRITE_ONLY, size);
            setArgument(clEmpty, 0, input);
            setArgument(clNegative, 0, input);
            setArgument(clNegative, 1, output);
            const double clWriteUs = measure([&] {
                checkCL(clEnqueueWriteBuffer(cl.queue, input, CL_TRUE, 0, size, host.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            }, options.repeats);
            checkCL(clEnqueueNDRangeKernel(cl.queue, clNegative, 1, nullptr, &size, &lineGroupSize, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
            checkCL(clFinish(cl.queue), "clFinish");
            const double clReadUs = measure([&] {
                checkCL(clEnqueueReadBuffer(cl.queue, output, CL_TRUE, 0, size, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
            }, options.repeats);
            const double clLaunchUs = measure([&] {
                checkCL(clEnqueueNDRangeKernel(cl.queue, clEmpty, 1, nullptr, &one, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
                checkCL(clFinish(cl.queue), "clFinish");
            }, options.repeats);
            std::vector<double> clKernels;
            const double clTotalUs = measure([&] {
                cl_event event = nullptr;
                checkCL(clEnqueueWriteBuffer(cl.queue, input, CL_TRUE, 0, size, host.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
                checkCL(clEnqueueNDRangeKernel(cl.queue, clNegative, 1, nullptr, &size, &lineGroupSize, 0, nullptr, &event), "clEnqueueNDRangeKernel");
                checkCL(clEnqueueReadBuffer(cl.queue, output, CL_TRUE, 0, size, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
                clKernels.push_back(kernelTimeUs(event));
                checkCL(clReleaseEvent(event), "clReleaseEvent");
            }, options.repeats);
            checkNegative(host, result.data(), "OpenCL");

            AlignedPixels alignedInput(size), alignedOutput(size);
            std::memcpy(alignedInput.bytes(), host.data(), size);
            cl_mem hostInput = createBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, alignedInput.size, alignedInput.data);
            cl_mem hostOutput = createBuffer(cl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, alignedOutput.size, alignedOutput.data);
            setArgument(clNegative, 0, hostInput);
            setArgument(clNegative, 1, hostOutput);
            const double clHostPointerUs = measure([&] {
                void* mapped = clEnqueueMapBuffer(cl.queue, hostInput, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, nullptr, nullptr, &error);
                checkCL(error, "clEnqueueMapBuffer");
                checkCL(clEnqueueUnmapMemObject(cl.queue, hostInput, mapped, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
                checkCL(clEnqueueNDRangeKernel(cl.queue, clNegative, 1, nullptr, &size, &lineGroupSize, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
                void* read = clEnqueueMapBuffer(cl.queue, hostOutput, CL_TRUE, CL_MAP_READ, 0, size, 0, nullptr, nullptr, &error);
                checkCL(error, "clEnqueueMapBuffer");
                checkCL(clEnqueueUnmapMemObject(cl.queue, hostOutput, read, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
                checkCL(clFinish(cl.queue), "clFinish");
            }, options.repeats);
            checkNegative(host, alignedOutput.bytes(), "OpenCL with CL_MEM_USE_HOST_PTR");
            for (cl_mem buffer : {input, output, hostInput, hostOutput}) clReleaseMemObject(buffer);

            id<MTLBuffer> metalInput = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
            id<MTLBuffer> metalOutput = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
            const double metalWriteUs = measure([&] { std::memcpy([metalInput contents], host.data(), size); }, options.repeats);
            runMetal(queue, metalNegative, @[metalInput, metalOutput], line, group);
            const double metalReadUs = measure([&] { std::memcpy(result.data(), [metalOutput contents], size); }, options.repeats);
            const double metalLaunchUs = measure([&] {
                runMetal(queue, metalEmpty, @[metalInput], MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
            }, options.repeats);
            std::vector<double> metalKernels;
            const double metalTotalUs = measure([&] {
                std::memcpy([metalInput contents], host.data(), size);
                metalKernels.push_back(runMetal(queue, metalNegative, @[metalInput, metalOutput], line, group));
                std::memcpy(result.data(), [metalOutput contents], size);
            }, options.repeats);
            checkNegative(host, result.data(), "Metal");
            std::memset(alignedOutput.data, 0, alignedOutput.size);
            id<MTLBuffer> noCopyInput = [device newBufferWithBytesNoCopy:alignedInput.data length:alignedInput.size
                                                                 options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> noCopyOutput = [device newBufferWithBytesNoCopy:alignedOutput.data length:alignedOutput.size
                                                                  options:MTLResourceStorageModeShared deallocator:nil];
            const double metalNoCopyUs = measure([&] {
                runMetal(queue, metalNegative, @[noCopyInput, noCopyOutput], line, group);
            }, options.repeats);
            checkNegative(host, alignedOutput.bytes(), "Metal without copying");

            std::ostringstream row;
            row << std::fixed << std::setprecision(3) << side << ',' << size << ',' << copyUs << ',' << clWriteUs << ',' << clReadUs << ','
                << clLaunchUs << ',' << clTotalUs << ',' << median(clKernels) << ',' << metalWriteUs << ',' << metalReadUs << ','
                << metalLaunchUs << ',' << metalTotalUs << ',' << median(metalKernels) << ',' << metalNoCopyUs << ',' << clHostPointerUs;
            appendCsv(options.output,
                      "side,bytes,memcpy_us,opencl_write_us,opencl_read_us,opencl_empty_launch_us,opencl_negative_total_us,"
                      "opencl_negative_kernel_us,metal_write_us,metal_read_us,metal_empty_launch_us,metal_negative_total_us,"
                      "metal_negative_kernel_us,metal_nocopy_negative_us,opencl_hostptr_negative_us",
                      row.str());
            std::cout << side << 'x' << side << ": memcpy " << copyUs << " us; OpenCL write " << clWriteUs << ", read " << clReadUs
                      << ", empty kernel " << clLaunchUs << ", negative " << clTotalUs << "; Metal write " << metalWriteUs << ", read "
                      << metalReadUs << ", empty kernel " << metalLaunchUs << ", negative " << metalTotalUs << ", without copying "
                      << metalNoCopyUs << std::endl;
        }

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
        const std::size_t threads = 1 << 16;
        const std::size_t local = 256;
        id<MTLBuffer> spinOutput = [device newBufferWithLength:threads * sizeof(float) options:MTLResourceStorageModeShared];
        cl_mem clSpinOutput = createBuffer(cl.context, CL_MEM_WRITE_ONLY, threads * sizeof(float));
        setArgument(clSpin, 0, clSpinOutput);
        for (int iterations : {2000, 20000, 100000}) {
            std::vector<double> hostTimes, commandTimes, counterTimes, clHostTimes, clEventTimes;
            for (int repeat = 0; repeat < 9; ++repeat) {
                @autoreleasepool {
                    MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
                    pass.sampleBufferAttachments[0].sampleBuffer = samples;
                    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
                    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
                    const auto start = std::chrono::steady_clock::now();
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDescriptor:pass];
                    [encoder setComputePipelineState:metalSpin];
                    [encoder setBuffer:spinOutput offset:0 atIndex:0];
                    [encoder setBytes:&iterations length:sizeof(int) atIndex:1];
                    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(local, 1, 1)];
                    [encoder endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                    const auto end = std::chrono::steady_clock::now();
                    NSData* resolved = [samples resolveCounterRange:NSMakeRange(0, 2)];
                    const auto* timestamps = static_cast<const MTLCounterResultTimestamp*>([resolved bytes]);
                    hostTimes.push_back(std::chrono::duration<double, std::micro>(end - start).count());
                    commandTimes.push_back(([command GPUEndTime] - [command GPUStartTime]) * 1e6);
                    counterTimes.push_back(static_cast<double>(timestamps[1].timestamp - timestamps[0].timestamp) / 1000.0);
                }
                setArgument(clSpin, 1, iterations);
                cl_event event = nullptr;
                const auto start = std::chrono::steady_clock::now();
                checkCL(clEnqueueNDRangeKernel(cl.queue, clSpin, 1, nullptr, &threads, &local, 0, nullptr, &event), "clEnqueueNDRangeKernel");
                checkCL(clFinish(cl.queue), "clFinish");
                const auto end = std::chrono::steady_clock::now();
                clHostTimes.push_back(std::chrono::duration<double, std::micro>(end - start).count());
                clEventTimes.push_back(kernelTimeUs(event));
                checkCL(clReleaseEvent(event), "clReleaseEvent");
            }
            std::ostringstream row;
            row << std::fixed << std::setprecision(1) << iterations << ',' << median(hostTimes) << ',' << median(commandTimes) << ','
                << median(counterTimes) << ',' << median(clHostTimes) << ',' << median(clEventTimes);
            appendCsv(unitsOutput, "iterations,host_us,command_buffer_us,counter_diff_div_1000,opencl_host_us,opencl_event_us", row.str());
            std::cout << "Long kernel, " << iterations << " iterations: host " << median(hostTimes) << " us, command buffer "
                      << median(commandTimes) << " us, Metal timestamps / 1000 " << median(counterTimes) << std::endl;
        }
        clReleaseMemObject(clSpinOutput);
        for (cl_kernel kernel : {clEmpty, clNegative, clSpin}) clReleaseKernel(kernel);
        clReleaseProgram(program);
        std::cout << "Saved " << options.output << " and " << unitsOutput << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
