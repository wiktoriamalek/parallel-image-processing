#include "opencl_utils.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

void checkCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

double kernelTimeUs(cl_event event) {
    cl_ulong start, end;
    checkCL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "clGetEventProfilingInfo");
    checkCL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "clGetEventProfilingInfo");
    double nanoseconds = static_cast<double>(end - start);
#ifdef __APPLE__
    // Apple's OpenCL returns these timestamps in mach_absolute_time ticks, not in nanoseconds as the standard says;
    // on Apple Silicon one tick is 125/3 = 41.67 ns, so without this conversion kernel times were about 42 times too small
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    nanoseconds = nanoseconds * timebase.numer / timebase.denom;
#endif
    return nanoseconds / 1000.0;
}

std::size_t roundUp(std::size_t value, std::size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

void enqueueLine(cl_command_queue queue, cl_kernel kernel, std::size_t count, cl_event* event) {
    const std::size_t global = roundUp(count, lineGroupSize);
    const std::size_t local = lineGroupSize;
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, event), "clEnqueueNDRangeKernel");
}

void enqueueTiles(cl_command_queue queue, cl_kernel kernel, std::size_t width, std::size_t height, cl_event* event) {
    const std::size_t global[2] = {roundUp(width, tileGroupSize), roundUp(height, tileGroupSize)};
    const std::size_t local[2] = {tileGroupSize, tileGroupSize};
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, event), "clEnqueueNDRangeKernel");
}

OpenCL::OpenCL(const std::string& kernelPath, const std::string& sourcePrefix)
    : OpenCL(std::vector<std::string>{kernelPath}, sourcePrefix) {}

OpenCL::OpenCL(const std::vector<std::string>& kernelPaths, const std::string& sourcePrefix) {
    try {
        cl_uint count = 0;
        checkCL(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
        if (count == 0) throw std::runtime_error("No OpenCL platform found");
        std::vector<cl_platform_id> platforms(count);
        checkCL(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
        for (auto platform : platforms) {
            cl_int error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            if (error == CL_DEVICE_NOT_FOUND) continue;
            checkCL(error, "clGetDeviceIDs");
            break;
        }
        if (!device) throw std::runtime_error("No OpenCL GPU found");

        cl_int error;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        checkCL(error, "clCreateContext");
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
        checkCL(error, "clCreateCommandQueue");

        std::vector<std::string> sources;
        if (!sourcePrefix.empty()) sources.push_back(sourcePrefix + '\n');
        std::string names;
        for (const std::string& kernelPath : kernelPaths) {
            std::ifstream file(kernelPath);
            if (!file) throw std::runtime_error("Could not open " + kernelPath);
            std::ostringstream sourceStream;
            sourceStream << file.rdbuf() << '\n';
            sources.push_back(sourceStream.str());
            names += (names.empty() ? "" : ", ") + kernelPath;
        }
        std::vector<const char*> texts;
        for (const std::string& source : sources) texts.push_back(source.c_str());
        program = clCreateProgramWithSource(context, static_cast<cl_uint>(texts.size()), texts.data(), nullptr, &error);
        checkCL(error, "clCreateProgramWithSource");
        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // OpenCL compile this program to the device - mac
        if (error != CL_SUCCESS) {
            std::size_t size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            if (size) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            throw std::runtime_error("Could not build " + names + ": " + log);
        }
    } catch (...) {
        release();
        throw;
    }
}

void OpenCL::release() {
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

OpenCL::~OpenCL() {
    release();
}
