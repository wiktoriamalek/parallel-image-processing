#pragma once
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstddef>
#include <string>
#include <vector>

void checkCL(cl_int error, const char* operation);
double kernelTimeUs(cl_event event);

constexpr std::size_t lineGroupSize = 256;
constexpr std::size_t tileGroupSize = 16;
std::size_t roundUp(std::size_t value, std::size_t multiple);
void enqueueLine(cl_command_queue queue, cl_kernel kernel, std::size_t count, cl_event* event = nullptr);
void enqueueTiles(cl_command_queue queue, cl_kernel kernel, std::size_t width, std::size_t height, cl_event* event = nullptr);

class OpenCL {
public:
    explicit OpenCL(const std::string& kernelPath, const std::string& sourcePrefix = "");
    explicit OpenCL(const std::vector<std::string>& kernelPaths, const std::string& sourcePrefix = "");
    ~OpenCL();
    OpenCL(const OpenCL&) = delete;
    OpenCL& operator=(const OpenCL&) = delete;

    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

private:
    void release();
};
