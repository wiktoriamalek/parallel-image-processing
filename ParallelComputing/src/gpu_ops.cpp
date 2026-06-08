#include "gpu_ops.hpp"
#include "image_io.hpp"
#include <OpenCL/opencl.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>

struct SimpleClContext { //obiekty opencl
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
};

static SimpleClContext makeContext() {
    SimpleClContext ctx;
    cl_int err; //dla kodu bledu
    clGetPlatformIDs(1, &ctx.platform, nullptr);
    clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, nullptr); //pierwszy potrzebny
    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, CL_QUEUE_PROFILING_ENABLE, &err);
    return ctx;
}

static double measureKernelTime(cl_event ev) {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr); //funkcja api opencl
    return (end - start) * 1e-6;
}

static cl_program buildProgram(cl_context ctx, cl_device_id dev, const char* srcFile) {
    std::string src = readFile(srcFile); //kod zrodlowy kernela
    const char* code = src.c_str();
    size_t len = src.length();
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &code, &len, &err);

    if (clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0'); //najwyzej blad6
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Błąd budowania programu OpenCL:\n" << log << std::endl;
        std::exit(1);
    }

    return prog;
}

void runInvertGPU(const char* clFile, unsigned char* img, std::size_t pix, double* outTime) {
    auto ctx = makeContext();
    size_t bytes = pix * 4;
    cl_int err;

    cl_mem buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, img, &err);
    cl_program prog = buildProgram(ctx.context, ctx.device, clFile);
    cl_kernel kernel = clCreateKernel(prog, "invertRGBA", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_uint), &pix);

    size_t global = pix;
    cl_event ev;
    clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &ev);
    clFinish(ctx.queue);

    clEnqueueReadBuffer(ctx.queue, buffer, CL_TRUE, 0, bytes, img, 0, nullptr, nullptr);
    if (outTime) *outTime = measureKernelTime(ev);

    clReleaseEvent(ev);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
}

void runKernelWH(const char* clFile, const char* func, unsigned char* img, int w, int h, double* outTime) {
    auto ctx = makeContext();
    size_t bytes = w * h * 4;
    cl_int err;

    cl_mem in = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, img, &err);
    cl_mem out = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    cl_program prog = buildProgram(ctx.context, ctx.device, clFile);
    cl_kernel kernel = clCreateKernel(prog, func, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &w);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &h);

    size_t global[2] = { (size_t)w, (size_t)h };
    cl_event ev;
    clEnqueueNDRangeKernel(ctx.queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, &ev);
    clFinish(ctx.queue);

    clEnqueueReadBuffer(ctx.queue, out, CL_TRUE, 0, bytes, img, 0, nullptr, nullptr);
    if (outTime) *outTime = measureKernelTime(ev);

    clReleaseEvent(ev);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseMemObject(in);
    clReleaseMemObject(out);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
}

void run_histogram_gpu(const char* clFile, const unsigned char* img, std::size_t pix, uint64_t H[3][256], double* outTime) {
    auto ctx = makeContext();
    size_t bytes = pix * 4;
    cl_int err;

    cl_mem input = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)img, &err);
    cl_mem r = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &err);
    cl_mem g = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &err);
    cl_mem b = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, 256 * sizeof(cl_uint), nullptr, &err);

    std::array<cl_uint, 256> zero{};
    clEnqueueWriteBuffer(ctx.queue, r, CL_FALSE, 0, sizeof(zero), zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctx.queue, g, CL_FALSE, 0, sizeof(zero), zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctx.queue, b, CL_TRUE,  0, sizeof(zero), zero.data(), 0, nullptr, nullptr);

    cl_program prog = buildProgram(ctx.context, ctx.device, clFile);
    cl_kernel kernel = clCreateKernel(prog, "histRGB", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input); //obraz wejsciowy w postaci uchwytu do bufora pamieci na gpu
    clSetKernelArg(kernel, 1, sizeof(cl_uint), &pix); //liczba pikseli
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &r);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &g);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &b);

    size_t global = 8192;
    cl_event ev;
    clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &ev);
    clFinish(ctx.queue);
    if (outTime) *outTime = measureKernelTime(ev);

    std::array<cl_uint, 256> tmp;
    clEnqueueReadBuffer(ctx.queue, r, CL_TRUE, 0, sizeof(tmp), tmp.data(), 0, nullptr, nullptr);
    for (int i = 0; i < 256; ++i) H[0][i] = tmp[i]; //przechowuje r g b
    clEnqueueReadBuffer(ctx.queue, g, CL_TRUE, 0, sizeof(tmp), tmp.data(), 0, nullptr, nullptr);
    for (int i = 0; i < 256; ++i) H[1][i] = tmp[i];
    clEnqueueReadBuffer(ctx.queue, b, CL_TRUE, 0, sizeof(tmp), tmp.data(), 0, nullptr, nullptr);
    for (int i = 0; i < 256; ++i) H[2][i] = tmp[i];

    clReleaseEvent(ev);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseMemObject(input);
    clReleaseMemObject(r);
    clReleaseMemObject(g);
    clReleaseMemObject(b);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
}
