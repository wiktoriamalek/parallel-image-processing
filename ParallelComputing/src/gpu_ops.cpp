// OpenCL wrapper, invert, blur/edge, hist
#include "gpu_ops.hpp"
#include "image_io.hpp"
#include <OpenCL/opencl.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <array>

// RAII-wrapper na kontekst OpenCL
struct ClCtx
{
    cl_platform_id  plat {};
    cl_device_id    dev  {};
    cl_context      ctx  {};
    cl_command_queue q   {};

    ClCtx()
    {
        cl_int err;
        clGetPlatformIDs(1, &plat, nullptr);
        clGetDeviceIDs  (plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
        ctx = clCreateContext      (nullptr, 1, &dev, nullptr, nullptr, &err);
        q   = clCreateCommandQueue (ctx, dev,CL_QUEUE_PROFILING_ENABLE, &err);
    }
    ~ClCtx()
    {
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
    }
};

static double kernelMS(cl_event ev)
{
    cl_ulong a = 0, b = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,sizeof(a), &a, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,sizeof(b), &b, nullptr);
    return (b - a) * 1e-6;
}

void runInvertGPU(const char* clFile,unsigned char* img, std::size_t pixels,double* outKernelMS)
{
    ClCtx cl;
    std::size_t bytes = pixels * 4;

    cl_int err = 0;
    cl_mem buf = clCreateBuffer(cl.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,bytes, img, &err);

    std::string src = readFile(clFile);
    const char* p   = src.c_str();
    std::size_t len = src.size();

    cl_program prog =clCreateProgramWithSource(cl.ctx, 1, &p, &len, &err);
    if(clBuildProgram(prog, 1, &cl.dev, nullptr, nullptr, nullptr)!= CL_SUCCESS)
    {
        size_t logSz = 0;
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,0, nullptr, &logSz);
        std::string log(logSz,'\0');
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,logSz, log.data(), nullptr);
        std::cerr << "Błąd kompilacji OpenCL:\n" << log << '\n';
        std::exit(1);
    }

    cl_kernel krn = clCreateKernel(prog, "invertRGBA", &err);
    clSetKernelArg(krn, 0, sizeof(cl_mem),  &buf);
    clSetKernelArg(krn, 1, sizeof(cl_uint), &pixels);

    size_t g = pixels;
    cl_event ev{};
    clEnqueueNDRangeKernel(cl.q, krn, 1, nullptr, &g, nullptr,0, nullptr, &ev);
    clFinish(cl.q);

    clEnqueueReadBuffer(cl.q, buf, CL_TRUE, 0, bytes, img,0, nullptr, nullptr);
    if(outKernelMS) *outKernelMS = kernelMS(ev);

    clReleaseEvent  (ev);
    clReleaseKernel (krn);
    clReleaseProgram(prog);
    clReleaseMemObject(buf);
}

// blur / edge
void runKernelWH(const char* clFile, const char* func,unsigned char* img, int w, int h,double* outKernelMS)
{
    ClCtx cl;
    std::size_t bytes = std::size_t(w) * h * 4;

    cl_int err = 0;
    cl_mem d_in  = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY  |CL_MEM_COPY_HOST_PTR,bytes, img, &err);
    cl_mem d_out = clCreateBuffer(cl.ctx, CL_MEM_WRITE_ONLY,bytes, nullptr, &err);

    std::string src = readFile(clFile);
    const char* p   = src.c_str();
    std::size_t len = src.size();

    cl_program prog =clCreateProgramWithSource(cl.ctx, 1, &p, &len, &err);
    
    if(clBuildProgram(prog, 1, &cl.dev, nullptr, nullptr, nullptr)!= CL_SUCCESS)
    {
        size_t logSz = 0;
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,0, nullptr, &logSz);
        std::string log(logSz,'\0');
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,logSz, log.data(), nullptr);
        std::cerr << "Błąd kompilacji OpenCL:\n" << log << '\n';
        std::exit(1);
    }

    cl_kernel krn = clCreateKernel(prog, func, &err);
    clSetKernelArg(krn, 0, sizeof(cl_mem),  &d_in);
    clSetKernelArg(krn, 1, sizeof(cl_mem),  &d_out);
    clSetKernelArg(krn, 2, sizeof(cl_uint), &w);
    clSetKernelArg(krn, 3, sizeof(cl_uint), &h);

    const size_t g[2] = { static_cast<size_t>(w),static_cast<size_t>(h) };
    cl_event ev{};
    clEnqueueNDRangeKernel(cl.q, krn, 2, nullptr, g, nullptr,0, nullptr, &ev);
    clFinish(cl.q);

    clEnqueueReadBuffer(cl.q, d_out, CL_TRUE, 0, bytes, img,0, nullptr, nullptr);
    if(outKernelMS) *outKernelMS = kernelMS(ev);

    clReleaseEvent  (ev);
    clReleaseKernel (krn);
    clReleaseProgram(prog);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
}

void runHistGPU(const char* clFile,const unsigned char* img, std::size_t pixels,uint64_t H[3][256],double* outKernelMS)
{
    ClCtx cl;
    std::size_t bytes = pixels * 4;

    cl_int err = 0;
    cl_mem d_img = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR,bytes,const_cast<unsigned char*>(img),&err);
    cl_mem d_R   = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE,256*sizeof(cl_uint), nullptr, &err);
    cl_mem d_G   = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE,256*sizeof(cl_uint), nullptr, &err);
    cl_mem d_B   = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE,256*sizeof(cl_uint), nullptr, &err);

    std::array<cl_uint,256> zero{};
    
    clEnqueueWriteBuffer(cl.q, d_R, CL_FALSE, 0, sizeof(zero),zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl.q, d_G, CL_FALSE, 0, sizeof(zero),zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl.q, d_B, CL_TRUE , 0, sizeof(zero),zero.data(), 0, nullptr, nullptr);

    std::string src = readFile(clFile);
    const char* p   = src.c_str();
    std::size_t len = src.size();

    cl_program prog =clCreateProgramWithSource(cl.ctx, 1, &p, &len, &err);
    if(clBuildProgram(prog, 1, &cl.dev, nullptr, nullptr, nullptr)!= CL_SUCCESS)
        
    {
        size_t logSz = 0;
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,0, nullptr, &logSz);
        std::string log(logSz,'\0');
        clGetProgramBuildInfo(prog, cl.dev, CL_PROGRAM_BUILD_LOG,logSz, log.data(), nullptr);
        std::cerr << "Błąd kompilacji OpenCL:\n" << log << '\n';
        std::exit(1);
    }

    cl_kernel krn = clCreateKernel(prog, "histRGB", &err);
    clSetKernelArg(krn, 0, sizeof(cl_mem),  &d_img);
    clSetKernelArg(krn, 1, sizeof(cl_uint), &pixels);
    clSetKernelArg(krn, 2, sizeof(cl_mem),  &d_R);
    clSetKernelArg(krn, 3, sizeof(cl_mem),  &d_G);
    clSetKernelArg(krn, 4, sizeof(cl_mem),  &d_B);

    size_t g = 8192;
    cl_event ev{};
    clEnqueueNDRangeKernel(cl.q, krn, 1, nullptr, &g, nullptr,0, nullptr, &ev);
    clFinish(cl.q);
    if(outKernelMS) *outKernelMS = kernelMS(ev);

    std::array<cl_uint,256> tmp;
    clEnqueueReadBuffer(cl.q, d_R, CL_TRUE, 0, sizeof(tmp),tmp.data(), 0, nullptr, nullptr);
    for(int b=0;b<256;++b) H[0][b] = tmp[b];

    clEnqueueReadBuffer(cl.q, d_G, CL_TRUE, 0, sizeof(tmp),tmp.data(), 0, nullptr, nullptr);
    for(int b=0;b<256;++b) H[1][b] = tmp[b];

    clEnqueueReadBuffer(cl.q, d_B, CL_TRUE, 0, sizeof(tmp),tmp.data(), 0, nullptr, nullptr);
    for(int b=0;b<256;++b) H[2][b] = tmp[b];

    clReleaseEvent  (ev);
    clReleaseKernel (krn);
    clReleaseProgram(prog);
    clReleaseMemObject(d_img);
    clReleaseMemObject(d_R);
    clReleaseMemObject(d_G);
    clReleaseMemObject(d_B);
}
