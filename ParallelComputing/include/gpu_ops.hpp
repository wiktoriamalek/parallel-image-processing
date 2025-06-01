#pragma once
#include <cstddef>
#include <cstdint>

void runInvertGPU(const char* clFile,unsigned char* img, std::size_t pixels,double* kernelMS = nullptr);

void runKernelWH(const char* clFile, const char* funcName,unsigned char* img, int w,int h,double* kernelMS = nullptr);

void runHistGPU(const char* clFile,const unsigned char* img, std::size_t pixels,uint64_t H[3][256],double* kernelMS = nullptr);
