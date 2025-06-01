#pragma once

#include <cstddef>

#include <cstdint>

void invertCPU(unsigned char* img, std::size_t pixels);

void boxBlurCPU (const unsigned char* src, unsigned char* dst,int w, int h);

void boxBlurCPUMT(const unsigned char* src, unsigned char* dst,int w, int h, int nThreads);

void sobelCPU (const unsigned char* src, unsigned char* dst,int w, int h);

void sobelCPUMT(const unsigned char* src, unsigned char* dst,int w, int h, int nThreads);

void histCPU (const unsigned char* img, std::size_t pixels,uint64_t H[3][256]);

void histCPUMT(const unsigned char* img, std::size_t pixels,uint64_t H[3][256], int nThreads);
