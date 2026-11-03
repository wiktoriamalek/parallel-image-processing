#include "region_growing.hpp"
#include "parallel_utils.hpp"
#include <stdexcept>
#include <queue>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <utility>

namespace {
constexpr int cpuTileSize = 128;
constexpr int gpuTileSize = 64;
constexpr int gpuGroupSize = 16;
constexpr int stepsBetweenChecks = 16;

void growTile(const Image& image, Image& mask, int seedValue, int tolerance, int startX, int startY, int endX, int endY,
              std::vector<int>& pending, std::vector<int>& queue, std::vector<int>& requests)
{
    int offsetX[4] = {-1, 1, 0, 0};
    int offsetY[4] = {0, 0, -1, 1};
    queue.clear();
    for (int index : pending)
    {
        if (mask.pixels[index] == 0 && std::abs(image.pixels[index] - seedValue) <= tolerance)
        {
            mask.pixels[index] = 255;
            queue.push_back(index);
        }
    }
    pending.clear();

    for (std::size_t head = 0; head < queue.size(); head++)
    {
        int currentX = queue[head] % image.width;
        int currentY = queue[head] / image.width;
        for (int direction = 0; direction < 4; direction++)
        {
            int neighbourX = currentX + offsetX[direction];
            int neighbourY = currentY + offsetY[direction];
            if (neighbourX < 0 || neighbourX >= image.width || neighbourY < 0 || neighbourY >= image.height) {continue;}
            int neighbourIndex = neighbourY * image.width + neighbourX;
            if (std::abs(image.pixels[neighbourIndex] - seedValue) > tolerance) {continue;}
            if (neighbourX < startX || neighbourX >= endX || neighbourY < startY || neighbourY >= endY)
            {
                requests.push_back(neighbourIndex);
            }
            else if (mask.pixels[neighbourIndex] == 0)
            {
                mask.pixels[neighbourIndex] = 255;
                queue.push_back(neighbourIndex);
            }
        }
    }
}
}

Image regionGrowingCPU(const Image& image, int seedX, int seedY, int tolerance)
{
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("regionGrowingCPU: invalid arguments");
    }

    Image mask;
    mask.width = image.width;
    mask.height = image.height;
    mask.pixels.resize((std::size_t)mask.width * mask.height);
    int seedIndex = seedY * image.width + seedX;
    int seedValue = image.pixels[seedIndex];
    mask.pixels[seedIndex] = 255;
    std::queue<int> pending;
    pending.push(seedIndex);
    int offsetX[4] = {-1 ,1, 0, 0};
    int offsetY[4] = { 0, 0, -1, 1};

    while (!pending.empty())
    {   // queue is a list of pixels that we yet need to check
        int currentIndex = pending.front(); // pixel A
        pending.pop(); // delete
        int currentX = currentIndex % image.width; // column
        int currentY = currentIndex / image.width; // row
        // if pixel B matched the spot, we want to add it to the queue and check the neighbours again
        for (int direction = 0; direction < 4; direction++)
        {
            int neighbourX = currentX + offsetX[direction];
            int neighbourY = currentY + offsetY[direction];
            if(neighbourX < 0 || neighbourX>=image.width || neighbourY < 0 || neighbourY>=image.height){continue;}
            int neighbourIndex = neighbourY * image.width + neighbourX;
            if (mask.pixels[neighbourIndex] != 0){continue;}
            int neighbourValue = image.pixels[neighbourIndex];
            int difference = std::abs(neighbourValue - seedValue);
            if (difference <= tolerance)
            {
                mask.pixels[neighbourIndex] = 255;
                pending.push(neighbourIndex);
            }
        }
    }

return mask;
}

Image regionGrowingCPUParallel(const Image& image, int seedX, int seedY, int tolerance, int threads)
{
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255 || threads <= 0)
    {
        throw std::invalid_argument("regionGrowingCPUParallel: invalid arguments");
    }

    Image mask;
    mask.width = image.width;
    mask.height = image.height;
    mask.pixels.resize((std::size_t)mask.width * mask.height);
    int seedIndex = seedY * image.width + seedX;
    int seedValue = image.pixels[seedIndex];
    int tilesX = (image.width + cpuTileSize - 1) / cpuTileSize;
    int tilesY = (image.height + cpuTileSize - 1) / cpuTileSize;
    auto tileOf = [&](int index) {
        return (index / image.width) / cpuTileSize * tilesX + (index % image.width) / cpuTileSize;
    };
    std::vector<std::vector<int>> pending(static_cast<std::size_t>(tilesX) * tilesY);
    std::vector<std::vector<int>> queues(threads);
    std::vector<std::vector<int>> requests(threads);
    std::vector<int> activeTiles;
    pending[tileOf(seedIndex)].push_back(seedIndex);

    while (true)
    {
        activeTiles.clear();
        for (int tile = 0; tile < static_cast<int>(pending.size()); tile++)
        {
            if (!pending[tile].empty()) {activeTiles.push_back(tile);}
        }
        if (activeTiles.empty()) {break;}

        int threadCount = std::min(threads, static_cast<int>(activeTiles.size()));
        std::atomic<std::size_t> nextTile{0};
        parallelRanges(threadCount, threadCount, [&](std::size_t, std::size_t, int thread) {
            for (std::size_t i = nextTile++; i < activeTiles.size(); i = nextTile++)
            {
                int tile = activeTiles[i];
                int startX = tile % tilesX * cpuTileSize;
                int startY = tile / tilesX * cpuTileSize;
                growTile(image, mask, seedValue, tolerance, startX, startY,
                         std::min(startX + cpuTileSize, image.width), std::min(startY + cpuTileSize, image.height),
                         pending[tile], queues[thread], requests[thread]);
            }
        });

        for (auto& threadRequests : requests)
        {
            for (int index : threadRequests) {pending[tileOf(index)].push_back(index);}
            threadRequests.clear();
        }
    }
    return mask;
}

RegionGrowingGPU::RegionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance, bool localTiles)
    : cl("kernels/region_growing.cl"),
      localTiles(localTiles),
      bufferSize(image.pixels.size() * sizeof(unsigned char)),
      globalWorkSize{static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height)},
      inputWidth(image.width),
      inputHeight(image.height)
{
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("RegionGrowingGPU: invalid arguments");
    }
    seedIndex = seedY * image.width + seedX;
    tilesX = (image.width + gpuTileSize - 1) / gpuTileSize;
    tilesY = (image.height + gpuTileSize - 1) / gpuTileSize;
    tileCount = static_cast<std::size_t>(tilesX) * tilesY;
    tileGlobalWorkSize[0] = static_cast<std::size_t>(tilesX) * gpuGroupSize;
    tileGlobalWorkSize[1] = static_cast<std::size_t>(tilesY) * gpuGroupSize;
    tileLocalWorkSize[0] = gpuGroupSize;
    tileLocalWorkSize[1] = gpuGroupSize;
    int seedTile = (seedY / gpuTileSize) * tilesX + seedX / gpuTileSize;
    int tileCountArgument = static_cast<int>(tileCount);
    try {
        cl_int error;
        inputBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        maskBufferA = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
        checkCL(error, "clCreateBuffer");
        changedBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        checkCL(error, "clCreateBuffer");

        initKernel = clCreateKernel(cl.program, "region_init", &error);
        checkCL(error, "clCreateKernel");
        checkCL(clSetKernelArg(initKernel, 0, sizeof(cl_mem), &maskBufferA), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 1, sizeof(int), &inputWidth), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 2, sizeof(int), &inputHeight), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 3, sizeof(int), &seedX), "clSetKernelArg");
        checkCL(clSetKernelArg(initKernel, 4, sizeof(int), &seedY), "clSetKernelArg");

        if (localTiles) {
            activeBufferA = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, tileCount * sizeof(cl_int), nullptr, &error);
            checkCL(error, "clCreateBuffer");
            activeBufferB = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, tileCount * sizeof(cl_int), nullptr, &error);
            checkCL(error, "clCreateBuffer");

            activeInitKernel = clCreateKernel(cl.program, "region_active_init", &error);
            checkCL(error, "clCreateKernel");
            tileKernel = clCreateKernel(cl.program, "region_tile_step", &error);
            checkCL(error, "clCreateKernel");

            checkCL(clSetKernelArg(activeInitKernel, 0, sizeof(cl_mem), &activeBufferA), "clSetKernelArg");
            checkCL(clSetKernelArg(activeInitKernel, 1, sizeof(int), &tileCountArgument), "clSetKernelArg");
            checkCL(clSetKernelArg(activeInitKernel, 2, sizeof(int), &seedTile), "clSetKernelArg");

            checkCL(clSetKernelArg(tileKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 1, sizeof(cl_mem), &maskBufferA), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 2, sizeof(cl_mem), &changedBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 5, sizeof(int), &inputWidth), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 6, sizeof(int), &inputHeight), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 8, sizeof(int), &tolerance), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 9, sizeof(int), &tilesX), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 10, sizeof(int), &tilesY), "clSetKernelArg");
        } else {
            maskBufferB = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bufferSize, nullptr, &error);
            checkCL(error, "clCreateBuffer");

            stepKernel = clCreateKernel(cl.program, "region_step", &error);
            checkCL(error, "clCreateKernel");

            checkCL(clSetKernelArg(stepKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 1, sizeof(cl_mem), &maskBufferA), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 2, sizeof(cl_mem), &maskBufferB), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 3, sizeof(cl_mem), &changedBuffer), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 4, sizeof(int), &inputWidth), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 5, sizeof(int), &inputHeight), "clSetKernelArg");
            checkCL(clSetKernelArg(stepKernel, 7, sizeof(int), &tolerance), "clSetKernelArg");
        }

    } catch (...) {
        release();
        throw;
    }
}

void RegionGrowingGPU::release() {
    for (cl_event event : kernelEvents) clReleaseEvent(event);
    kernelEvents.clear();
    if (initKernel) clReleaseKernel(initKernel);
    if (stepKernel) clReleaseKernel(stepKernel);
    if (activeInitKernel) clReleaseKernel(activeInitKernel);
    if (tileKernel) clReleaseKernel(tileKernel);
    if (inputBuffer) clReleaseMemObject(inputBuffer);
    if (maskBufferA) clReleaseMemObject(maskBufferA);
    if (maskBufferB) clReleaseMemObject(maskBufferB);
    if (activeBufferA) clReleaseMemObject(activeBufferA);
    if (activeBufferB) clReleaseMemObject(activeBufferB);
    if (changedBuffer) clReleaseMemObject(changedBuffer);

}

RegionGrowingGPU::~RegionGrowingGPU() {
    release();
}

void RegionGrowingGPU::launch(cl_kernel kernel, cl_uint dimensions, const std::size_t* global, const std::size_t* local, double* kernelUs) {
    cl_event event = nullptr;
    checkCL(clEnqueueNDRangeKernel(cl.queue, kernel, dimensions, nullptr, global, local, 0, nullptr, kernelUs ? &event : nullptr), "clEnqueueNDRangeKernel");
    if (kernelUs) kernelEvents.push_back(event);
}

void RegionGrowingGPU::addKernelTimes(double* kernelUs) {
    for (cl_event event : kernelEvents) {
        *kernelUs += kernelTimeUs(event);
        checkCL(clReleaseEvent(event), "clReleaseEvent");
    }
    kernelEvents.clear();
}

void RegionGrowingGPU::run(const Image& image, Image& output, double* kernelUs) {
    if (static_cast<std::size_t>(output.width) != globalWorkSize[0] || static_cast<std::size_t>(output.height) != globalWorkSize[1] ||
        image.width != inputWidth || image.height != inputHeight ||
        image.pixels.size() != bufferSize || output.pixels.size() != bufferSize) {
        throw std::runtime_error("Image dimensions do not match the GPU buffers");
    }
    if (kernelUs) *kernelUs = 0.0;
    int seedValue = image.pixels[seedIndex];
    if (localTiles) {
        checkCL(clSetKernelArg(tileKernel, 7, sizeof(int), &seedValue), "clSetKernelArg");
    } else {
        checkCL(clSetKernelArg(stepKernel, 6, sizeof(int), &seedValue), "clSetKernelArg");
    }

    checkCL(clEnqueueWriteBuffer(cl.queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    launch(initKernel, 2, globalWorkSize, nullptr, kernelUs);
    checkCL(clFinish(cl.queue), "clFinish");

    if (localTiles) runTiles(kernelUs);
    else runPixelSteps(kernelUs);

    checkCL(clEnqueueReadBuffer(cl.queue, resultMask, CL_TRUE, 0, bufferSize, output.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer"); // CL_TRUE means: dont return from clEnqueueReadBuffer until the readout is done
}

void RegionGrowingGPU::runPixelSteps(double* kernelUs) {
    cl_mem previousMask = maskBufferA;
    cl_mem nextMask = maskBufferB;
    cl_int changed = 1;
    iterationCount = 0;

    while (changed!=0)
    {
        changed = 0;
        checkCL(clEnqueueWriteBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        checkCL(clSetKernelArg(stepKernel, 1, sizeof(cl_mem), &previousMask), "clSetKernelArg");
        checkCL(clSetKernelArg(stepKernel, 2, sizeof(cl_mem), &nextMask), "clSetKernelArg");
        launch(stepKernel, 2, globalWorkSize, nullptr, kernelUs);
        checkCL(clEnqueueReadBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        addKernelTimes(kernelUs);
        std::swap(previousMask, nextMask);
        iterationCount++;
    }
    resultMask = previousMask;
}

void RegionGrowingGPU::runTiles(double* kernelUs) {
    launch(activeInitKernel, 1, &tileCount, nullptr, kernelUs);
    cl_mem previousActive = activeBufferA;
    cl_mem nextActive = activeBufferB;
    cl_int changed = 1;
    iterationCount = 0;

    while (changed != 0)
    {
        changed = 0;
        checkCL(clEnqueueWriteBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        for (int step = 0; step < stepsBetweenChecks; step++)
        {
            checkCL(clSetKernelArg(tileKernel, 3, sizeof(cl_mem), &previousActive), "clSetKernelArg");
            checkCL(clSetKernelArg(tileKernel, 4, sizeof(cl_mem), &nextActive), "clSetKernelArg");
            launch(tileKernel, 2, tileGlobalWorkSize, tileLocalWorkSize, kernelUs);
            std::swap(previousActive, nextActive);
            iterationCount++;
        }
        checkCL(clEnqueueReadBuffer(cl.queue, changedBuffer, CL_TRUE, 0, sizeof(cl_int), &changed, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        addKernelTimes(kernelUs);
    }
    resultMask = maskBufferA;
}

Image regionGrowingGPU(const Image& image, int seedX, int seedY, int tolerance, bool localTiles) {
    if (image.width <= 0 || image.height <= 0 || seedX < 0 || seedX >=image.width || seedY <0 || seedY >=image.height|| tolerance < 0 || tolerance > 255)
    {
        throw std::invalid_argument("regionGrowingGPU: invalid arguments");
    }
    Image output;
    output.width = image.width;
    output.height = image.height;
    output.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    RegionGrowingGPU gpu(image, seedX, seedY, tolerance, localTiles);
    gpu.run(image, output);
    return output;
}
