#include "otsu.hpp"
#include "image_io.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>

double median(std::vector<long long> values) {
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static void checkOpenCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path imagePath = argc > 1 ? argv[1] : "images/cms_resolution_change/cms_256.png";
        std::string imageName = imagePath.stem().string();
        Image image = loadImage(imagePath.string());

        Histogram histogram = calculateHistogram(image);
        Histogram histogram2 = calculateHistogram2Threads(image);
        Histogram histogramGPU = calculateHistogramGPU(image);

        if (histogramGPU != histogram) {
            throw std::runtime_error("GPU histogram differs from CPU!");
        }
        if (histogram2 != histogram) {
            throw std::runtime_error("CPU 1 and 2 threads histograms differ!");
        }

        std::vector<long long> cpu1Times;
        std::vector<long long> cpu2Times;
        std::vector<long long> gpuAll;
        std::vector<long long> gpuKernelOnly;
        Image segmented;
        Image segmented2Threads;

        for (int repeat = 0; repeat < 100; repeat++) {
            auto start = std::chrono::steady_clock::now();
            segmented = otsuSegmentation(image);
            auto end = std::chrono::steady_clock::now();
            cpu1Times.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            );
        }

        for (int repeat = 0; repeat < 100; repeat++) {
            auto start = std::chrono::steady_clock::now();
            segmented2Threads = otsuSegmentation2Threads(image);
            auto end = std::chrono::steady_clock::now();
            cpu2Times.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            );
            if (segmented2Threads.pixels != segmented.pixels) {
                throw std::runtime_error("CPU 1 and 2 threads masks differ!");
            }
        }

        Image segmentedGPU = otsuSegmentationGPU(image);
        if (segmentedGPU.pixels != segmented.pixels) {
            throw std::runtime_error("GPU segmentation differs from CPU!");
        }

        cl_uint platformCount = 0;
        checkOpenCL(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs");
        std::vector<cl_platform_id> platforms(platformCount);
        checkOpenCL(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_device_id device = nullptr;
        for (auto platform : platforms) {
            cl_uint count = 0;
            cl_int error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
            if (error == CL_DEVICE_NOT_FOUND) {
                continue;
            }
            checkOpenCL(error, "clGetDeviceIDs");
            if (count > 0) {
                checkOpenCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");
                break;
            }
        }
        if (!device) {
            throw std::runtime_error("No OpenCL GPU found");
        }

        cl_int error;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        checkOpenCL(error, "clCreateContext");
        cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
        checkOpenCL(error, "clCreateCommandQueue");

        std::size_t bufferSize = image.pixels.size() * sizeof(unsigned char);
        std::array<cl_uint, 256> gpuHistogram{};
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &error);
        checkOpenCL(error, "clCreateBuffer");
        cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(gpuHistogram), nullptr, &error);
        checkOpenCL(error, "clCreateBuffer");
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &error);
        checkOpenCL(error, "clCreateBuffer");

        std::ifstream kernelFile("kernels/otsu.cl");
        if (!kernelFile) {
            throw std::runtime_error("Could not open kernels/otsu.cl");
        }
        std::stringstream kernelStream;
        kernelStream << kernelFile.rdbuf();
        std::string kernelSource = kernelStream.str();
        const char* source = kernelSource.c_str();
        cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
        checkOpenCL(error, "clCreateProgramWithSource");
        checkOpenCL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");
        cl_kernel histogramKernel = clCreateKernel(program, "histogramKernel", &error);
        checkOpenCL(error, "clCreateKernel");
        cl_kernel thresholdKernel = clCreateKernel(program, "thresholdKernel", &error);
        checkOpenCL(error, "clCreateKernel");

        int Npixels = static_cast<int>(image.pixels.size());
        std::size_t globalWorkSize = image.pixels.size();
        checkOpenCL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkOpenCL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer), "clSetKernelArg");
        checkOpenCL(clSetKernelArg(histogramKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        checkOpenCL(clSetKernelArg(thresholdKernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg");
        checkOpenCL(clSetKernelArg(thresholdKernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg");
        checkOpenCL(clSetKernelArg(thresholdKernel, 2, sizeof(int), &Npixels), "clSetKernelArg");
        Image gpuResult = image;

        for (int repeat = 0; repeat < 100; repeat++) {
            cl_event histogramEvent;
            cl_event thresholdEvent;
            gpuHistogram.fill(0);

            auto gpuTotalStart = std::chrono::steady_clock::now();
            checkOpenCL(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bufferSize, image.pixels.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkOpenCL(clEnqueueWriteBuffer(queue, histogramBuffer, CL_TRUE, 0, sizeof(gpuHistogram), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
            checkOpenCL(clEnqueueNDRangeKernel(queue, histogramKernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, &histogramEvent), "clEnqueueNDRangeKernel");
            checkOpenCL(clEnqueueReadBuffer(queue, histogramBuffer, CL_TRUE, 0, sizeof(gpuHistogram), gpuHistogram.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");

            Histogram currentHistogram{};
            std::copy(gpuHistogram.begin(), gpuHistogram.end(), currentHistogram.begin());
            int threshold = calculateOtsuThreshold(currentHistogram);
            checkOpenCL(clSetKernelArg(thresholdKernel, 3, sizeof(int), &threshold), "clSetKernelArg");
            checkOpenCL(clEnqueueNDRangeKernel(queue, thresholdKernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, &thresholdEvent), "clEnqueueNDRangeKernel");
            checkOpenCL(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, gpuResult.pixels.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
            auto gpuTotalEnd = std::chrono::steady_clock::now();

            cl_ulong histogramStart, histogramEnd, thresholdStart, thresholdEnd;
            checkOpenCL(clGetEventProfilingInfo(histogramEvent, CL_PROFILING_COMMAND_START, sizeof(histogramStart), &histogramStart, nullptr), "clGetEventProfilingInfo");
            checkOpenCL(clGetEventProfilingInfo(histogramEvent, CL_PROFILING_COMMAND_END, sizeof(histogramEnd), &histogramEnd, nullptr), "clGetEventProfilingInfo");
            checkOpenCL(clGetEventProfilingInfo(thresholdEvent, CL_PROFILING_COMMAND_START, sizeof(thresholdStart), &thresholdStart, nullptr), "clGetEventProfilingInfo");
            checkOpenCL(clGetEventProfilingInfo(thresholdEvent, CL_PROFILING_COMMAND_END, sizeof(thresholdEnd), &thresholdEnd, nullptr), "clGetEventProfilingInfo");
            checkOpenCL(clReleaseEvent(histogramEvent), "clReleaseEvent");
            checkOpenCL(clReleaseEvent(thresholdEvent), "clReleaseEvent");

            gpuAll.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(gpuTotalEnd - gpuTotalStart).count()
            );
            gpuKernelOnly.push_back((histogramEnd - histogramStart) + (thresholdEnd - thresholdStart));

            if (currentHistogram != histogram) {
                throw std::runtime_error("GPU histogram differs from CPU!");
            }
            if (gpuResult.pixels != segmented.pixels) {
                throw std::runtime_error("GPU segmentation differs from CPU!");
            }
        }

        checkOpenCL(clReleaseKernel(histogramKernel), "clReleaseKernel");
        checkOpenCL(clReleaseKernel(thresholdKernel), "clReleaseKernel");
        checkOpenCL(clReleaseProgram(program), "clReleaseProgram");
        checkOpenCL(clReleaseMemObject(inputBuffer), "clReleaseMemObject");
        checkOpenCL(clReleaseMemObject(histogramBuffer), "clReleaseMemObject");
        checkOpenCL(clReleaseMemObject(outputBuffer), "clReleaseMemObject");
        checkOpenCL(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
        checkOpenCL(clReleaseContext(context), "clReleaseContext");

        double medianCPU1 = median(cpu1Times);
        double medianCPU2 = median(cpu2Times);
        double medianGPUAll = median(gpuAll);
        double medianGPUKernel = median(gpuKernelOnly) / 1000.0;
        int threshold = calculateOtsuThreshold(histogram);

        std::cout << "GPU and CPU segmentation are identical!\n";
        std::cout << "Median time taken on single-thread CPU: " << medianCPU1 << " us\n";
        std::cout << "Median time taken on 2-threads on CPU: " << medianCPU2 << " us\n";
        std::cout << "Median GPU total (from writing to reading after kernel operations): " << medianGPUAll << " us\n";
        std::cout << "Median GPU kernel only: " << medianGPUKernel << " us\n";

        std::filesystem::path outputDirectory = "results/otsu";
        std::filesystem::create_directories(outputDirectory);

        saveImage((outputDirectory / (imageName + "_otsu_cpu_1t.png")).string(), segmented);
        saveImage((outputDirectory / (imageName + "_otsu_cpu_2t.png")).string(), segmented2Threads);
        saveImage((outputDirectory / (imageName + "_otsu_gpu.png")).string(), gpuResult);

        std::ofstream histogramFile(outputDirectory / (imageName + "_histogram.csv"));
        if (!histogramFile) {
            throw std::runtime_error("Could not open histogram CSV");
        }
        histogramFile << "intensity,cpu_1t_count,cpu_2t_count,gpu_count\n";
        for (std::size_t intensity = 0; intensity < histogram.size(); intensity++) {
            histogramFile << intensity << ","
                          << histogram[intensity] << ","
                          << histogram2[intensity] << ","
                          << histogramGPU[intensity] << "\n";
        }
        histogramFile.close();
        if (!histogramFile) {
            throw std::runtime_error("Could not write histogram CSV");
        }

        std::filesystem::path csvPath = "results/otsu_benchmarks.csv";
        std::string header = "image,width,height,threshold,cpu_1t_us,cpu_2t_us,gpu_kernel_us,gpu_total_us";
        std::ostringstream rowKey;
        rowKey << std::quoted(imageName, '"', '"') << "," << image.width << "," << image.height << ",";
        std::string quotedKey = rowKey.str();
        std::string unquotedKey = imageName + "," + std::to_string(image.width) + "," + std::to_string(image.height) + ",";

        std::ostringstream newRow;
        newRow << std::setprecision(12)
               << quotedKey << threshold << ","
               << medianCPU1 << "," << medianCPU2 << ","
               << medianGPUKernel << "," << medianGPUAll;

        std::vector<std::string> rows;
        bool updated = false;
        if (std::filesystem::exists(csvPath)) {
            std::ifstream existingFile(csvPath);
            if (!existingFile) {
                throw std::runtime_error("Could not read benchmark CSV");
            }
            std::string line;
            if (std::getline(existingFile, line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (line != header) {
                    throw std::runtime_error("Unexpected benchmark CSV header");
                }
            }
            while (std::getline(existingFile, line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }
                if (line.rfind(quotedKey, 0) == 0 || line.rfind(unquotedKey, 0) == 0) {
                    if (!updated) {
                        rows.push_back(newRow.str());
                        updated = true;
                    }
                } else {
                    rows.push_back(line);
                }
            }
            if (existingFile.bad()) {
                throw std::runtime_error("Could not read benchmark CSV");
            }
        }
        if (!updated) {
            rows.push_back(newRow.str());
        }

        std::filesystem::path temporaryPath = csvPath.string() + ".tmp";
        std::ofstream file(temporaryPath);
        if (!file) {
            throw std::runtime_error("Could not open benchmark CSV");
        }
        file << header << "\n";
        for (const auto& row : rows) {
            file << row << "\n";
        }
        file.close();
        if (!file) {
            throw std::runtime_error("Could not write benchmark CSV");
        }
        std::filesystem::rename(temporaryPath, csvPath);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
