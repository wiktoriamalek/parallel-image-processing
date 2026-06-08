#include "cpu_ops.hpp"
#include "gpu_ops.hpp"
#include "image_io.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

//bez std zeby w razie czego moc uzyc begin()

int main(int argc, char** argv) { //tablica lancuchow znakow
    if (argc < 5) {
        std::cerr << "Użycie: img_proc [invert|edge|blur|hist] in.png out [gpu|cpu] [wątki]\n"; //w razie bledu
        return 1;
    }

    std::string op = argv[1]; //operacja
    std::string inF = argv[2]; //sciezka do testowego png
    std::string outF = argv[3]; //sciezka do zapisywanego pliku wynikowego
    std::string mode = argv[4]; //cpu czy gpu
    int nThreads = (argc >= 6) ? std::stoi(argv[5]) : 1; // jesli nie poda sie liczby watkow to jest defaultowo 1

    int w = 0, h = 0;
    auto buffer = loadPNG(inF, w, h);
    auto* img = buffer.data();
    std::size_t pixels = static_cast<std::size_t>(w) * h; //swiadoma zmiana z int na size_t

    auto tStart = std::chrono::high_resolution_clock::now();
    double kernelTime = -1.0;

    if (op == "invert") {
        if (mode == "gpu") runInvertGPU("kernels/invert.cl", img, pixels, &kernelTime);
        else invertCPU(img, pixels);
    } else if (op == "edge") {
        if (mode == "gpu") {
            runKernelWH("kernels/edge.cl", "sobelRGBA", img, w, h, &kernelTime);
        } else {
            std::vector<unsigned char> tmp(pixels * 4);
            if (nThreads <= 1)
                sobelCPU(img, tmp.data(), w, h);
            else
                sobelCPUMT(img, tmp.data(), w, h, nThreads);
            std::copy(tmp.begin(), tmp.end(), img);
        }
    } else if (op == "blur") {
        if (mode == "gpu") {
            for (int i = 0; i < 40; ++i)
                runKernelWH("kernels/blur.cl", "boxBlurRGBA", img, w, h, &kernelTime);
        } else {
            std::vector<unsigned char> tmp(pixels * 4);
            if (nThreads <= 1)
                boxBlurCPU(img, tmp.data(), w, h);
            else
                boxBlurCPUMT(img, tmp.data(), w, h, nThreads);
            std::copy(tmp.begin(), tmp.end(), img);
        }
    } else if (op == "hist") {
        uint64_t H[3][256] = {};
        if (mode == "gpu")
            run_histogram_gpu("kernels/hist.cl", img, pixels, H, &kernelTime);
        else if (nThreads <= 1)
            histCPU(img, pixels, H);
        else
            histCPUMT(img, pixels, H, nThreads);

        std::ofstream csv(outF);
        if (!csv) {
            std::cerr << "Nie mogę zapisać " << outF << "\n";
            return 1;
        }
        csv << "bin,R,G,B\n";
        for (int b = 0; b < 256; ++b)
            csv << b << ',' << H[0][b] << ',' << H[1][b] << ',' << H[2][b] << '\n';
    } else {
        std::cerr << "Nieznana operacja " << op << "\n";
        return 2;
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count() / 1000.0;

    if (op != "hist") savePNG(outF, img, w, h);

    std::cout << "Tryb: " << mode;
    if (mode == "cpu") std::cout << " (" << nThreads << " wąt.)";
    std::cout << "\nCzas: " << totalTime << " ms\n";
    if (kernelTime >= 0) std::cout << "   kernel GPU: " << kernelTime << " ms\n";

    return 0;
}
