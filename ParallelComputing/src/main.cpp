#include "cpu_ops.hpp"
#include "gpu_ops.hpp"
#include "image_io.hpp"
#include <fstream>   
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>

int main(int argc,char** argv)
{
    if(argc < 5)
    {
        std::cerr<< "Użycie: img_proc [invert|edge|blur|hist] in.png out [gpu|cpu] [wątki]\n";
        return 1;
    }
    
    std::string op   = argv[1];
    std::string inF  = argv[2];
    std::string outF = argv[3];
    std::string mode = argv[4];
    int nTh          = (argc>=6)? std::stoi(argv[5]) : 1;

    int w,h;
    auto buf = loadPNG(inF,w,h);
    auto* img = buf.data();
    std::size_t pixels = std::size_t(w)*h;

    auto t0 = std::chrono::high_resolution_clock::now();
    double kernelMS = -1.0;

    if(op=="invert")
    {
        if(mode=="gpu") runInvertGPU("kernels/invert.cl",img,pixels,&kernelMS);
        else            invertCPU(img,pixels);
    }
    
    else if(op=="edge")
    {
        if(mode=="gpu")
        {
            runKernelWH("kernels/edge.cl","sobelRGBA",img,w,h,&kernelMS);
        }
        else
        {
            std::vector<unsigned char> out(pixels*4);
            if(nTh<=1) sobelCPU (img,out.data(),w,h);
            else       sobelCPUMT(img,out.data(),w,h,nTh);
            std::copy(out.begin(),out.end(),img);
        }
    }
    else if(op=="blur")
    {
        if(mode=="gpu")
        {
            int passes=40;
            for(int i=0;i<passes;++i)
                runKernelWH("kernels/blur.cl","boxBlurRGBA",img,w,h,&kernelMS);
        }
        else
        {
            std::vector<unsigned char> out(pixels*4);
            if(nTh<=1) boxBlurCPU (img,out.data(),w,h);
            else       boxBlurCPUMT(img,out.data(),w,h,nTh);
            std::copy(out.begin(),out.end(),img);
        }
    }
    else if(op=="hist")
    {
        uint64_t H[3][256];
        if(mode=="gpu")
            runHistGPU("kernels/hist.cl",img,pixels,H,&kernelMS);
        else
        {
            if(nTh<=1) histCPU (img,pixels,H);
            else       histCPUMT(img,pixels,H,nTh);
        }
        std::ofstream csv(outF);
        if(!csv)
        { std::cerr<<"Nie mogę zapisać "<<outF<<"\n"; return 1; }
        csv<<"bin,R,G,B\n";
        for(int b=0;b<256;++b)
            csv<<b<<','<<H[0][b]<<','<<H[1][b]<<','<<H[2][b]<<'\n';
    }
    else
    {
        std::cerr<<"Nieznana operacja "<<op<<"\n"; return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double totalMS =std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0;

    if(op!="hist") savePNG(outF,img,w,h);

    std::cout<<"Tryb: "<<mode<<(mode=="cpu"?" ("+std::to_string(nTh)+" wąt.)":"")<<"\nCzas: "<<totalMS<<" ms\n";
    if(kernelMS>=0) std::cout<<"   kernel GPU: "<<kernelMS<<" ms\n";
    return 0;
}
