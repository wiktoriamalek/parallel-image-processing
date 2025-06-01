// implementacje operacji CPU (1- i wielowątkowych)
#include "cpu_ops.hpp"
#include <algorithm>
#include <thread>
#include <cmath>
#include <array>

// jednowątkowe operacje
void invertCPU(unsigned char* img, std::size_t pixels)
{
    for(std::size_t i = 0; i < pixels*4; i += 4)
    {
        img[i  ] = 255 - img[i  ];
        img[i+1] = 255 - img[i+1];
        img[i+2] = 255 - img[i+2];
    }
}

void boxBlurCPU(const unsigned char* src, unsigned char* dst,int w, int h)
{
    auto idx = [w](int x,int y){ return (y*w + x)*4; };

    for(int y=0; y<h; ++y)
        
        for(int x=0; x<w; ++x)
        {
            int sum[4] = {0}, cnt = 0;
            
            for(int dy=-1; dy<=1; ++dy)
            {
                int yy = std::clamp(y+dy, 0, h-1);
                
                for(int dx=-1; dx<=1; ++dx)
                {
                    int xx = std::clamp(x+dx, 0, w-1);
                    const unsigned char* p = src + idx(xx,yy);
                    
                    for(int c=0; c<4; ++c) sum[c] += p[c];
                    ++cnt;
                }
            }
            unsigned char* o = dst + idx(x,y);
            
            for(int c=0; c<4; ++c) o[c] = sum[c] / cnt;
        }
}

void sobelCPU(const unsigned char* src, unsigned char* dst,int w, int h)
{
    auto lum = [](const unsigned char* p)
    {
        return (77*p[0] + 150*p[1] + 29*p[2]) >> 8;
    };
    
    auto idx = [w](int x,int y){ return (y*w + x)*4; };

    for(int y=0; y<h; ++y)
        for(int x=0; x<w; ++x)
        {
            int L[3][3];
            for(int dy=-1; dy<=1; ++dy)
            {
                int yy = std::clamp(y+dy, 0, h-1);
                for(int dx=-1; dx<=1; ++dx)
                {
                    int xx = std::clamp(x+dx, 0, w-1);
                    L[dy+1][dx+1] = lum(src + idx(xx,yy));
                }
            }
            int gx =  L[0][2] + 2*L[1][2] + L[2][2]- L[0][0] - 2*L[1][0] - L[2][0];
            int gy =  L[2][0] + 2*L[2][1] + L[2][2]- L[0][0] - 2*L[0][1] - L[0][2];

            unsigned char m =static_cast<unsigned char>((std::abs(gx)+std::abs(gy)) >> 2);

            unsigned char* o = dst + idx(x,y);
            o[0]=o[1]=o[2]=m;  o[3]=255;
        }
}

// wielowątkowe odpowiedniki
void boxBlurCPUMT(const unsigned char* src,unsigned char* dst,int w,int h,int nTh)
{
    std::vector<std::thread> pool;
    int rows = (h + nTh - 1) / nTh;

    for(int t=0; t<nTh; ++t)
    {
        int y0 = t*rows,
            y1 = std::min(h, y0 + rows);

        pool.emplace_back([=]{
            auto idx=[w](int x,int y)
            {
                return (y*w+x)*4;
            };
            
            for(int y=y0; y<y1; ++y)
                for(int x=0; x<w; ++x)
                {
                    int sum[4]={0}, cnt=0;
                    
                    for(int dy=-1; dy<=1; ++dy)
                    {
                        int yy = std::clamp(y+dy,0,h-1);
                        for(int dx=-1; dx<=1; ++dx)
                        {
                            int xx = std::clamp(x+dx,0,w-1);
                            const unsigned char* p = src + idx(xx,yy);
                            for(int c=0; c<4; ++c) sum[c] += p[c];
                            ++cnt;
                        }
                    }
                    unsigned char* o = dst + idx(x,y);
                    for(int c=0; c<4; ++c) o[c] = sum[c]/cnt;
                }
        });
    }
    for(auto& th : pool) th.join();
}

void sobelCPUMT(const unsigned char* src,unsigned char* dst,int w,int h,int nTh)
{
    std::vector<std::thread> pool;
    int rows = (h + nTh - 1) / nTh;

    for(int t=0; t<nTh; ++t)
    {
        int y0 = t*rows,
            y1 = std::min(h, y0 + rows);

        pool.emplace_back([=]
                          {
            auto lum=[](const unsigned char* p)
                        {
                            return (77*p[0]+150*p[1]+29*p[2]) >> 8;
                        };
            auto idx=[w](int x,int y)
            {
                return (y*w+x)*4;
            };

            for(int y=y0; y<y1; ++y)
                for(int x=0; x<w; ++x)
                {
                    int L[3][3];
                    for(int dy=-1; dy<=1; ++dy)
                    {
                        int yy = std::clamp(y+dy,0,h-1);
                        for(int dx=-1; dx<=1; ++dx)
                        {
                            int xx = std::clamp(x+dx,0,w-1);
                            L[dy+1][dx+1] = lum(src + idx(xx,yy));
                        }
                    }
                    int gx =  L[0][2] + 2*L[1][2] + L[2][2]- L[0][0] - 2*L[1][0] - L[2][0];
                    int gy =  L[2][0] + 2*L[2][1] + L[2][2]- L[0][0] - 2*L[0][1] - L[0][2];

                    unsigned char m =static_cast<unsigned char>((std::abs(gx)+std::abs(gy)) >> 2);

                    unsigned char* o = dst + idx(x,y);
                    o[0]=o[1]=o[2]=m; o[3]=255;
                }
        });
    }
    
    for(auto& th : pool) th.join();
}

void histCPU(const unsigned char* img,std::size_t pixels,uint64_t H[3][256])
{
    std::fill(&H[0][0], &H[0][0] + 3*256, 0);

    for(std::size_t i=0; i<pixels; ++i)
    {
        const unsigned char* p = img + i*4;
        ++H[0][p[0]];
        ++H[1][p[1]];
        ++H[2][p[2]];
    }
}

void histCPUMT(const unsigned char* img,std::size_t pixels,uint64_t H[3][256],int nTh)
{
    std::vector<std::array<uint64_t,256*3>> local(nTh);
    std::fill(&H[0][0], &H[0][0] + 3*256, 0);

    auto worker = [&](int tid){
        std::size_t chunk = (pixels + nTh - 1) / nTh;
        std::size_t a = tid * chunk,b = std::min(pixels, a + chunk);

        auto& L = local[tid];
        std::fill(L.begin(), L.end(), 0);

        for(std::size_t i=a; i<b; ++i)
        {
            const unsigned char* p = img + i*4;
            ++L[0*256 + p[0]];
            ++L[1*256 + p[1]];
            ++L[2*256 + p[2]];
        }
    };

    std::vector<std::thread> pool;
    for(int t=0; t<nTh; ++t) pool.emplace_back(worker, t);
    for(auto& th : pool) th.join();

    for(int t=0; t<nTh; ++t)
        for(int c=0; c<3; ++c)
            for(int b=0; b<256; ++b)
                H[c][b] += local[t][c*256 + b];
}
