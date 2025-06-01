#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "image_io.hpp"
#include <fstream>
#include <iostream>

std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
    {
        std::cerr << "Nie mogę otworzyć pliku: " << path << '\n';
        std::exit(1);
    }
    
    return
    { std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>() };
}

std::vector<unsigned char> loadPNG(const std::string& file,int& w, int& h)
{
    int channels;
    unsigned char* raw =stbi_load(file.c_str(), &w, &h, &channels, 4);   // wymuszamy 4 kanały
    if(!raw)
    {
        std::cerr << "stb_image: nie można otworzyć " << file << '\n';
        std::exit(1);
    }
    return
    {
        raw, raw + w*h*4
    };
}

void savePNG(const std::string& file,const unsigned char* img, int w, int h)
{
    if(!stbi_write_png(file.c_str(), w, h, 4, img, w*4))
        std::cerr << "stb_image_write: błąd zapisu " << file << '\n';
}
