#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h" //wczytywanie obrazow
#include "stb_image_write.h" //zapisywanie obrazow

#include "image_io.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Nie można otworzyć pliku: " << path << '\n';
        std::exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), {});
}

std::vector<unsigned char> loadPNG(const std::string& file, int& w, int& h) {
    int channels = 0;
    unsigned char* raw = stbi_load(file.c_str(), &w, &h, &channels, 4); // wymuszone RGBA
    if (!raw) {
        std::cerr << "Błąd wczytywania obrazu " << file << "\n";
        std::exit(1);
    }

    std::vector<unsigned char> buffer(raw, raw + w * h * 4);
    stbi_image_free(raw);
    return buffer;
}

void savePNG(const std::string& file, const unsigned char* img, int w, int h) {
    if (!stbi_write_png(file.c_str(), w, h, 4, img, w * 4)) {
        std::cerr << "Nie udało się zapisać obrazu do pliku: " << file << '\n';
    }
}
