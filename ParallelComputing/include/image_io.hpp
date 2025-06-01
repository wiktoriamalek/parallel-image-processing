#pragma once
#include <string>
#include <vector>

std::string readFile(const std::string& path);

std::vector<unsigned char> loadPNG(const std::string& file,int& w,int& h);
void savePNG(const std::string& file,const unsigned char* img, int w,int h);
