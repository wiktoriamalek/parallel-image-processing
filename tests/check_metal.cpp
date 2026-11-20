#include "metal_utils.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        std::cout << "GPU Metal: " << metalDeviceName() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}