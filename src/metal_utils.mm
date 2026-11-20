#include "metal_utils.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>

std::string metalDeviceName() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (!device) {
            throw std::runtime_error("No Metal GPU found");
        }

        return std::string([[device name] UTF8String]);
    }
}