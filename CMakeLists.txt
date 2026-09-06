cmake_minimum_required(VERSION 3.16)
project(img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(img_proc
    src/main.cpp
    tests/test_image_io.cpp
        tests/image_thresholding_io.cpp
)
target_include_directories(img_proc PRIVATE include)

find_package(OpenCL REQUIRED)
target_link_libraries(img_proc PRIVATE OpenCL::OpenCL)

if(UNIX AND NOT APPLE)
    target_link_libraries(img_proc PRIVATE pthread)
endif()
