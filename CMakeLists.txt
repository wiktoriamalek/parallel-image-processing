cmake_minimum_required(VERSION 3.16)
project(img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

add_library(image_processing
        src/opencl_utils.cpp
        src/common_benchmarking.cpp
        src/convolution.cpp
        src/otsu.cpp
        src/distortion.cpp
        tests/test_image_io.cpp
        tests/image_thresholding_io.cpp
        src/bernsen.cpp
)

target_include_directories(image_processing PUBLIC include)
target_link_libraries(image_processing PUBLIC OpenCL::OpenCL Threads::Threads)

add_executable(img_proc src/main.cpp)
add_executable(check_opencl tests/check_opencl.cpp)
add_executable(thresholding_time_comparison tests/thresholding_time_comparison.cpp)
add_executable(identity_convolution_test tests/convolution_test.cpp)
add_executable(otsu_test tests/otsu_test.cpp)
add_executable(distortion_test tests/distortion_test.cpp)
add_executable(bernsen_test tests/bernsen_test.cpp)

foreach(target img_proc check_opencl thresholding_time_comparison identity_convolution_test otsu_test distortion_test bernsen_test)
    target_link_libraries(${target} PRIVATE image_processing)
endforeach()
