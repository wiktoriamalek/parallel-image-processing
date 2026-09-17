cmake_minimum_required(VERSION 3.16)
project(img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(img_proc
        src/main.cpp
        tests/test_image_io.cpp
        tests/image_thresholding_io.cpp
)

target_include_directories(img_proc PRIVATE include)
target_link_libraries(img_proc PRIVATE OpenCL::OpenCL)

add_executable(check_opencl
        tests/check_opencl.cpp
        tests/test_image_io.cpp
        tests/image_thresholding_io.cpp
)

target_include_directories(check_opencl PRIVATE include)
target_link_libraries(check_opencl PRIVATE OpenCL::OpenCL)

add_executable(thresholding_time_comparison
        tests/thresholding_time_comparison.cpp
        tests/test_image_io.cpp
        tests/image_thresholding_io.cpp
)

add_executable(identity_convolution_test
        src/convolution.cpp
        tests/test_image_io.cpp
        tests/convolution_test.cpp
)

target_include_directories(identity_convolution_test PRIVATE include)
target_link_libraries(identity_convolution_test PRIVATE OpenCL::OpenCL)

target_include_directories(thresholding_time_comparison PRIVATE include)
target_link_libraries(thresholding_time_comparison PRIVATE OpenCL::OpenCL)


add_executable(otsu_test
        src/otsu.cpp
        tests/otsu_test.cpp
        tests/test_image_io.cpp
)

target_include_directories(otsu_test PRIVATE include)

if(UNIX AND NOT APPLE)
    target_link_libraries(img_proc PRIVATE pthread)
    target_link_libraries(check_opencl PRIVATE pthread)
    target_link_libraries(thresholding_time_comparison PRIVATE pthread)
endif()
