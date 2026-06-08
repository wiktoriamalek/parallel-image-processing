cmake_minimum_required(VERSION 3.16)
project(img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC src/*.cpp)

add_executable(img_proc ${SRC})
target_include_directories(img_proc PRIVATE include)

find_package(OpenCL REQUIRED)
target_link_libraries(img_proc PRIVATE OpenCL::OpenCL)

if(UNIX AND NOT APPLE)
    target_link_libraries(img_proc PRIVATE pthread)
endif()
