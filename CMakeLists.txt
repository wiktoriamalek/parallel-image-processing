cmake_minimum_required(VERSION 3.16)
project(img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

add_library(image_processing
        src/opencl_utils.cpp
        src/common_benchmarking.cpp
        src/convolution.cpp
        src/otsu.cpp
        src/distortion.cpp
        src/image_io.cpp
        src/thresholding.cpp
        src/bernsen.cpp
        src/median_filter.cpp
        src/resize.cpp
        src/region_growing.cpp
        src/negative.cpp
        src/contrast.cpp
        src/peaks_valleys_detection.cpp
        src/canny.cpp
)

target_include_directories(image_processing PUBLIC include)
target_link_libraries(image_processing PUBLIC OpenCL::OpenCL Threads::Threads)
if(APPLE)
    enable_language(OBJCXX)

    set_target_properties(image_processing PROPERTIES
            OBJCXX_STANDARD 17
            OBJCXX_STANDARD_REQUIRED ON
    )

    target_sources(image_processing PRIVATE src/metal_utils.mm src/metal_algorithms.mm)

    set_source_files_properties(src/metal_utils.mm src/metal_algorithms.mm PROPERTIES
            COMPILE_OPTIONS "-fobjc-arc"
    )

    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
    file(GLOB METAL_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/*.metal")
    list(REMOVE_ITEM METAL_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/median_filter.metal")
    set(METAL_AIR_FILES)
    foreach(kernel_source IN LISTS METAL_KERNEL_SOURCES)
        get_filename_component(kernel_name "${kernel_source}" NAME_WE)
        set(kernel_air "${CMAKE_CURRENT_BINARY_DIR}/metal/${kernel_name}.air")
        add_custom_command(OUTPUT "${kernel_air}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metal"
                COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -std=metal3.0 -O2 -fno-fast-math -ffp-contract=off
                        -c "${kernel_source}" -o "${kernel_air}"
                DEPENDS "${kernel_source}"
                VERBATIM
        )
        list(APPEND METAL_AIR_FILES "${kernel_air}")
    endforeach()
    set(METAL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/metal/image_processing.metallib")
    add_custom_command(OUTPUT "${METAL_LIBRARY}"
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR_FILES} -o "${METAL_LIBRARY}"
            DEPENDS ${METAL_AIR_FILES}
            VERBATIM
    )
    add_custom_target(metal_kernels DEPENDS "${METAL_LIBRARY}")
    add_dependencies(image_processing metal_kernels)
    target_compile_definitions(image_processing PRIVATE
            METAL_LIBRARY_PATH="${METAL_LIBRARY}"
            METAL_MEDIAN_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/median_filter.metal"
    )

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

    target_link_libraries(image_processing PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )
endif()
add_executable(img_proc src/main.cpp)
add_executable(check_opencl tests/check_opencl.cpp)
add_executable(thresholding_time_comparison tests/thresholding_time_comparison.cpp)
add_executable(identity_convolution_test tests/convolution_test.cpp)
add_executable(otsu_test tests/otsu_test.cpp)
add_executable(distortion_test tests/distortion_test.cpp)
add_executable(bernsen_test tests/bernsen_test.cpp)
add_executable(median_filter_test tests/median_filter_test.cpp)
add_executable(region_growing_test tests/region_growing_test.cpp)
add_executable(multi_otsu_test tests/multi_otsu_test.cpp)
add_executable(negative_test tests/negative_test.cpp)
add_executable(contrast_test tests/contrast_test.cpp)
add_executable(brown_conrady_test tests/brown_conrady_test.cpp)
add_executable(peaks_valleys_detection_test tests/peaks_valleys_detection_test.cpp)
add_executable(niblack_sauvola_test tests/niblack_sauvola_test.cpp)
add_executable(core_scheduling_test tests/core_scheduling_test.cpp)
add_executable(canny_test tests/canny_test.cpp)

foreach(target img_proc check_opencl thresholding_time_comparison identity_convolution_test otsu_test distortion_test bernsen_test median_filter_test region_growing_test multi_otsu_test negative_test contrast_test brown_conrady_test peaks_valleys_detection_test niblack_sauvola_test core_scheduling_test canny_test)
    target_link_libraries(${target} PRIVATE image_processing)
endforeach()

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/qt)
    add_executable(check_metal tests/check_metal.cpp)
    target_link_libraries(check_metal PRIVATE image_processing)
endif()
find_package(Qt6 COMPONENTS Widgets Concurrent QUIET)
add_library(algorithm_benchmarking gui/algorithms.cpp)
target_include_directories(algorithm_benchmarking PUBLIC gui)
target_link_libraries(algorithm_benchmarking PUBLIC image_processing)

if(Qt6_FOUND)
    add_executable(img_gui
            gui/main.cpp
            gui/main_window.cpp
            gui/main_window.hpp
            gui/image_view.cpp
            gui/image_view.hpp
            gui/algorithms.hpp
    )
    set_target_properties(img_gui PROPERTIES AUTOMOC ON)
    target_link_libraries(img_gui PRIVATE algorithm_benchmarking Qt6::Widgets Qt6::Concurrent)
    target_compile_definitions(img_gui PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
else()
    message(STATUS "Qt 6 not found, img_gui will not be built")
endif()
