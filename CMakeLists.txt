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
        src/gabor.cpp
)

target_include_directories(image_processing PUBLIC include)
target_link_libraries(image_processing PUBLIC OpenCL::OpenCL Threads::Threads)

add_executable(median_network_codegen src/median_network_codegen.cpp src/median_filter.cpp src/opencl_utils.cpp)
target_include_directories(median_network_codegen PRIVATE include)
target_link_libraries(median_network_codegen PRIVATE OpenCL::OpenCL Threads::Threads)
set(MEDIAN_NETWORKS_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(MEDIAN_NETWORKS_FILE "${MEDIAN_NETWORKS_DIRECTORY}/median_networks_generated.inc")
add_custom_command(OUTPUT "${MEDIAN_NETWORKS_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MEDIAN_NETWORKS_DIRECTORY}"
        COMMAND median_network_codegen "${MEDIAN_NETWORKS_FILE}"
        DEPENDS median_network_codegen)
add_custom_target(median_networks DEPENDS "${MEDIAN_NETWORKS_FILE}")
add_dependencies(image_processing median_networks)
target_include_directories(image_processing PRIVATE "${MEDIAN_NETWORKS_DIRECTORY}")
target_compile_definitions(image_processing PRIVATE MEDIAN_COMPILED_NETWORKS)
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
                COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -std=metal3.0 -O2 -fno-fast-math -ffp-contract=on
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
add_executable(cpu_vectorization_test tests/cpu_vectorization_test.cpp)
add_executable(gabor_test tests/gabor_test.cpp)

foreach(target img_proc check_opencl thresholding_time_comparison identity_convolution_test otsu_test distortion_test bernsen_test median_filter_test region_growing_test multi_otsu_test negative_test contrast_test brown_conrady_test peaks_valleys_detection_test niblack_sauvola_test core_scheduling_test canny_test cpu_vectorization_test gabor_test)
    target_link_libraries(${target} PRIVATE image_processing)
endforeach()

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/qt)
    add_executable(metal_test tests/metal_test.cpp)
    target_link_libraries(metal_test PRIVATE image_processing)
    add_executable(check_metal tests/check_metal.cpp)
    target_link_libraries(check_metal PRIVATE image_processing)

    set(METAL_COMPARISON_SOURCES tests/opencl_metal_overhead.mm tests/opencl_metal_workgroups.mm tests/metal_fp_contract.mm
            tests/metal_median_options.mm tests/mps_comparison.mm)
    set_source_files_properties(${METAL_COMPARISON_SOURCES} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    foreach(source IN LISTS METAL_COMPARISON_SOURCES)
        get_filename_component(target "${source}" NAME_WE)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE image_processing)
    endforeach()
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    target_link_libraries(mps_comparison PRIVATE ${MPS_FRAMEWORK})
    target_compile_definitions(opencl_metal_workgroups PRIVATE METAL_LIBRARY_PATH="${METAL_LIBRARY}")
    add_dependencies(opencl_metal_workgroups metal_kernels)

    set(METAL_VARIANT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metal_variants")
    set(METAL_VARIANT_LIBRARIES)
    function(add_metal_variant name)
        cmake_parse_arguments(VARIANT "" "" "SOURCES;FLAGS" ${ARGN})
        set(air_files)
        foreach(kernel_source IN LISTS VARIANT_SOURCES)
            get_filename_component(kernel_name "${kernel_source}" NAME_WE)
            set(kernel_air "${METAL_VARIANT_DIRECTORY}/${name}_${kernel_name}.air")
            add_custom_command(OUTPUT "${kernel_air}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${METAL_VARIANT_DIRECTORY}"
                    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal ${VARIANT_FLAGS} -c "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_source}" -o "${kernel_air}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_source}"
                    VERBATIM
            )
            list(APPEND air_files "${kernel_air}")
        endforeach()
        set(library "${METAL_VARIANT_DIRECTORY}/${name}.metallib")
        add_custom_command(OUTPUT "${library}"
                COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${air_files} -o "${library}"
                DEPENDS ${air_files}
                VERBATIM
        )
        set(METAL_VARIANT_LIBRARIES ${METAL_VARIANT_LIBRARIES} "${library}" PARENT_SCOPE)
    endfunction()
    set(FP_SOURCES kernels/metal/distortion_correction.metal kernels/metal/convolution.metal)
    add_metal_variant(fp_off SOURCES ${FP_SOURCES} FLAGS -std=metal3.0 -O2 -fno-fast-math -ffp-contract=off)
    add_metal_variant(fp_on SOURCES ${FP_SOURCES} FLAGS -std=metal3.0 -O2 -fno-fast-math -ffp-contract=on)
    add_metal_variant(fp_fast SOURCES ${FP_SOURCES} FLAGS -std=metal3.0 -O2 -fno-fast-math -ffp-contract=fast)
    add_metal_variant(fp_fastmath SOURCES ${FP_SOURCES} FLAGS -std=metal3.0 -O2)
    add_metal_variant(median_project SOURCES kernels/metal/median_filter.metal FLAGS -std=metal3.0 -O2 -fno-fast-math -ffp-contract=on -DMEDIAN_RADIUS=1)
    add_metal_variant(median_default SOURCES kernels/metal/median_filter.metal FLAGS -O2 -DMEDIAN_RADIUS=1)
    add_metal_variant(median_size SOURCES kernels/metal/median_filter.metal FLAGS -std=metal3.0 -Os -DMEDIAN_RADIUS=1)
    add_custom_target(metal_variants DEPENDS ${METAL_VARIANT_LIBRARIES})
    foreach(target metal_fp_contract metal_median_options)
        add_dependencies(${target} metal_variants)
        target_compile_definitions(${target} PRIVATE METAL_VARIANT_DIRECTORY="${METAL_VARIANT_DIRECTORY}")
    endforeach()
endif()
find_package(Qt6 COMPONENTS Widgets Concurrent QUIET)
add_library(algorithm_benchmarking gui/algorithms.cpp)
target_include_directories(algorithm_benchmarking PUBLIC gui)
target_link_libraries(algorithm_benchmarking PUBLIC image_processing)

if(APPLE)
    add_executable(metal_comparison tests/metal_comparison.cpp)
    target_link_libraries(metal_comparison PRIVATE algorithm_benchmarking)
    target_compile_definitions(metal_comparison PRIVATE IMAGE_BUILD_TYPE="$<CONFIG>")
    add_executable(opencv_comparison tests/opencv_comparison.cpp)
    target_link_libraries(opencv_comparison PRIVATE algorithm_benchmarking)
endif()

if(Qt6_FOUND)
    add_executable(img_gui
            gui/main.cpp
            gui/main_window.cpp
            gui/main_window.hpp
            gui/image_view.cpp
            gui/image_view.hpp
            gui/theme.cpp
            gui/theme.hpp
            gui/algorithms.hpp
    )
    set_target_properties(img_gui PROPERTIES AUTOMOC ON)
    target_link_libraries(img_gui PRIVATE algorithm_benchmarking Qt6::Widgets Qt6::Concurrent)
    target_compile_definitions(img_gui PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
else()
    message(STATUS "Qt 6 not found, img_gui will not be built")
endif()
