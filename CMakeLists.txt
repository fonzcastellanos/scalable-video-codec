cmake_minimum_required(VERSION 3.2)

project(scalable-video-codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SVC_MOTION_SSE2 "svc: enable SSE2 versions of motion-related algorithms if SSE2 is supported" ON)

find_package(OpenCV REQUIRED)

add_library(cli cli.cpp)

add_library(motion motion.cpp)

add_library(draw draw.cpp)
target_link_libraries(draw PUBLIC ${OpenCV_LIBS})

add_library(decoder_config decoder_config.cpp)
target_link_libraries(decoder_config PRIVATE cli)

add_executable(decoder decoder.cpp)
target_link_libraries(decoder PRIVATE ${OpenCV_LIBS} decoder_config)

add_library(encoder_config encoder_config.cpp)
target_link_libraries(encoder_config PUBLIC motion PRIVATE cli)

add_executable(encoder encoder.cpp)
target_link_libraries(encoder PRIVATE ${OpenCV_LIBS} encoder_config motion draw)

add_executable(encoder-visualizer encoder.cpp)
target_link_libraries(encoder-visualizer PRIVATE ${OpenCV_LIBS} encoder_config motion draw)
target_compile_definitions(encoder-visualizer PRIVATE VISUALIZE)

if (SVC_MOTION_SSE2)
    target_compile_definitions(encoder_config PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder-visualizer PRIVATE SVC_MOTION_SSE2)
endif()