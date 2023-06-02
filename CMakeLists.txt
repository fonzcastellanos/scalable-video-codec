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

add_library(decoder-lib decoder.cpp)

add_library(encoder-lib encoder.cpp)
target_link_libraries(encoder-lib PUBLIC ${OpenCV_LIBS} motion)

add_subdirectory(cmd)