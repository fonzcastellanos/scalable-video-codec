cmake_minimum_required(VERSION 3.2)

project(scalable-video-codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SVC_MOTION_SSE2 "svc: enable SSE2 versions of motion-related algorithms if SSE2 is supported" ON)

find_package(OpenCV REQUIRED)

add_library(cli cli.cpp)

add_library(motion motion.cpp)

add_library(thread thread.cpp)

add_library(draw draw.cpp)
target_link_libraries(draw PUBLIC ${OpenCV_LIBS})

add_executable(encoder encoder.cpp)
target_link_libraries(encoder PRIVATE ${OpenCV_LIBS} cli motion thread draw)

add_executable(encoder-visualizer encoder.cpp)
target_link_libraries(encoder-visualizer PRIVATE ${OpenCV_LIBS} cli motion thread draw)
target_compile_definitions(encoder-visualizer PRIVATE VISUALIZE)

if (SVC_MOTION_SSE2)
    target_compile_definitions(encoder PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder-visualizer PRIVATE SVC_MOTION_SSE2)
endif()

add_executable(decoder decoder.cpp)
target_link_libraries(decoder PRIVATE ${OpenCV_LIBS} cli)

# add_executable(test-perf-mad test_perf_mad.cpp mad.cpp)

# add_executable(test-motion test_motion.cpp mad.cpp)
# target_link_libraries(test-motion PRIVATE ${OpenCV_LIBS} motion)

# add_executable(test-queue test_queue.cpp)