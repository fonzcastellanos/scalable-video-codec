cmake_minimum_required(VERSION 3.2)

project(scalable-video-codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(cli cli.cpp)

add_library(motion motion.cpp)

add_library(draw draw.cpp)
target_link_libraries(draw PUBLIC ${OpenCV_LIBS})

add_executable(encoder encoder.cpp encoder_config.cpp)
target_link_libraries(encoder PRIVATE ${OpenCV_LIBS} cli motion draw)

add_executable(encoder-visualizer encoder.cpp encoder_config.cpp)
target_link_libraries(encoder-visualizer PRIVATE ${OpenCV_LIBS} cli motion draw)
target_compile_definitions(encoder-visualizer PRIVATE VISUALIZE)

add_executable(decoder decoder.cpp decoder_config.cpp)
target_link_libraries(decoder PRIVATE ${OpenCV_LIBS} cli)