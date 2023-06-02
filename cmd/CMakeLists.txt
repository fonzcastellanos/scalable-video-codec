add_executable(encoder encoder.cpp)
target_link_libraries(encoder PRIVATE ${OpenCV_LIBS} cli encoder-lib motion draw)
target_include_directories(encoder PRIVATE ..)

add_executable(encoder-visualizer encoder.cpp)
target_link_libraries(encoder-visualizer PRIVATE ${OpenCV_LIBS} cli encoder-lib motion draw)
target_compile_definitions(encoder-visualizer PRIVATE VISUALIZE)
target_include_directories(encoder-visualizer PRIVATE ..)

if (SVC_MOTION_SSE2)
    target_compile_definitions(encoder PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder-visualizer PRIVATE SVC_MOTION_SSE2)
endif()

add_executable(decoder decoder.cpp)
target_link_libraries(decoder PRIVATE ${OpenCV_LIBS} cli decoder-lib)
target_include_directories(decoder PRIVATE ..)