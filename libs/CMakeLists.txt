add_library(cli cli.cpp)

add_library(motion motion.cpp)

add_library(thread thread.cpp)

add_library(draw draw.cpp)
target_link_libraries(draw PUBLIC ${OpenCV_LIBS})

add_library(encoder-lib encoder.cpp)
target_link_libraries(encoder-lib PUBLIC ${OpenCV_LIBS} motion)

add_library(encoder-visualizer-lib encoder.cpp)
target_link_libraries(encoder-visualizer-lib PUBLIC ${OpenCV_LIBS} motion PRIVATE draw)
target_compile_definitions(encoder-visualizer-lib PRIVATE VISUALIZE)

if (SVC_MOTION_SSE2)
    target_compile_definitions(encoder-lib PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder-visualizer-lib PRIVATE SVC_MOTION_SSE2)
endif()


add_library(decoder-lib decoder.cpp)
target_link_libraries(decoder-lib PRIVATE ${OpenCV_LIBS})