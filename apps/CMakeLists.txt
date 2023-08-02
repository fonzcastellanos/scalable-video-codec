add_executable(encoder encoder.cpp)
target_link_libraries(encoder PRIVATE ${OpenCV_LIBS} cli motion thread encoder-lib)
target_include_directories(encoder PRIVATE ${PROJECT_SOURCE_DIR}/libs) 

add_executable(encoder-visualizer encoder.cpp)
target_link_libraries(encoder-visualizer PRIVATE ${OpenCV_LIBS} cli motion thread encoder-visualizer-lib)
target_include_directories(encoder-visualizer PRIVATE ${PROJECT_SOURCE_DIR}/libs) 

if (SVC_MOTION_SSE2)
    target_compile_definitions(encoder PRIVATE SVC_MOTION_SSE2)
    target_compile_definitions(encoder-visualizer PRIVATE SVC_MOTION_SSE2)
endif()

add_executable(decoder decoder.cpp)
target_link_libraries(decoder PRIVATE cli thread decoder-lib)
target_include_directories(decoder PRIVATE ${PROJECT_SOURCE_DIR}/libs) 