cmake_minimum_required(VERSION 3.2)

project(scalable-video-codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SVC_MOTION_SSE2 "svc: enable SSE2 versions of motion-related algorithms if SSE2 is supported" ON)

find_package(OpenCV REQUIRED)

add_subdirectory(libs)
add_subdirectory(apps)

# add_executable(test-perf-mad test_perf_mad.cpp mad.cpp)

# add_executable(test-motion test_motion.cpp mad.cpp)
# target_link_libraries(test-motion PRIVATE ${OpenCV_LIBS} motion)

# add_executable(test-queue test_queue.cpp)