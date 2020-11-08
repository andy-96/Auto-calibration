cmake_minimum_required(VERSION 3.10)
project(runFilter)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)
include_directories("/usr/include/eigen3")
add_executable(runFilter
        Calibration.cpp
        Calibration.h
        Kalman_filter.cpp
        Kalman_filter.h
        main.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(runFilter ${OpenCV_LIBS})