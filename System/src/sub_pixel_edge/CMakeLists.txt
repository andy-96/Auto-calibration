cmake_minimum_required(VERSION 3.10)
project(runSubPixelEdge)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(runSubPixelEdge
        edgeTest.cpp
        edgeTest.h
        main.cpp
        WriteFile.h)
find_package(OpenCV REQUIRED)
target_link_libraries(runSubPixelEdge ${OpenCV_LIBS} )
