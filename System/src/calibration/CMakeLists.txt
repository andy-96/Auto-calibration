cmake_minimum_required(VERSION 3.10)
project(runCalibrate)

set(CMAKE_CXX_STANDARD 11)
include_directories(.)
add_executable(runCalibrate
        CMakeLists.txt
        main.cpp)

find_package( OpenCV REQUIRED)
target_link_libraries(runCalibrate ${OpenCV_LIBS} )