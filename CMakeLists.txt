cmake_minimum_required(VERSION 3.16.3) # Or whatever version you use

project(SSA_r)

find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3.7 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
link_directories(${Eigen3_LIBRARY_DIRS})

add_executable(SSA_r  main.cpp)

target_link_libraries (SSA_r ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES})
