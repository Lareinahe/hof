cmake_minimum_required(VERSION 3.8)
project(HOF)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(HOF ${SOURCE_FILES})
target_link_libraries(HOF ${OpenCV_LIBS})