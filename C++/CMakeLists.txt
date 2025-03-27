cmake_minimum_required(VERSION 3.30)
project(Mandelbrot)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/raylib)

add_executable(${PROJECT_NAME} main.cpp)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(${PROJECT_NAME} raylib)
