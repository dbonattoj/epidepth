cmake_minimum_required(VERSION 3.6)
project(epi CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(SYSTEM external/include)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

add_compile_options(-Wall -Wextra -Wno-sign-compare)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB_RECURSE LIB_SRC "lib/*.cc" "lib/*.h" "lib/*.tcc")
file(GLOB_RECURSE PROG_SRC "prog/*.cc" "prog/*.h" "prog/*.tcc")

add_executable(epi ${PROG_SRC} ${LIB_SRC})

target_include_directories(epi PRIVATE lib)
target_link_libraries(epi ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES})

