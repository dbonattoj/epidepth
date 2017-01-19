cmake_minimum_required(VERSION 3.6)
project(epi CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM external/include)
add_compile_options(-Wall -Wextra -Wno-sign-compare)

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB_RECURSE LIB_SRC "lib/*.cc" "lib/*.h" "lib/*.tcc")
file(GLOB_RECURSE PROG_SRC "prog/*.cc" "prog/*.h" "prog/*.tcc")

include_directories(SYSTEM lib)

add_executable(epi ${PROG_SRC} ${LIB_SRC})

target_link_libraries(epi ${OpenCV_LIBS})
