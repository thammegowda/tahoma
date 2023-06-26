cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(rtgpp CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(src)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch/include/torch/csrc/api/include")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_subdirectory(src)

