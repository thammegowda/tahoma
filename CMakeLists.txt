cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(rtgpp CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO  libtorch as fetchcontent https://cmake.org/cmake/help/latest/module/FetchContent.html
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch" CACHE PATH "libTorch path")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    "${CMAKE_SOURCE_DIR}/libs/libtorch/include/torch/csrc/api/include"
    "${CMAKE_SOURCE_DIR}/libs/sentencepiece/src"
    "${CMAKE_SOURCE_DIR}/libs/spdlog/include"
    "${CMAKE_SOURCE_DIR}/libs/tomlplusplus/include"
    )

add_subdirectory(src)
