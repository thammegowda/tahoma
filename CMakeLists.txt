cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(rtgp CXX C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_CCACHE "Use ccache compiler cache (https://ccache.dev)" ON)

find_package(Git QUIET)

set(CMAKE_BUILD_TYPE Debug)

# TODO  libtorch as fetchcontent https://cmake.org/cmake/help/latest/module/FetchContent.html
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch" CACHE PATH "libTorch path")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXT_LIBS "${TORCH_LIBRARIES}")


if(USE_CCACHE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -C")
endif()


include_directories(
    "${CMAKE_SOURCE_DIR}/libs/libtorch/include/torch/csrc/api/include"
    "${CMAKE_SOURCE_DIR}/libs/sentencepiece/src"
    "${CMAKE_SOURCE_DIR}/libs/spdlog/include"
    "${CMAKE_SOURCE_DIR}/libs/tomlplusplus/include"
    )


# Add the SentencePiece library
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/sentencepiece)
set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/backward-cpp)


add_subdirectory(src)
