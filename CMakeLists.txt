cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
# older cmake doesnt support CUDA_STANDARD 20
project(tahoma CXX C)
set(CMAKE_CXX_STANDARD 23)  # cuda doesnt support c++23 yet
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

# set options with default values
option(USE_CCACHE "Use ccache compiler cache (https://ccache.dev)" OFF)
option(USE_CUDA "Use CUDA libtorch. Default is CPU only libtorch" OFF)
option(USE_ROCM "Use ROCM libtorch. Default is CPU only libtorch" OFF)
option(USE_PYTORCH "Use libtorch from the available pytorch installation
     (i.e., torch.utils.cmake_prefix_path). This can be CPU/CUDA/ROCM build." OFF)
if (USE_CUDA AND USE_ROCM)
    message(FATAL_ERROR "USE_CUDA and USE_ROCM cannot be set at the same time")
endif()

set(PYTHON_EXECUTABLE python CACHE FILEPATH "Python executable name path. Used to find libtorch from pytorch installation")
find_package(Git QUIET)

set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -ggdb3 -O0")
    message(STATUS "Debug build. Flags set to retain most debug info and disable all optimizations")
endif()


# run git submodule update --init --recursive
execute_process(
    COMMAND git submodule update --init --recursive --no-fetch
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)


# execute_process(
#     COMMAND bash  ${CMAKE_SOURCE_DIR}/libs/setup.sh
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# # only one of USE_CUDA, USE_ROCM, USE_PYTORCH can be set
if ((USE_CUDA AND USE_ROCM) OR
    (USE_CUDA AND USE_PYTORCH) OR
    (USE_ROCM AND USE_PYTORCH))
    message(FATAL_ERROR "Only one of USE_CUDA, USE_ROCM, USE_PYTORCH can be set")
endif()


if (USE_CUDA)
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-cu121" CACHE PATH "libTorch CUDA path")
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA root path")
        #set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH "CUDA root path")
    endif()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

elseif (USE_ROCM)
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-rocm" CACHE PATH "libTorch ROCM path")
elseif (USE_PYTORCH)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE LIBTORCH_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
else()
    # CPU only build by default
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-cpu" CACHE PATH "libTorch CPU path")
    # debug build here
    #set(LIBTORCH_PATH "/mnt/home/tg/work/repos/other/pytorch/libtorch-cpu" CACHE PATH "libTorch CPU path")
endif()

message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}" CACHE PATH "libTorch path")

# add_subdirectory(${CMAKE_SOURCE_DIR}/libs/pytorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXT_LIBS "${TORCH_LIBRARIES}")

if(USE_CCACHE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -C")
endif()



include_directories(
    "${CMAKE_SOURCE_DIR}/libs/libtorch/include/torch/csrc/api/include"
    #"${CMAKE_SOURCE_DIR}/libs/pytorch/torch/include/"
    "${CMAKE_SOURCE_DIR}/libs/sentencepiece/src"
    "${CMAKE_SOURCE_DIR}/libs/spdlog/include"
    "${CMAKE_SOURCE_DIR}/libs/yaml-cpp/include"
    )

# "${CMAKE_SOURCE_DIR}/libs/tomlplusplus/include"


### SentencePiece library
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/sentencepiece)
set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)

### yaml-cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/yaml-cpp)
set(EXT_LIBS ${EXT_LIBS} yaml-cpp)

###
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/backward-cpp)

add_subdirectory(src)
