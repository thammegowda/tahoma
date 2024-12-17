cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
# older cmake doesnt support CUDA_STANDARD 20  https://stackoverflow.com/q/75010206/1506477
# if you cannot upgrade cmake, you may use your older cmake to compile CPU only build or use older CUDA version

project(tahoma CXX C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# cuda does not support c++23 yet
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)
set (CMAKE_COLOR_DIAGNOSTICS ON)  # colored output

# see https://github.com/bombela/backward-cpp?tab=readme-ov-file#libraries-to-read-the-debug-info
set (BACKWARD_HAS_DW on)   # sudo apt install libdw-dev
#set (BACKWARD_HAS_BFD on)
#set (BACKWARD_HAS_DWARF on)

# set options with default values
option(USE_CCACHE "Use ccache compiler cache (https://ccache.dev)" ON)
option(USE_CUDA "Use CUDA libtorch. Default is CPU only libtorch" OFF)
option(USE_ROCM "Use ROCM libtorch. Default is CPU only libtorch" OFF)
option(USE_PYTORCH "Use libtorch from the available pytorch installation
(i.e., torch.utils.cmake_prefix_path). This can be CPU/CUDA/ROCM build." OFF)
if (USE_CUDA AND USE_ROCM)
    message(FATAL_ERROR "USE_CUDA and USE_ROCM cannot be set at the same time")
endif()
option(COMPILE_TESTS "Compile tests under the ./tests directory" OFF)

set(PYTHON_EXECUTABLE python CACHE FILEPATH "Python executable name path. Used to find libtorch from pytorch installation")
find_package(Git QUIET)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -ggdb3 -O0")
    message(STATUS "CMAKE_BUILD_TYPE=Debug. Flags set to retain most debug info while disabling all optimizations")
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
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-cu124" CACHE PATH "libTorch CUDA path")
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA root path")
    endif()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
elseif (USE_ROCM)
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-rocm" CACHE PATH "libTorch ROCM path")
elseif (USE_PYTORCH)  # libtorch from pytorch installation
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE LIBTORCH_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
else()  # CPU only build by default
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-cpu" CACHE PATH "libTorch CPU path")
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
    "${LIBTORCH_PATH}/include"
    "${LIBTORCH_PATH}/include/torch/csrc/api/include"
    "${CMAKE_SOURCE_DIR}/libs/sentencepiece/src"
    "${CMAKE_SOURCE_DIR}/libs/spdlog/include"
    "${CMAKE_SOURCE_DIR}/libs/yaml-cpp/include"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/libs/backward-cpp"
    "${CMAKE_SOURCE_DIR}/libs/cnpy"
)


### SentencePiece library
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/sentencepiece)
set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)

### yaml-cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/yaml-cpp)
set(EXT_LIBS ${EXT_LIBS} yaml-cpp)

###
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/backward-cpp)


### Our code base : src

set(TAHOMA_SOURCES
    src/config.cpp
    src/data.cpp
    src/inference/decoder.cpp
    src/inference/predictor.cpp
    src/layer/transformer.cpp
    src/model.cpp
    src/model/transformer_lm.cpp
    src/model/transformer_nmt.cpp
    src/model/mt5.cpp
    src/serialize.cpp
    src/train/criterion.cpp
    src/train/loss_computer.cpp
    src/train/stats_counter.cpp
    src/train/trainer.cpp
    src/utils.cpp
    src/vocab.cpp

    # third party libs sources
    libs/cnpy/cnpy.cpp
)

add_library(tahoma_lib STATIC ${TAHOMA_SOURCES})
set_target_properties(tahoma_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_lib PUBLIC ${EXT_LIBS})
target_include_directories(tahoma_lib PUBLIC "${CMAKE_SOURCE_DIR}/include")


add_executable(tahoma_exe src/main.cpp)
set_target_properties(tahoma_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(tahoma_exe PROPERTIES OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_exe PUBLIC tahoma_lib Backward::Interface)

### Tests ###

if(COMPILE_TESTS)
    add_subdirectory(tests)
endif()
