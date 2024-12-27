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
option(USE_STATIC_LIBS "Use static libraries" OFF)

set(TAHOMA_VERSION "0.0.1" CACHE STRING "Tahoma version")


set(PYTHON_EXECUTABLE python CACHE FILEPATH "Python executable name path. Used to find libtorch from pytorch installation")
find_package(Git QUIET)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -ggdb3 -O0")
    message(STATUS "CMAKE_BUILD_TYPE=Debug. Flags are set to retain the most debug info and disabled all optimizations")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    message(STATUS "CMAKE_BUILD_TYPE=Release. Flags set to optimize for performance")
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

if(USE_STATIC_LIBS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    message(STATUS "Using static libraries")
endif()

# run git submodule update --init --recursive
execute_process(
    COMMAND git submodule update --init --recursive --no-fetch
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

execute_process(
    COMMAND bash  ${CMAKE_SOURCE_DIR}/libs/setup.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

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
list(APPEND EXT_LIBS ${TORCH_LIBRARIES})
#message(STATUS "  >>>EXT_LIBS: ${EXT_LIBS}")

if(USE_CCACHE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -C")
endif()


set(_LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")
include_directories(
    "${LIBTORCH_PATH}/include"
    "${LIBTORCH_PATH}/include/torch/csrc/api/include"
    "${_LIBS_DIR}/sentencepiece/src"
    "${_LIBS_DIR}/spdlog/include"
    "${_LIBS_DIR}/yaml-cpp/include"
    "${_LIBS_DIR}/backward-cpp"
    "${_LIBS_DIR}/cnpy"
    "${_LIBS_DIR}/argparse"
    "${CMAKE_SOURCE_DIR}/src/include"
)


### SentencePiece library
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/sentencepiece)
set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)

### yaml-cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/yaml-cpp)
set(EXT_LIBS ${EXT_LIBS} yaml-cpp)

###
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/backward-cpp)


### Our code base : src/cpp
set(_SRC_DIR "src/cpp")
set(TAHOMA_SOURCES
    ${_SRC_DIR}/config.cpp
    ${_SRC_DIR}/data.cpp
    ${_SRC_DIR}/inference/decoder.cpp
    ${_SRC_DIR}/inference/predictor.cpp
    ${_SRC_DIR}/layer/transformer.cpp
    ${_SRC_DIR}/model.cpp
    ${_SRC_DIR}/model/transformer_lm.cpp
    ${_SRC_DIR}/model/transformer_nmt.cpp
    ${_SRC_DIR}/model/mt5.cpp
    ${_SRC_DIR}/serialize.cpp
    ${_SRC_DIR}/train/criterion.cpp
    ${_SRC_DIR}/train/loss_computer.cpp
    ${_SRC_DIR}/train/stats_counter.cpp
    ${_SRC_DIR}/train/trainer.cpp
    ${_SRC_DIR}/utils.cpp
    ${_SRC_DIR}/vocab.cpp
    # third party libs sources
    ${_LIBS_DIR}/cnpy/cnpy.cpp
)

add_library(tahoma_lib STATIC ${TAHOMA_SOURCES})
set_target_properties(tahoma_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "tahoma")
target_include_directories(tahoma_lib PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(tahoma_lib PUBLIC ${EXT_LIBS})


add_executable(tahoma_exe ${_SRC_DIR}/main.cpp)
set_target_properties(tahoma_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(tahoma_exe PROPERTIES OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_exe PUBLIC tahoma_lib Backward::Interface)


# Install tahoma_exe and the associated shared objects
install(TARGETS tahoma_exe tahoma_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/include/ DESTINATION include)

# Install other necessary files if any
#install(DIRECTORY ${LIBTORCH_PATH}/lib/ DESTINATION lib)
install(CODE "
    message(STATUS \"Installation complete; Lets fix the libs\" )
    execute_process(
        COMMAND python ${CMAKE_SOURCE_DIR}/scripts/post_build.py -a -b ${CMAKE_BINARY_DIR} -i ${CMAKE_INSTALL_PREFIX} -v ${TAHOMA_VERSION}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
")

### Tests ###

if(COMPILE_TESTS)
    add_subdirectory(tests)
endif()
