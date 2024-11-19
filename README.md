# Tahoma


## Setup


Step 0: Download source code

```bash

git submodule update --init --recursive
```


Step 1: Download libtorch

```bash
libs/setup.sh
```

Step2:

Install cmake, gcc etc. Note: we are coding based on C++23 standard.

```bash
sudo apt install gcc-12 g++-12 cmake build-essential pkg-config libgoogle-perftools-dev

# Using backwards-cpp to get debug stacktrace on crashes/exceptions
# And we need these for backwards-cpp to work
sudo apt install libdw-dev libunwind-dev
```

> TODO: update the exact set of libs and lower bound on versions to support C++20


## Build

```bash

# build using cmake and default compiler
cmake -B build .
# build using your specifed compiler. Eg. clang-17
CC=clang-17 CXX=clang++-17 cmake . -B build
cmake . -B build -DCMAKE_CC_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++17 .
# compile
make -j

# sample run
./rtgp workdir -c ../examples/nmt/transformer.toml

```

## Compile and Run Tests

Tests are powered by Cmake and CTest. To enable set `-DCOMPILE_TESTS=on`

```bash
cmake -B build -DCCOMPILE_TESTS=on
cmake --build build -j
# produces an executable at build/tests/tahoma-tests
# you may directly run the executable e.g. for debugging via gdb
# but recommeneded way is ctest

ctest -V --test-dir build
```

## VS Code Setup

* Install "C/C++ Extension Pack" by Microsoft
* Search for "C++ Select Intellisense Configuration" and point to your C++ installation.

## Build LibTorch from Source

This project relies on libTorch. It'd be useful to know how to build libTorch


## Developer Notes

. include/ <-- Header files (.h)
. src/   <--  All implementations (.cpp)

Header files and implementations should mirror each other.

Layer is a lower level code, model is higher level abstraction. Model uses one or more layers.


```
include/layer.h             <-- common code for all layers, e.g. base classes
include/layer/transformer.h <-- transformer layer specific code
include/layer/rnn.h         <-- rnn layer specific coide

include/model.h                  <-- common code  e.g. base classes
include/model/transformer_lm.h   <-- LM
include/model/transformer_nmt.h  <-- NMT

```

