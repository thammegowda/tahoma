# Tahoma

Tahoma is a project for NLP/NMT/LLM experimentation. It is written in C++ to
overcome a shortcomings of Python ecosystem. This project is inspired by Marian NMT,
however, it differs from Marian in a very significant way. The backend of this
experimentation project is powered by libTorch! While MarianNMT does "all by own",
Tahoma builds upon libTorch; this is intentional. This project is started in 2024,
hence we use latest C++ standard (C++23) which makes a lot of things simpler. 


## Setup

Step 0: Download source code

```bash
git clone --recurse-submodules $URL

# Option 2: if you'd already cloned without submodule
git clone $URL
cd tahoma
git submodule update --init --recursive
```


Step 1: Download libtorch

```bash
libs/setup.sh
```

Step2:

Install cmake, gcc etc. Note: we are coding based on C++23 standard.

```bash
sudo apt install gcc-12 g++-12 cmake build-essential pkg-config libgoogle-perftools-dev libdw-dev libdwarf-dev
```


## Build

```bash
# build using cmake and default compiler
cmake . -B build -DUSE_CUDA=on -DCOMPILE_TESTS=on
cmake --build build -j

build/tahoma -h 
```

## Metrics

### Metricx

Step 1. Download model and save it as .npz

```bash
MODEL=metricx-24-hybrid-large-v2p6
TOKENIZER=mt5-base

scripts/torch2npz.py -m google/$MODEL -o tmp/$MODEL
scripts/torch2npz.py -m google/$TOKENIZER -o tmp/$TOKENIZER
cp tmp/$TOKENIZER/spiece.model tmp/$MODEL/
```
torch2npz.py requires python environment with huggingface transformers. 
Alternatively, if you have access, you may obtain converted models from blob storage @ `https://tgwus2.blob.core.windows.net/data/cache/tahoma/models/`

Step 2. Run 
```bash
CUDA_VISIBLE_DEVICES=0
MODEL=metricx-24-hybrid-large-v2p6
echo -e "source\tcandidate\treference" | build/tahoma predict -m tmp/$MODEL/model.npz -v tmp/$MODEL/spiece.model
echo -e "source\tcandidate" | build/tahoma predict -m tmp/$MODEL/model.npz -v tmp/$MODEL/spiece.model --qe
```


## Run Tests

Tests are powered by Cmake and CTest. To enable set `-DCOMPILE_TESTS=on`

```bash
cmake -B build -DCOMPILE_TESTS=on -DUSE_CUDA=on
cmake --build build -j

# produces an executable at build/tests/tahoma-tests
# you may directly run the executable e.g. for debugging via gdb
 but recommeneded way is ctest
ctest -V --test-dir build/tests
```


## VS Code Setup

* Install "C/C++ Extension Pack" by Microsoft
* Search for "C++ Select Intellisense Configuration" and point to your C++ installation.

## Build LibTorch from Source

This project relies on libTorch. It'd be useful to know how to build libTorch


## Developer Notes

```
include/   <-- Header files (.h)
src/      <--  All implementations (.cpp)
```

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

