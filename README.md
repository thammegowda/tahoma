
## Setup

Currently working with CPU backend. Make sure to not have CUDA or torch+cuda libs in your environment (such as conda enviornment).


Step 1:
Download libtorch

```bash
lib/setup.sh
```

Step2:

Install cmake, gcc etc. Note: we are coding based on C++20 standard.

```bash
sudo apt install gcc cmake

# to get debug stacktrace on crash
sudo apt install libdw-dev libunwind-dev
```

> TODO: update the exact set of libs and lower bound on versions to support C++20



## Build

```bash
# create and cd into build dir
mkdir build && cd build

# build using cmake and make
cmake .. && make -j

# sample run
./rtgp workdir -c ../examples/nmt/transformer.toml

```

## VS Code Setup

* Install "C/C++ Extension Pack" by Microsoft
* Search for "C++ Select Intellisense Configuration" and point to your C++ installation.
