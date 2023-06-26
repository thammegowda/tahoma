
## Setup

We need `libtorch` in the compiler's environment, easiest way is to create conda environment and install pytorch.


```bash
conda create -n rtgpp python=3.9
conda activate rtgpp

# install pytorch (cpu only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# this is the path
ls $(python -c 'import torch;print(torch.utils.cmake_prefix_path)')

```
