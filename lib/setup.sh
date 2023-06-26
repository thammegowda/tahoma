
#!/usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $MYDIR

# PyTorch version

libtorch_base="https://download.pytorch.org/libtorch"
libtorch_version="libtorch-cxx11-abi-shared-with-deps-2.0.1"

cpu_url="$libtorh_base/cpu/$libtorch_version%2Bcpu.zip"
cuda_url="$libtorh_base/cu118/$libtorch_version%2Bcu118.zip"
rocm_url="$libtorh_base/rocm5.4.2/$libtorch_version%2Brocm5.4.2.zip"

libtorch_url=$cpu_url
# Download libtorch
# testing cpu version for now
if [[ -f libtorch/lib/libtorch.so ]]; then
    echo "libtorch already exists"
else
    wget -O libtorch.cpu.zip $libtorch_url && unzip libtorch.cpu.zip && rm libtorch.cpu.zip
fi
