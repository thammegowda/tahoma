#!/usr/bin/env bash
set -eu
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# versions
LIBTORCH_VERSION=2.1.0
CUDA_VERSION=cu121
ROCM_VERSION=rocm5.6

libtorch_base="https://download.pytorch.org/libtorch"
libtorch_version="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}"

cpu_url="${libtorch_base}/cpu/$libtorch_version%2Bcpu.zip"
cuda_url="${libtorch_base}/$CUDA_VERSION/$libtorch_version%2B${CUDA_VERSION}.zip"
rocm_url="${libtorch_base}/$ROCM_VERSION/$libtorch_version%2B${ROCM_VERSION}.zip"

download() {
    local name="$1"
    local url="$2"
    local flag=$name/_SETUP_OK

    if [[ -f $flag ]]; then
        echo "$name already exists. Skipping. 'rm $flag' to force redownload."
        return 0
    fi
    if [[ -f $name.zip && -f $name.zip._OK ]]; then
        echo "$name already exists"
    else
        rm -f $name.zip $name.zip._OK
        wget -O $name.zip "${url}" && touch $name.zip._OK
    fi

    rm -rf $name  # remove incomplete downloads
    unzip $name.zip -d $name \
        && mv $name/libtorch/* $name/ \
        && rm -rf $name/libtorch \
        && touch $flag
}

main() {
    cd $MYDIR
    for tool in wget unzip; do
        if ! command -v $tool &> /dev/null; then
            echo "$tool could not be found. Please install and rerun."
            exit 2
        fi
    done
    download libtorch-cpu "$cpu_url"
    download libtorch-$CUDA_VERSION "$cuda_url"
    #download libtorch-$ROCM_VERSION "$rocm_url"
}

main "$@"