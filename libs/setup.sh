#!/usr/bin/env bash
set -euo pipefail
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# versions
LIBTORCH_VERSION=2.4.1
CUDA_VERSION=cu124
ROCM_VERSION=rocm6.1
DEBUG=0

libtorch_base="https://download.pytorch.org/libtorch"
libtorch_version="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}"

cpu_url="${libtorch_base}/cpu/$libtorch_version%2Bcpu.zip"
cuda_url="${libtorch_base}/$CUDA_VERSION/$libtorch_version%2B${CUDA_VERSION}.zip"
rocm_url="${libtorch_base}/$ROCM_VERSION/$libtorch_version%2B${ROCM_VERSION}.zip"

download() {
    set -eEuo pipefail
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
        wget -q --show-progress -O $name.zip "${url}" && touch $name.zip._OK
    fi

    rm -rf $name  # remove incomplete downloads
    unzip -q $name.zip -d $name \
        && mv $name/libtorch/* $name/ \
        && rm -rf $name/libtorch

    # reverse engineered from libtorch build script
    # https://github.com/pytorch/builder/blob/6f3530cd/manywheel/build_libtorch.sh#L193
    CRC32=$(objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) $name/lib/libtorch_cpu.so)
    DEBUG_URL=$(dirname $url)/debug-$(basename $url .zip)-$CRC32.zip

    # check if _OK file exists, else download
    if [[ $DEBUG -eq 1 && ! -f $name-debug.zip._OK ]]; then
        rm -f $name-debug.zip $name-debug.zip._OK
        wget -q --show-progress -O $name-debug.zip "$DEBUG_URL" \
            && unzip -q -j $name-debug.zip -d $name/lib/ \
            && touch $name-debug.zip._OK
    fi
    touch $flag
}

main() {
    set -euo pipefail
    cd $MYDIR
    if [ $# -eq 0 ]; then
        selected_tasks=(cpu cuda)
    else
        selected_tasks=("$@")
    fi
    for tool in wget unzip; do
        if ! command -v $tool &> /dev/null; then
            echo "$tool could not be found. Please install and rerun."
            exit 2
        fi
    done

    for task in ${selected_tasks[@]}; do
        case $task in
            cpu) download libtorch-cpu "$cpu_url" ;;
            cuda) download libtorch-$CUDA_VERSION "$cuda_url" ;;
            rocm) download libtorch-$ROCM_VERSION "$rocm_url" ;;
            *) echo "ERROR: Unknown task: $task. Supported cpu cuda rocm" ;;
        esac
    done
    echo "Done."
}

main "$@"
