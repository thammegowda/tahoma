#!/usr/bin/env bash
# This script downloads models and data for regression tests
#
# Created by TG on 2025-02-08

set -euo pipefail

MYDIR=$(dirname $0)
MYDIR=$(cd $MYDIR && pwd)
MODELS=$MYDIR/models2

log() {
    echo -e "\e[32m$@\e[0m" >&2
}

AZBASE="https://tgwus2.blob.core.windows.net/data/cache/tahoma/models"

# Azure SAS tokens cache to avoid regenerating for each model
declare -A SAS_TOKS=()

get_SAS(){
    set -euo pipefail
    local url="$1"
    local container="$(echo ${url} | cut -d/ -f4)"
    local account="$(echo ${url} | cut -d/ -f3 | cut -d. -f1)"
    #local perms="delmoprw"
    #  https://learn.microsoft.com/en-us/rest/api/storageservices/create-user-delegation-sas#specify-permissions
    local perms="rl"

    local key="$account/$container"
    local sas="${SAS_TOKS[$key]:-}"
    if [[ -z "$sas" ]]; then
        local expiry=$(date -u -d "15 minutes" '+%Y-%m-%dT%H:%MZ')
        sas="$(az storage fs generate-sas --auth-mode login --as-user --output tsv --only-show-errors \
            --account-name $account --name $container --expiry $expiry --permissions $perms)"
        SAS_TOKS[$key]="$sas"
    fi
    echo "$sas"
}


# Download models
get_models(){
    set -euo pipefail
    for model in metricx-24-hybrid-large-v2p6; do
        if [[ ! -d $MODELS/$model || -z "$(ls $MODELS/$model)" ]]; then
            sas="$(get_SAS $AZBASE)"
            src="$AZBASE/${model}?${sas}"
            dest="$MODELS/$model/"
            log "Download: $src -> $dest"
            mkdir -p $dest
            azcopy cp --recursive "$src" "$dest"
        fi
    done
}


# step 1: get models
get_models

# step 2: get data
# get_data
