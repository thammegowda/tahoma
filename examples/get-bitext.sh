#!/usr/bin/env bash
# Download sample datasets and create SentencePiece models

# Created by Thamme "TG" Gowda, circa 2024-10
set -eu

mydir=$(dirname $0)
root=$(realpath $mydir/..)
data_dir=$root/data

# Download the data
tools=(mtdata spm_train)
# have mtdata 0.4.2 or higher. older version had a bug in echo
tasks=(eng-kan eng-deu)

if [ $# -eq 0 ]; then
    selected_tasks=("${tasks[@]}")
else
    selected_tasks=("$@")
fi

echo "Selected tasks: ${selected_tasks[@]}"

# add spm_train to path
spm_build_path=$root/build/libs/sentencepiece/src
[[ -d $spm_build_path ]] || {
    echo "Missing $spm_build_path; Please build the project first. "
    exit 1
}

export PATH=$spm_build_path:$PATH

SPM_ARGS="--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --character_coverage=0.9995 --byte_fallback"

for tool in ${tools[@]}; do
    if ! command -v $tool &> /dev/null; then
        echo "$tool could not be found. Please install and/or add to PATH first."
        exit 1
    fi
done


clean-segs(){
    sed 's/\r//g' \
    | awk -F'\t' '{
        all_ok = 1
        for (i = 1; i <= NF; i++) {
            if ($i == "") {
                all_ok = 0
                break
            }
        }
        if (all_ok) {
            print
        }
    }'
}


#### English-Kannada ####
if [[ " ${selected_tasks[@]} " =~ " eng-kan " ]]; then
    out_dir=$data_dir/eng-kan
    mkdir -p $out_dir
    [[ -f $out_dir/._OK ]] || {
        # mtdata echo Statmt-pmindia-1-eng-kan | awk 'NF>0' > $out_dir/train.eng-kan.tsv
        mtdata echo AI4Bharath-samananthar-0.2-eng-kan | clean-segs > $out_dir/train.eng-kan.tsv
        mtdata echo Flores-flores200_dev-1-eng-kan | clean-segs > $out_dir/dev.eng-kan.tsv
        mtdata echo Flores-flores200_devtest-1-eng-kan | clean-segs > $out_dir/test.eng-kan.tsv
        for split in train dev test; do
            cut -f1 $out_dir/$split.eng-kan.tsv > $out_dir/$split.eng
            cut -f2 $out_dir/$split.eng-kan.tsv > $out_dir/$split.kan
        done
        touch $out_dir/._OK
    }
    [[ -s $out_dir/vocab.joint.8k.model ]] || {
        echo "Sampling 1M lines for vocab"
        cat $out_dir/train.eng $out_dir/train.kan | shuf | head -n 1000000 > $out_dir/combined.train.txt
        spm_train --input $out_dir/combined.train.txt --model_prefix=$out_dir/vocab.joint.8k --vocab_size=8000 $SPM_ARGS
    }
fi


###########  English-German  ###########
if [[ " ${selected_tasks[@]} " =~ " eng-deu " ]]; then
    out_dir=$data_dir/eng-deu
    mkdir -p $out_dir
    recipe_id=vaswani_etal_2017_ende
    [[ -f $out_dir/._OK ]] || {
        mtdata echo Statmt-europarl-9-deu-eng | clean-segs > $out_dir/train.deu-eng.tsv
        mtdata echo Statmt-newstest_deen-2020-deu-eng | clean-segs > $out_dir/dev.deu-eng.tsv
        mtdata echo Statmt-newstest_deen-2021-deu-eng | clean-segs > $out_dir/test.deu-eng.tsv
        for split in train dev test; do
            cut -f1 $out_dir/$split.deu-eng.tsv > $out_dir/$split.deu
            cut -f2 $out_dir/$split.deu-eng.tsv > $out_dir/$split.eng
        done
        touch $out_dir/._OK
    }

    [[ -f $out_dir/vocab.joint.8k.model ]] || {
        echo "Training SentencePiece model... This could take a while."
        cat $out_dir/train.deu $out_dir/train.eng | shuf | head -n 1000000  > $out_dir/combined.train.txt
        spm_train --input $out_dir/combined.train.txt --model_prefix=$out_dir/vocab.joint.8k --vocab_size=8000 $SPM_ARGS
    }
fi
