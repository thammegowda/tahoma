#!/usr/env/bin bash
set -eux

mydir=$(dirname $0)
root=$(realpath $mydir/../..)

# Download the data
tools=(mtdata spm_train)

# add spm_train to path
export PATH=$PATH:$root/build/libs/sentencepiece/src 

SPM_ARGS="--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --character_coverage=0.9995 --byte_fallback"

for tool in ${tools[@]}; do
    if ! command -v $tool &> /dev/null; then
        echo "$tool could not be found. Please install and/or add to PATH first."
        exit 1
    fi
done

#### English-Kannada ####
out_dir=$mydir/eng-kan
mkdir -p $out_dir
[[ -f $out_dir/._OK ]] || {
    # mtdata echo Statmt-pmindia-1-eng-kan | awk 'NF>0' > $out_dir/train.eng-kan.tsv
    mtdata echo AI4Bharath-samananthar-0.2-eng-kan | awk 'NF>0' > $out_dir/train.eng-kan.tsv
    mtdata echo Flores-flores200_dev-1-eng-kan | awk 'NF>0' > $out_dir/dev.eng-kan.tsv
    mtdata echo Flores-flores200_devtest-1-eng-kan | awk 'NF>0' > $out_dir/test.eng-kan.tsv
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

###########  English-German  ###########
out_dir=$mydir/eng-deu
mkdir -p $out_dir
recipe_id=vaswani_etal_2017_ende
[[ -f $out_dir/._OK ]] || {
    mtdata echo Statmt-europarl-9-deu-eng | awk 'NF>0' > $out_dir/train.deu-eng.tsv
    mtdata echo Statmt-newstest_deen-2020-deu-eng | awk 'NF>0' > $out_dir/dev.deu-eng.tsv
    mtdata echo Statmt-newstest_deen-2021-deu-eng | awk 'NF>0' > $out_dir/test.deu-eng.tsv
    for split in train dev test; do
        cut -f1 $out_dir/$split.deu-eng.tsv > $out_dir/$split.deu
        cut -f2 $out_dir/$split.deu-eng.tsv > $out_dir/$split.eng
    done
    touch $out_dir/._OK
}

[[ -f $out_dir/vocab.joint.8k.model ]] || {    
    echo "Training SentencePiece model... This could take a while."
    cat $out_dir/train.deu $out_dir/train.eng | shuf | head -n 1000000 > $out_dir/combined.train.txt
    spm_train --input $out_dir/combined.train.txt --model_prefix=$out_dir/vocab.joint.8k --vocab_size=8000 $SPM_ARGS
}
