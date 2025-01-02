This code was tested to work on docker container `thammegowda.azurecr.io/marian/marian-cuda12.3-ubuntu22.04:latest`

> You may use this amulet config to obtain the exact environment: https://machinetranslation.visualstudio.com/Marian/_git/mt-detect?version=GBtg/metricx&path=/e2e-eval/amlt-metricx.yml


```bash
#URL: https://tgwus2.blob.core.windows.net/data/bins/tahoma/20241231-tahoma-0.0.1.tgz
TAHOMA_PKG="/mnt/tg/data/bins/tahoma/20241231-tahoma-0.0.1.tgz"
#URL: https://tgwus2.blob.core.windows.net/data/cache/tahoma/models/
MODELS_CACHE=/mnt/tg/data/cache/tahoma/models

# pick the model name
MODEL=metricx-24-hybrid-large-v2p6
#MODEL=metricx-24-hybrid-xl-v2p6
#MODEL=metricx-24-hybrid-xxl-v2p6


## setup:
TAHOMA_HOME=$HOME/.local/tahoma
export PATH=$TAHOMA_HOME/bin:$PATH
which tahoma >& /dev/null || {
  mkdir -p $TAHOMA_HOME
  tar xf $TAHOMA_PKG -C $TAHOMA_HOME;
}
which tahoma >& /dev/null || {
  echo "Error: tahoma not found" >&2
   exit 1
}

# workaround for memory fragmentation https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

batch_size=64   # for large
if [[ $MODEL =~ -xl ]]; then batch_size=16;
elif [[ $MODEL =~ -xxl ]]; then batch_size=8;

m="$MODELS_CACHE/$MODEL/model.npz"
v="$MODELS_CACHE/$MODEL/spiece.model"
max_length=1024

# maps STDIN TSV to STDOUT
score_cmd="tahoma predict --qe --fp16 --mini-batch $batch_size --maxi-batch 100 --max-length $max_length -m $m -v $v"
# using awk to flip the score sign since the lower is better
score_cmd+="| awk '{print -\$1}'"

# TSV of source and target sentences
pigz -dc input.tsv.gz | cut -f1,2 \
  | eval $score_cmd \
  | pv -abtrl -Wn -i 60 \
  | pigz -c > output.tsv.gz

```