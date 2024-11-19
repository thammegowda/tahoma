#!/usr/bin/env bash

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUT_DIR=$MYDIR/mnist
mkdir -p $OUT_DIR
echo "Downloading MNIST data to $OUT_DIR"

# downloading from mirror: https://github.com/cvdfoundation/mnist
BASE_URL=https://storage.googleapis.com/cvdf-datasets/mnist

for name in {train,t10k}-{images-idx3,labels-idx1}-ubyte; do
  url=$BASE_URL/$name
  outfile=$OUT_DIR/$name
  if [ -s $outfile ]; then
    echo "$outfile already exists, skipping"
  else
    rm -f $outfile.gz.tmp $outfile.gz $outfile # remove any partial downloads
    wget $url.gz -O $outfile.gz.tmp && mv $outfile.gz.tmp $outfile.gz && gunzip -k $outfile.gz
  fi
done
