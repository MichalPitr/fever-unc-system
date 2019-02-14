#!/bin/bash

export PYTHONPATH=src
export MODEL=unc

mkdir -pv fever2/pipeline/$MODEL/full/$1
mkdir -pv fever2/working/$MODEL/full/$1-working


if [ "$(wc -l < fever2/$1)" -eq "$(wc -l < fever2/pipeline/$MODEL/full/$1)" ];
then echo 'Skipping making predictions as this exists';
else
    python src/pipeline/fast_pipeline.py \
        --in-file fever2/$1 \
        --out-file fever2/pipeline/$MODEL/full.$1 \
        --working-dir fever2/working/$MODEL/full/$1-working
fi;