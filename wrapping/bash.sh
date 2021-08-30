#!/bin/bash -l

conda activate dlc

sbatch \
--verbose \
--output=output.txt \
--partition=p.gpu \
--gres=gpu:1 \
--qos=longrun \
--time=60-0 \
./run.sh

