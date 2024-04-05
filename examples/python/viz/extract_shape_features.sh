#!/usr/bin/env bash
source /learnedsss/setpath.sh
# source ~/.bashrc

export PYTHONPATH=/nanogui/build/python:$PYTHONPATH
cd /learnedsss/pysrc
python extract_shape_features.py --mesh $1 --out_file $2 \
--albedo  $3 --sigma_t $4 --g $5 --eta $6