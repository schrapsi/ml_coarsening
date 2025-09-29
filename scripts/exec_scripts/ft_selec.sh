#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
srun uv run -m src.feature_selection \
 data.features_file=/nfs/home/schrape/ml_coarsening/configs/data/features/cost_0_reduced.txt \
 data.graphs_file=/nfs/home/schrape/ml_coarsening/configs/data/graphs/mss_1_20.txt \
 top_n=32