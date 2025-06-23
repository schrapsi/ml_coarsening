#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
srun uv run -m src.train \
 trainer.enable_progress_bar=False \
 trainer=gpu \
 data.features_file=/nfs/home/schrape/ml_coarsening/configs/data/features/cost_0_41.txt \
 data.graphs_file=/nfs/home/schrape/ml_coarsening/configs/data/graphs/mss_21_45.txt \
 paths.log_dir=/nfs/work/students/ml_coarsening/logs/