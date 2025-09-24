#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
srun uv run -m src.train \
 trainer.enable_progress_bar=False \
 trainer=gpu \
 model.net.hidden_sizes="[48, 32, 16]" \
 model.net.with_batch_norm=False \
 model.optimizer.weight_decay=0.00001 \
 data.features_file=/nfs/home/schrape/ml_coarsening/configs/data/features/top_32_features_of_cost_0_reduced_for_mss_1_20.txt \
 data.graphs_file=/nfs/home/schrape/ml_coarsening/configs/data/graphs/mss_1_20.txt \
 data.data_amount=1000000 \
 paths.log_dir=/nfs/work/students/ml_coarsening/logs/