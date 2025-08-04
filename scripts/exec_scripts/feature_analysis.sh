#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull

# Set paths
DATA_DIR="/nfs/work/graph_benchmark_sets/ml_coarsening/extended_set/"
GRAPHS_FILE="/nfs/home/schrape/ml_coarsening/configs/data/graphs/2graphs.txt"
FEATURES_FILE="/nfs/home/schrape/ml_coarsening/configs/data/features/all.txt"
MODEL_PATH="/nfs/work/students/ml_coarsening/logs/train/runs/2025-06-12_vigilant_moose_257/checkpoints/epoch_602.ckpt"

# Run feature importance analysis
python -c "
from src.utils.feature_importance import analyze_feature_importance
analyze_feature_importance(
    data_dir='$DATA_DIR',
    graph_names=['wing', '4elt'],
    features_file='$FEATURES_FILE',
    model_path='$MODEL_PATH'
)
"