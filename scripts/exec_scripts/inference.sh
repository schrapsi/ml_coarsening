#!/bin/bash

#!/bin/bash
RUNS_DIR=/nfs/work/students/ml_coarsening/logs/train/runs
GRAPHS_DIR=$HOME/ml_coarsening/configs/data/graphs

GRAPH_SET=mss_1_20_eval
MODEL_DIR=2025-08-11_balanced_ram_761
EPOCH=018
MODEL_CLASS="MLCoarseningModule"

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
HYDRA_FULL_ERROR=1 srun uv run -m src.inference \
  ckpt_path=$RUNS_DIR/$MODEL_DIR/checkpoints/epoch_$EPOCH.ckpt \
  data.graphs_file=$GRAPHS_DIR/$GRAPH_SET.txt \
  model_class=$MODEL_CLASS