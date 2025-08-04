#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
HYDRA_FULL_ERROR=1 srun uv run -m src.inference \
  ckpt_path=/nfs/work/students/ml_coarsening/logs/train/runs/2025-06-03_clever_grasshopper_761/checkpoints/epoch_022.ckpt