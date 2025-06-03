#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
#srun uv run -m src.train trainer.enable_progress_bar=False data=mss_1_20
srun uv run -m src.train trainer.enable_progress_bar=False data=mss_1_20 data.features_file=${paths.root_dir}/configs/data/features/cost_0_half.txt