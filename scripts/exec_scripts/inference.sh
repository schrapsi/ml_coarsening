#!/bin/bash

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
HYDRA_FULL_ERROR=1 srun uv run -m src.inference