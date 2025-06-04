#!/bin/bash
RUNS_DIR=/nfs/work/students/ml_coarsening/logs/train/runs
MODEL_DIR=2025-06-03_optimistic_newt_320
EPOCH=014

cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git pull
HYDRA_FULL_ERROR=1 srun uv run -m src.inference \
  ckpt_path=$RUNS_DIR/$MODEL_DIR/checkpoints/epoch_$EPOCH.ckpt

echo "====================="
echo "graph predictions done"
echo "====================="

cd ~ || exit
cd mt-kahypar/build/ || exit
spack env activate kahypar
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make clean
make MtKaHyPar -j
spack env deactivate

echo "===================="
echo "building file done"
echo "===================="

cd ~ || exit
spack env activate test
source hypergraph_partitioner/env.sh
source bachelor_thesis/venv/bin/activate
cd hypergraph_partitioner/experiments/ || exit
python ~/hypergraph_partitioner/setup_experiments.py $RUNS_DIR/$MODEL_DIR/experiment.json -f
python ~/hypergraph_partitioner/experiments/execute_experiments.py $RUNS_DIR/$MODEL_DIR/experiment.json


