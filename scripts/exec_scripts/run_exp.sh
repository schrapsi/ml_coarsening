#!/bin/bash
RUNS_DIR=/nfs/work/students/ml_coarsening/logs/train/runs

GRAPH_SET=gnn_eval
MODEL_DIR=2000-01-01_gnn_run_001

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

EXP_JSON=$RUNS_DIR/$MODEL_DIR/predictions/${GRAPH_SET}_experiment.json

cd ~ || exit
spack env activate test
source hypergraph_partitioner/env.sh
source bachelor_thesis/venv/bin/activate
cd hypergraph_partitioner/experiments/ || exit
python ~/hypergraph_partitioner/setup_experiments.py $EXP_JSON -f
python ~/hypergraph_partitioner/experiments/execute_experiments.py $EXP_JSON

echo "===================="
echo "experiments completed"
echo "===================="


TODAY_DATE=$(date +"%Y-%-m-%-d")
MODEL_NAME=${MODEL_DIR#*_}
RESULTS_DIR="$HOME/hypergraph_partitioner/experiments/${TODAY_DATE}_${MODEL_NAME}"

# Destination directory
DEST_DIR="$RUNS_DIR/$MODEL_DIR/exp_results"

mkdir -p "$DEST_DIR"

cp "$RESULTS_DIR/mt_kahypar_ml.csv" "$DEST_DIR/${MODEL_NAME}-${GRAPH_SET}.csv"
cp "$RESULTS_DIR/mt_kahypar_ml_constrained.csv" "$DEST_DIR/${MODEL_NAME}-${GRAPH_SET}-c.csv"

echo "===================="
echo "results copied to $DEST_DIR"
echo "===================="


