#!/bin/bash
RUNS_DIR=/nfs/work/students/ml_coarsening/logs/train/runs
GRAPHS_DIR=$HOME/ml_coarsening/configs/data/graphs
TODAY_DATE=$(date +"%Y-%-m-%-d")


GRAPH_SET=all
MODEL_DIR=2000-01-01_shrink_factor_2_5
MODEL_BRANCH=cheap-model-fix


cd ~ || exit
cd ml_coarsening || exit
spack env activate ml_coarsening
source .venv/bin/activate
git checkout feat/new-inference
git pull
HYDRA_FULL_ERROR=1 srun uv run -m src.utils.direct_inference_setup \
  model_dir=$RUNS_DIR/$MODEL_DIR/ \
  data.graphs_file=$GRAPHS_DIR/$GRAPH_SET.txt \
  flags="'--c-min-accepted-shrink-factor=2.5'"



cd ~ || exit
cd mt-kahypar/ || exit
git checkout nikolai/$MODEL_BRANCH

EXP_JSON=$RUNS_DIR/$MODEL_DIR/experiment_sets/${GRAPH_SET}_experiment.json

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


MODEL_NAME=${MODEL_DIR#*_}
RESULTS_DIR="$HOME/hypergraph_partitioner/experiments/${TODAY_DATE}_${MODEL_NAME}"

# Destination directory
DEST_DIR="$RUNS_DIR/$MODEL_DIR/exp_results"

mkdir -p "$DEST_DIR"

cp "$RESULTS_DIR/ml_inside_uc.csv" "$DEST_DIR/${MODEL_NAME}-${GRAPH_SET}.csv"
cp "$RESULTS_DIR/ml_inside_c.csv" "$DEST_DIR/${MODEL_NAME}-${GRAPH_SET}-c.csv"
cp "$RESULTS_DIR/ml_inside_lp.csv" "$DEST_DIR/${MODEL_NAME}-${GRAPH_SET}-lp.csv"

echo "===================="
echo "results copied to $DEST_DIR"
echo "===================="


